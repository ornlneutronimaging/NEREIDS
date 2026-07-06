//! ENDF-6 File 2 resonance parameter parser.
//!
//! Parses the fixed-width 80-character ENDF-6 format to extract resolved
//! resonance region (RRR) parameters.
//!
//! ## ENDF-6 Line Format
//! Each line is exactly 80 characters:
//! - Cols 1-11:  Field 1 (floating point or integer)
//! - Cols 12-22: Field 2
//! - Cols 23-33: Field 3
//! - Cols 34-44: Field 4
//! - Cols 45-55: Field 5
//! - Cols 56-66: Field 6
//! - Cols 67-70: MAT number
//! - Cols 71-72: MF (file number)
//! - Cols 73-75: MT (section number)
//! - Cols 76-80: Line sequence
//!
//! ## SAMMY Reference
//! - SAMMY manual Section 9 (ENDF-6 format)
//! - SAMMY source: `sammy/src/endf/` module

use crate::resonance::*;
use nereids_core::elements::isotope_from_za;

/// ENDF radius unit conversion factor.
///
/// ENDF-6 Formats Manual §2.1: "AP, APE, APT are in units of 10⁻¹² cm."
/// Physics convention: 1 fm = 10⁻¹³ cm, so 10⁻¹² cm = 10 fm.
/// All ENDF radii (AP, APL, APE, APT, AP(E) tables) are multiplied by this
/// factor at parse time so that downstream physics uses true femtometers.
///
/// SAMMY applies the identical ×10 conversion when reading ENDF:
///   `FillSammyRmatrixFromRMat.cpp` line 422: `newChannel->getApe() * 10.0`
const ENDF_RADIUS_TO_FM: f64 = 10.0;

/// Parse ENDF-6 File 2 resonance parameters from raw ENDF text.
///
/// Extracts all MF=2, MT=151 lines and parses the resolved resonance region.
///
/// # Arguments
/// * `endf_text` — Full ENDF file contents as a string.
///
/// # Returns
/// `ResonanceData` containing all parsed resonance parameters.
pub fn parse_endf_file2(endf_text: &str) -> Result<ResonanceData, EndfParseError> {
    // Extract MF=2, MT=151 lines (resonance parameters).
    let lines: Vec<&str> = endf_text
        .lines()
        .filter(|line| {
            if line.len() < 75 {
                return false;
            }
            let mf = line[70..72].trim();
            let mt = line[72..75].trim();
            mf == "2" && mt == "151"
        })
        .collect();

    if lines.is_empty() {
        return Err(EndfParseError::MissingSection(
            "No MF=2, MT=151 data found".to_string(),
        ));
    }

    let mut pos = 0;

    // HEAD record: ZA, AWR, 0, 0, NIS, 0
    let head = parse_cont(&lines, &mut pos)?;
    let za = head.c1 as u32;
    let awr = head.c2;
    let nis = checked_count(head.n1, "NIS")?; // number of isotopes (usually 1)

    // ENDF-6 §2.1 requires NIS >= 1 for a valid resonance evaluation. NIS=0
    // would leave the parser with no isotope subsection to read and fall
    // through to a confusing "unconsumed data lines" downstream failure;
    // reject up-front with a clear message.
    if nis == 0 {
        return Err(EndfParseError::UnsupportedFormat(
            "MF=2 NIS=0: no isotopes declared. ENDF-6 §2.1 requires NIS >= 1 \
             for a valid resonance evaluation."
                .into(),
        ));
    }
    // ENDF-6 §2.1: a material with NIS>1 contains multiple isotope subsections,
    // each carrying its own ZAI, ABN, LFW, and NER ranges (e.g. natural-element
    // evaluations such as nat-C with ZAI={6012,6013}). The reference reader
    // OpenScale (File2.cpp:71-87) stores these in a Vec<ResonanceIsotope>
    // tagged with per-isotope ABN, and downstream physics combines the per-
    // isotope cross sections with that ABN weighting. NEREIDS's `ResonanceData`
    // has no per-isotope discriminator and no abundance field; silently
    // flattening multi-isotope subsections into one flat range list would
    // discard ZAI/ABN and produce abundance-blind cross sections. Reject
    // the case explicitly until a proper multi-isotope container is wired
    // through. Single-isotope (NIS=1) ENDF evaluations remain fully supported.
    if nis > 1 {
        return Err(EndfParseError::UnsupportedFormat(format!(
            "MF=2 NIS={nis} > 1 multi-isotope materials are not supported. \
             Each NEREIDS ResonanceData represents a single isotope. \
             For multi-isotope ENDF evaluations, split the material into per-isotope \
             files or use sammy_to_resonance_data_multi from nereids_endf::sammy."
        )));
    }

    let isotope = isotope_from_za(za)?;
    let mut all_ranges = Vec::new();

    for _ in 0..nis {
        // Isotope CONT: ZAI, ABN, 0, LFW, NER, 0
        let iso_cont = parse_cont(&lines, &mut pos)?;
        let _zai = iso_cont.c1 as u32;
        let _abn = iso_cont.c2; // abundance
        let lfw = iso_cont.l2; // fission width flag (LFW=1 → energy-dependent fission widths in URR)
        let ner = checked_count(iso_cont.n1, "NER")?; // number of energy ranges

        for _ in 0..ner {
            // Range CONT: EL, EH, LRU, LRF, NRO, NAPS
            let range_cont = parse_cont(&lines, &mut pos)?;
            let energy_low = range_cont.c1;
            let energy_high = range_cont.c2;
            let lru = range_cont.l1; // 1=resolved, 2=unresolved
            let lrf = range_cont.l2; // resonance formalism

            if lru == 2 {
                // Unresolved resonance region (LRU=2).
                // URR uses average level-spacing/width parameters; cross-sections are
                // computed via Hauser-Feshbach in nereids_physics::urr.
                //
                // Unsupported sub-formats are skipped gracefully so that the resolved
                // resonance ranges in the same evaluation remain accessible.
                // Hard errors are reserved for genuinely malformed records.

                // NRO=range_cont.n1: if non-zero a TAB1 AP(E) record immediately follows
                // the range CONT before the URR SPI/AP/NLS CONT.
                // ENDF-6 §2.2.2.
                let nro_urr = range_cont.n1;
                let naps_urr = range_cont.n2; // scattering radius calculation flag
                let ap_table_urr = if nro_urr != 0 {
                    let mut tab = parse_tab1(&lines, &mut pos)?;
                    // AP(E) y-values are in 10⁻¹² cm; convert to fm.
                    for pt in &mut tab.points {
                        pt.1 *= ENDF_RADIUS_TO_FM;
                    }
                    Some(tab)
                } else {
                    None
                };

                // LRF=1 and LRF=2 with LFW=0 are fully supported.
                // Unsupported combinations are skipped so that the resolved
                // resonance ranges in the same evaluation remain accessible.
                if lrf != 1 && lrf != 2 {
                    skip_urr_body(&lines, &mut pos)?;
                    continue;
                }

                // LFW=1 (energy-dependent fission widths):
                // LRF=2: record layout identical to LFW=0 — handled below.
                // LRF=1: different record layout (shared energy LIST).
                // Reference: ENDF-6 §2.2.2.1 Case B; SAMMY File2Unres.f90.
                if lfw != 0 && lrf == 1 {
                    let mut urr_ctx = RangeParseContext {
                        lines: &lines,
                        pos: &mut pos,
                        energy_low,
                        energy_high,
                        naps: naps_urr,
                        ap_table: ap_table_urr,
                    };
                    let urr_range = parse_urr_lfw1_lrf1(&mut urr_ctx)?;
                    all_ranges.push(urr_range);
                    continue;
                }

                let mut urr_ctx = RangeParseContext {
                    lines: &lines,
                    pos: &mut pos,
                    energy_low,
                    energy_high,
                    naps: naps_urr,
                    ap_table: ap_table_urr,
                };
                let urr_range = parse_urr_range(&mut urr_ctx, lrf)?;
                all_ranges.push(urr_range);
                continue;
            }

            if lru == 0 {
                // LRU=0: scattering-radius-only range (no resonance parameters).
                // ENDF-6 §2.2: after the range CONT (and optional TAB1 if NRO!=0),
                // a single CONT record follows: [SPI, AP, 0, 0, NLS=0, 0].
                // We consume the NRO TAB1 if present, then the CONT, and skip.
                let nro_lru0 = range_cont.n1;
                if nro_lru0 != 0 {
                    // TAB1 AP(E) already follows; consume it.
                    let _tab = parse_tab1(&lines, &mut pos)?;
                }
                // CONT: SPI, AP, 0, 0, NLS=0, 0
                // Validate NLS=0 (#123): a non-zero NLS in an LRU=0 range is
                // malformed and would cause the parser to look for L-groups that
                // don't exist, misaligning the cursor for subsequent ranges.
                let spi_cont = parse_cont(&lines, &mut pos)?;
                // ENDF-6 §2.2: the SPI/AP CONT is [SPI, AP, 0, 0, NLS=0, 0].
                // Validate that L1 and L2 are both zero — non-zero values
                // indicate a malformed or mis-identified record.
                if spi_cont.l1 != 0 {
                    return Err(EndfParseError::UnsupportedFormat(format!(
                        "LRU=0 range: L1={} in SPI/AP CONT record must be 0",
                        spi_cont.l1
                    )));
                }
                if spi_cont.l2 != 0 {
                    return Err(EndfParseError::UnsupportedFormat(format!(
                        "LRU=0 range: L2={} in SPI/AP CONT record must be 0",
                        spi_cont.l2
                    )));
                }
                if spi_cont.n1 != 0 {
                    return Err(EndfParseError::UnsupportedFormat(format!(
                        "LRU=0 range: NLS={} in SPI/AP CONT record must be 0 \
                         (scattering-radius-only ranges have no L-groups)",
                        spi_cont.n1
                    )));
                }
                if spi_cont.n2 != 0 {
                    return Err(EndfParseError::UnsupportedFormat(format!(
                        "LRU=0 range: N2={} in SPI/AP CONT record must be 0",
                        spi_cont.n2
                    )));
                }
                continue;
            }

            if lru != 1 {
                return Err(EndfParseError::UnsupportedFormat(format!(
                    "LRU={} not supported (expected 0=scattering-radius-only, 1=resolved, or 2=unresolved)",
                    lru
                )));
            }

            let nro = range_cont.n1; // energy-dependent scattering radius flag
            let naps = range_cont.n2; // scattering radius calculation flag

            // If NRO != 0, a TAB1 record immediately follows giving AP(E).
            // Parse and store it; scattering_radius_at(E) will interpolate it
            // at each energy point.  Reference: ENDF-6 §2.2.1; SAMMY mlb/mmlb1.f90.
            let ap_table = if nro != 0 {
                let mut tab = parse_tab1(&lines, &mut pos)?;
                // AP(E) y-values are in 10⁻¹² cm; convert to fm.
                for pt in &mut tab.points {
                    pt.1 *= ENDF_RADIUS_TO_FM;
                }
                Some(tab)
            } else {
                None
            };

            // ENDF-6 Formats Manual: LRF values for resolved resonance region
            // LRF=1: Single-Level Breit-Wigner (SLBW)
            // LRF=2: Multi-Level Breit-Wigner (MLBW)
            // LRF=3: Reich-Moore
            // LRF=4: Adler-Adler (deprecated, not supported)
            // LRF=7: R-Matrix Limited (general)
            let formalism = match lrf {
                1 => ResonanceFormalism::SLBW,
                2 => ResonanceFormalism::MLBW,
                3 => ResonanceFormalism::ReichMoore,
                7 => ResonanceFormalism::RMatrixLimited,
                _ => {
                    return Err(EndfParseError::UnsupportedFormat(format!(
                        "LRF={} not yet supported",
                        lrf
                    )));
                }
            };

            let mut ctx = RangeParseContext {
                lines: &lines,
                pos: &mut pos,
                energy_low,
                energy_high,
                naps,
                ap_table,
            };
            let range = match formalism {
                ResonanceFormalism::MLBW | ResonanceFormalism::SLBW => {
                    parse_bw_range(&mut ctx, formalism)?
                }
                ResonanceFormalism::ReichMoore => parse_reich_moore_range(&mut ctx)?,
                ResonanceFormalism::RMatrixLimited => parse_rmatrix_limited_range(&mut ctx, awr)?,
                ResonanceFormalism::Unresolved => {
                    // Unreachable: Unresolved is only assigned in the LRU=2 branch above.
                    unreachable!("Unresolved formalism should not appear in LRU=1 dispatch");
                }
            };
            all_ranges.push(range);
        }
    }

    // Multi-MAT detection (#114, #123): since `lines` is pre-filtered to
    // MF=2/MT=151, any unconsumed lines are definitively from another material.
    // The previous character-based heuristic for distinguishing "real data" from
    // SEND/FEND/MEND/TEND records was overly complex — those section-end records
    // use different MF/MT codes and are already excluded by the filter above.
    //
    // Assumption: trailing whitespace-only lines that happen to pass the MF/MT
    // filter (i.e. have " 2" at cols 70-72 and "151" at cols 72-75) would also
    // trigger this check.  In practice, ENDF files do not contain such lines —
    // trailing blanks either lack the MF/MT fields entirely or use MF=0/MT=0,
    // both of which are excluded by the filter in `parse_endf_file2`.
    if pos < lines.len() {
        return Err(EndfParseError::UnsupportedFormat(
            "Multiple materials detected in MF=2/MT=151: unconsumed data lines \
             remain after parsing the first material. Multi-MAT files are not \
             supported; split the file into single-material ENDF files."
                .to_string(),
        ));
    }

    Ok(ResonanceData {
        isotope,
        za,
        awr,
        ranges: all_ranges,
    })
}

/// Shared context for ENDF range parsers.
///
/// Groups the file-position state (`lines`, `pos`) with the fields from the
/// range CONT record that every range parser needs, eliminating long argument
/// lists.
struct RangeParseContext<'a> {
    lines: &'a [&'a str],
    pos: &'a mut usize,
    energy_low: f64,
    energy_high: f64,
    naps: i32,
    ap_table: Option<Tab1>,
}

/// Parse a Breit-Wigner (SLBW or MLBW) resolved resonance range.
///
/// ENDF-6 File 2, LRF=1 (SLBW) / LRF=2 (MLBW):
/// - CONT: SPI, AP, 0, 0, NLS, 0
/// - For each L-value:
///   - CONT: AWRI, 0.0, L, 0, 6*NRS, NRS
///   - LIST: NRS resonances, each 6 values: ER, AJ, GT, GN, GG, GF
///
/// Reference: ENDF-6 Formats Manual Section 2.2.1.1
fn parse_bw_range(
    ctx: &mut RangeParseContext<'_>,
    formalism: ResonanceFormalism,
) -> Result<ResonanceRange, EndfParseError> {
    // CONT: SPI, AP, 0, 0, NLS, 0
    // ENDF AP is in 10⁻¹² cm; convert to fm (×10).
    let cont = parse_cont(ctx.lines, ctx.pos)?;
    let target_spin = cont.c1;
    let scattering_radius = cont.c2 * ENDF_RADIUS_TO_FM;
    let nls = checked_count(cont.n1, "NLS")?; // number of L-values

    let mut l_groups = Vec::with_capacity(nls);

    for _ in 0..nls {
        // CONT: AWRI, QX, L, LRX, 6*NRS, NRS
        let l_cont = parse_cont(ctx.lines, ctx.pos)?;
        let awr_l = l_cont.c1;
        let qx = l_cont.c2; // Q-value for competitive width (eV)
        // Validate L is non-negative (#123): negative L1 wraps to a huge u32.
        if l_cont.l1 < 0 {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "BW range: negative L={}",
                l_cont.l1
            )));
        }
        let l_val = l_cont.l1 as u32;
        let lrx = l_cont.l2; // competitive width flag
        let n1 = checked_count(l_cont.n1, "N1")?; // should be 6*NRS
        let nrs = checked_count(l_cont.n2, "NRS")?; // number of resonances

        // Validate N1 == 6*NRS (#123): a mismatch means the record is malformed
        // and reading N1 values would over-/under-consume lines.
        if n1 != 6 * nrs {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "BW range L={l_val}: N1={n1} != 6*NRS={} (NRS={nrs})",
                6 * nrs
            )));
        }

        let mut resonances = Vec::with_capacity(nrs);

        // Each resonance is 6 values on one line (or spanning lines).
        // In ENDF format, LIST records pack 6 values per line.
        let total_values = nrs * 6;
        let values = parse_list_values(ctx.lines, ctx.pos, total_values)?;

        for i in 0..nrs {
            let base = i * 6;
            resonances.push(Resonance {
                energy: values[base],  // ER
                j: values[base + 1],   // AJ
                gn: values[base + 3],  // GN (neutron width)
                gg: values[base + 4],  // GG (gamma width)
                gfa: values[base + 5], // GF (fission width)
                gfb: 0.0,              // Not used in BW
                                       // Note: values[base+2] is GT (total width) — derived, not stored
            });
        }

        l_groups.push(LGroup {
            l: l_val,
            awr: awr_l,
            apl: 0.0, // Not in BW format
            qx,
            lrx,
            resonances,
        });
    }

    Ok(ResonanceRange {
        energy_low: ctx.energy_low,
        energy_high: ctx.energy_high,
        resolved: true,
        formalism,
        target_spin,
        scattering_radius,
        naps: ctx.naps,
        ap_table: ctx.ap_table.take(),
        l_groups,
        rml: None,
        urr: None,
        r_external: vec![],
    })
}

/// Parse a Reich-Moore resolved resonance range.
///
/// ENDF-6 File 2, LRF=2:
/// ENDF-6 File 2, LRF=3 (Reich-Moore):
/// - CONT: SPI, AP, 0, 0, NLS, 0
/// - For each L-value:
///   - CONT: AWRI, APL, L, 0, 6*NRS, NRS
///   - LIST: NRS resonances, each 6 values: ER, AJ, GN, GG, GFA, GFB
///
/// Reference: ENDF-6 Formats Manual Section 2.2.1.3
/// Reference: SAMMY manual Section 2 (R-matrix theory)
fn parse_reich_moore_range(
    ctx: &mut RangeParseContext<'_>,
) -> Result<ResonanceRange, EndfParseError> {
    // CONT: SPI, AP, 0, 0, NLS, 0
    // ENDF AP is in 10⁻¹² cm; convert to fm (×10).
    let cont = parse_cont(ctx.lines, ctx.pos)?;
    let target_spin = cont.c1;
    let scattering_radius = cont.c2 * ENDF_RADIUS_TO_FM;
    let nls = checked_count(cont.n1, "NLS")?; // number of L-values

    let mut l_groups = Vec::with_capacity(nls);

    for _ in 0..nls {
        // CONT: AWRI, APL, L, 0, 6*NRS, NRS
        let l_cont = parse_cont(ctx.lines, ctx.pos)?;
        let awr_l = l_cont.c1;
        let apl = l_cont.c2 * ENDF_RADIUS_TO_FM; // L-dependent scattering radius
        // Validate L is non-negative (#123): negative L1 wraps to a huge u32.
        if l_cont.l1 < 0 {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "Reich-Moore range: negative L={}",
                l_cont.l1
            )));
        }
        let l_val = l_cont.l1 as u32;
        let n1 = checked_count(l_cont.n1, "N1")?; // should be 6*NRS
        let nrs = checked_count(l_cont.n2, "NRS")?; // number of resonances

        // Validate N1 == 6*NRS (#123): a mismatch means the record is malformed
        // and reading N1 values would over-/under-consume lines.
        if n1 != 6 * nrs {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "Reich-Moore range L={l_val}: N1={n1} != 6*NRS={} (NRS={nrs})",
                6 * nrs
            )));
        }

        let mut resonances = Vec::with_capacity(nrs);

        // Each resonance is 6 values: ER, AJ, GN, GG, GFA, GFB
        let total_values = nrs * 6;
        let values = parse_list_values(ctx.lines, ctx.pos, total_values)?;

        for i in 0..nrs {
            let base = i * 6;
            resonances.push(Resonance {
                energy: values[base],  // ER (eV)
                j: values[base + 1],   // AJ (total J)
                gn: values[base + 2],  // GN (neutron width, eV)
                gg: values[base + 3],  // GG (gamma width, eV)
                gfa: values[base + 4], // GFA (fission width 1, eV)
                gfb: values[base + 5], // GFB (fission width 2, eV)
            });
        }

        l_groups.push(LGroup {
            l: l_val,
            awr: awr_l,
            apl,
            qx: 0.0, // Not used in Reich-Moore
            lrx: 0,  // Not used in Reich-Moore
            resonances,
        });
    }

    Ok(ResonanceRange {
        energy_low: ctx.energy_low,
        energy_high: ctx.energy_high,
        resolved: true,
        formalism: ResonanceFormalism::ReichMoore,
        target_spin,
        scattering_radius,
        naps: ctx.naps,
        ap_table: ctx.ap_table.take(),
        l_groups,
        rml: None,
        urr: None,
        r_external: vec![],
    })
}

/// Parse an R-Matrix Limited (LRF=7) resolved resonance range.
///
/// ## ENDF-6 Record Layout (File 2, MT=151, after range CONT + optional TAB1)
///
/// ```text
/// CONT:  [SPI, AP, IFG, KRM, NJS, KRL]
///        SPI = target spin, AP = global scattering radius (fm),
///        NJS = number of spin groups (J,π)
///
/// LIST:  [0, 0, NPP, 0, 12*NPP, NPP]   ← particle pair definitions
///        12 values per pair: [MA, MB, ZA, ZB, IA, IB, Q, PNT, SHF, MT, PA, PB]
///
/// For each spin group j = 1..NJS:
///   LIST: [AJ, PJ, KBK, KPS, 6*(NCH+1), NCH+1]   ← header + channels
///         First 6 values: header row [0, 0, 0, 0, 0, NCH]
///         NCH × 6 values: [IPP, L, SCH, BND, APE, APT] per channel
///
///   LIST: [0, 0, 0, NRS, 6*NX, NX]                  ← resonance parameters
///         Per ENDF-6 §2.2.1.6 and SAMMY mrml01.f:413-415, NRS is in L2
///         (resonance count for this spin group) and NX is in N2 (number of
///         packed 6-float ENDF rows = NRS · ceil(stride/6) where stride is
///         NCH+1 for KRM=2 and NCH+2 for KRM=3); N1 = 6*NX.
///         KRM=2: stride ≥ NCH+1; per resonance: [ER, γ_1, ..., γ_NCH, <padding>]
///         KRM=3: stride ≥ NCH+2; per resonance: [ER, Γγ, Γ_1, ..., Γ_NCH, <padding>]
/// ```
///
/// Reference: ENDF-6 Formats Manual §2.2.1.6; SAMMY rml/mrml01.f
fn parse_rmatrix_limited_range(
    ctx: &mut RangeParseContext<'_>,
    awr: f64,
) -> Result<ResonanceRange, EndfParseError> {
    // CONT: [SPI, AP, IFG, KRM, NJS, KRL]
    // ENDF AP is in 10⁻¹² cm; convert to fm (×10).
    let cont = parse_cont(ctx.lines, ctx.pos)?;
    let target_spin = cont.c1;
    let scattering_radius = cont.c2 * ENDF_RADIUS_TO_FM;
    // IFG (L1): radius unit flag.
    //   IFG=0: AP, APE, APT are in 10⁻¹² cm — universal in ENDF/B-VIII.0.
    //   IFG=1: radii are in units of ℏ/k (energy-dependent) — not supported here.
    // SAMMY's WriteRrEndf.cpp always writes IFG=0 and its reader never checks it,
    // confirming IFG=1 is not used in practice.
    // Reference: ENDF-6 §2.2.1.6; SAMMY ndf/WriteRrEndf.cpp line 363.
    let ifg = cont.l1;
    if ifg != 0 {
        return Err(EndfParseError::UnsupportedFormat(format!(
            "LRF=7 IFG={ifg} (energy-dependent radii) is not supported (only IFG=0)"
        )));
    }
    let krm = cont.l2 as u32; // R-matrix type: 2=standard, 3=Reich-Moore approx
    // P2: Validate KRM at parse time so the physics code never sees an unknown type.
    // KRM=0/1/4 are defined in the ENDF spec but not supported here.
    // Reference: ENDF-6 §2.2.1.6; SAMMY rml/mrml01.f KRM field.
    if krm != 2 && krm != 3 {
        return Err(EndfParseError::UnsupportedFormat(format!(
            "LRF=7 KRM={krm} is not supported (only KRM=2 and KRM=3)"
        )));
    }
    let njs = checked_count(cont.n1, "NJS")?; // number of spin groups
    // KRL (N2): kinematics flag.
    //   KRL=0: non-relativistic kinematics — universal in ENDF/B-VIII.0.
    //   KRL=1: relativistic kinematics — not supported here.
    // SAMMY's WriteRrEndf.cpp always writes KRL=0.
    // Reference: ENDF-6 §2.2.1.6; SAMMY ndf/WriteRrEndf.cpp line 366.
    let krl = cont.n2;
    if krl != 0 {
        return Err(EndfParseError::UnsupportedFormat(format!(
            "LRF=7 KRL={krl} (relativistic kinematics) is not supported (only KRL=0)"
        )));
    }

    // LIST: [0, 0, NPP, 0, 12*NPP, NPP]  — particle pair definitions
    // NPP is authoritative in L1; N2 is nominally equal but can encode a
    // different count in some files (e.g. N2 = 2*NPP).  Always derive from L1.
    // Reference: ENDF-6 Formats Manual §2.2.1.6 Table 2.1.
    let pp_cont = parse_cont(ctx.lines, ctx.pos)?;
    let npp = checked_count(pp_cont.l1, "NPP")?;
    let pp_values = parse_list_values(ctx.lines, ctx.pos, npp * 12)?;

    // Validate-and-narrow an ENDF integer-coded particle-pair flag.  ENDF integer
    // fields are whole numbers, so a fractional or non-finite f64 indicates a
    // malformed record and must not be silently truncated/saturated: PNT=1.7
    // would narrow to 1 and PNT=NaN to 0, both bypassing the {0,1} range check
    // below.  Applied to PNT and SHF — the two flags the physics branches on.
    fn pp_int_flag(value: f64, field: &str, idx: usize) -> Result<i32, EndfParseError> {
        if !value.is_finite() || value.fract() != 0.0 {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "LRF=7 particle pair {idx}: {field}={value} is not a finite integer"
            )));
        }
        Ok(value as i32)
    }

    let mut particle_pairs = Vec::with_capacity(npp);
    for i in 0..npp {
        let b = i * 12;
        particle_pairs.push(ParticlePair {
            ma: pp_values[b],
            mb: pp_values[b + 1],
            za: pp_values[b + 2],
            zb: pp_values[b + 3],
            ia: pp_values[b + 4],
            ib: pp_values[b + 5],
            q: pp_values[b + 6],
            pnt: pp_int_flag(pp_values[b + 7], "PNT", i)?,
            shf: pp_int_flag(pp_values[b + 8], "SHF", i)?,
            mt: pp_values[b + 9] as u32,
            pa: pp_values[b + 10],
            pb: pp_values[b + 11],
        });
    }

    for (i, pp) in particle_pairs.iter().enumerate() {
        // PNT (Lpent) must be 0 or 1.  SAMMY's Check_Quantum (rml/mrml03.f:22)
        // rejects Lpent ∉ {0,1} via Wrongi; PNT=2 ("ASSIGN") is defined in the
        // ENDF-6 spec but neither SAMMY nor NEREIDS implements it.  Validate up
        // front so the physics code never sees an unknown penetrability flag.
        if pp.pnt != 0 && pp.pnt != 1 {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "LRF=7 particle pair {i}: PNT={} is not supported (only PNT=0 \
                 and PNT=1; SAMMY rejects Lpent outside {{0,1}})",
                pp.pnt
            )));
        }
        // A massless pair (photon/eliminated channel, MA=0) must carry PNT=0:
        // SAMMY always assigns Lpent=0 to the photon channel, and the physics
        // evaluator's penetrability branch (which divides by a reduced mass)
        // is only entered for PNT=1.  Reject the inconsistent combination so a
        // massless PNT=1 pair can never reach a divide-by-zero reduced mass.
        if pp.ma < 0.5 && pp.pnt != 0 {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "LRF=7 particle pair {i}: massless pair (MA={}) with PNT={} is \
                 invalid; a photon/eliminated channel must have PNT=0",
                pp.ma, pp.pnt
            )));
        }
        // A PNT=1 pair drives the penetrability path, which forms the reduced
        // mass μ = MA·MB/(MA+MB) (rmatrix_limited.rs).  Validate the reduced mass
        // itself — computed exactly as the physics does — is finite and strictly
        // positive, so the physics never sees a non-finite μ.  Checking MA/MB > 0
        // alone is insufficient: pathological huge (but finite) masses can still
        // overflow MA·MB to ∞.  This also covers the MA+MB = 0 / sign cases.
        if pp.pnt == 1 {
            let reduced_mass = pp.ma * pp.mb / (pp.ma + pp.mb);
            if !(pp.ma.is_finite()
                && pp.mb.is_finite()
                && pp.ma > 0.0
                && pp.mb > 0.0
                && reduced_mass.is_finite()
                && reduced_mass > 0.0)
            {
                return Err(EndfParseError::UnsupportedFormat(format!(
                    "LRF=7 particle pair {i}: PNT=1 requires finite positive masses \
                     yielding a finite reduced mass (MA={}, MB={})",
                    pp.ma, pp.mb
                )));
            }
        }
        // Coulomb + SHF=1: closed-channel Coulomb shift at imaginary argument is
        // unimplemented.  Reject at parse time rather than silently producing wrong
        // dispersive terms near threshold.
        // Reference: SAMMY rml/mrml07.f — Pghcou is only called for open channels.
        if pp.za.abs() > 0.5 && pp.zb.abs() > 0.5 && pp.shf == 1 {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "LRF=7 particle pair {i}: Coulomb channel (za={}, zb={}) with \
                 SHF=1 is not supported; closed-channel Coulomb shift at \
                 imaginary rho is not yet implemented",
                pp.za, pp.zb
            )));
        }
    }

    // All particle-pair types are now fully supported (with the SHF=1 restriction above):
    // - PNT 0/1: branched on pp.pnt in rmatrix_limited.rs (SAMMY Lpent); PNT∉{0,1} rejected above.
    // - SHF 0/1: respected by the shf field in rmatrix_limited.rs.
    // - Coulomb channels (pp.za > 0 && pp.zb > 0): routed through
    //   nereids_physics::coulomb (Steed's CF1+CF2, SAMMY coulomb/mrml08.f90).
    let mut spin_groups = Vec::with_capacity(njs);

    for _ in 0..njs {
        // LIST: [AJ, PJ, KBK, KPS, 6*(NCH+1), NCH+1]
        // First 6*(NCH+1) values: header row [0,0,0,0,0,NCH] then NCH×6 channel defs.
        let sg_cont = parse_cont(ctx.lines, ctx.pos)?;
        let aj = sg_cont.c1;
        let pj = sg_cont.c2; // explicit parity field; may be 0.0 when parity is in sign(AJ)
        let kbk = sg_cont.l1; // background R-matrix flag
        let kps = sg_cont.l2; // phase shift flag

        // KBK: background R-matrix correction (R-external function on a
        // subset of channels). Per the printed ENDF-6 §2.2.1.6 Tables 2.4/2.5
        // KBK is described as a nonzero flag with NCH background records,
        // while the reference reader OpenScale
        // (external/openScale/repo/packages/ScaleUtils/EndfLib/endf/File2.cpp:444-524)
        // treats KBK as a sparse record count, with each subrecord's L1 holding
        // the 1-based channel index and L2 holding the LBK formalism flag
        // (LBK ∈ {0=no payload, 1=two TAB1, 2=LIST(5), 3=LIST(3)}). The two
        // conventions disagree on (a) the loop bound, (b) the per-subrecord
        // control-field positions, and (c) the payload shape per LBK value.
        //
        // No ENDF/B-VIII.0 evaluation in the local cache has nonzero KBK or
        // KPS to disambiguate, and the only nonzero example located on disk
        // is OpenScale's synthetic F-19 R-external test fixture
        // (Ampx/TestRunner/test/data/polident/f19_rext.endf), which follows
        // the OpenScale convention. NEREIDS's previous layout matched neither
        // convention. Until a policy decision is made (strict-manual vs.
        // OpenScale-compat) and a real ENDF/B-VIII.0 evaluation with R-external
        // is available to validate against, reject nonzero KBK explicitly so
        // the parser cannot silently misalign the stream past this spin group.
        //
        // The reject runs immediately after reading the spin-group CONT —
        // before parsing the (potentially large) channel and resonance LISTs —
        // so that unsupported files fail fast without wasting allocation and
        // parsing work on records that will be discarded.
        if kbk != 0 {
            let nch_plus_one_raw = sg_cont.n2;
            return Err(EndfParseError::UnsupportedFormat(format!(
                "LRF=7 KBK={kbk} != 0 (R-external background) for spin group with \
                 NCH+1={nch_plus_one_raw}: \
                 the ENDF-6 manual vs. OpenScale layout dispute is unresolved and NEREIDS \
                 does not yet parse nonzero KBK. Use the SAMMY .par/.inp converter \
                 (sammy_to_resonance_data_multi) if R-external is required."
            )));
        }

        // KPS: tabulated penetrability/phase-shift override per channel.
        // Same documentation-vs-implementation dispute as KBK above
        // (OpenScale File2.cpp:439-441 throws "kps > 0 for lrf=7 not yet
        // supported" and never reads the subrecords). NEREIDS rejects nonzero
        // KPS for the same reason: no validated reference layout, no real
        // evaluation to test against.
        if kps != 0 {
            let nch_plus_one_raw = sg_cont.n2;
            return Err(EndfParseError::UnsupportedFormat(format!(
                "LRF=7 KPS={kps} != 0 (tabulated penetrability/phase-shift override) \
                 for spin group with NCH+1={nch_plus_one_raw}: \
                 NEREIDS does not yet parse nonzero KPS. \
                 OpenScale itself rejects this case (\"kps > 0 for lrf=7 not yet supported\")."
            )));
        }

        // AJ encodes both the spin and, in some evaluations, the parity.
        // ENDF/B-VIII.0 evaluations such as W-184 use negative AJ for odd-parity
        // spin groups (e.g., AJ=-0.5, AJ=-1.5) and set PJ=0.
        // Statistical weight formula (2J+1)/... requires J > 0; negative J yields
        // zero or negative weights and drives non-physical cross-sections.
        // Fix: J = |AJ|; parity from sign(AJ) when PJ is absent (PJ=0).
        // Reference: ENDF-6 §2.2.1.6; SAMMY rml/mrml01.f Scan_File_2.
        let j = aj.abs();
        let parity = if pj != 0.0 {
            pj.signum()
        } else if aj < 0.0 {
            -1.0
        } else {
            1.0
        };
        let npl = checked_count(sg_cont.n1, "NPL")?; // 6*(NCH+1)
        let nch_plus_one = checked_count(sg_cont.n2, "NCH+1")?; // NCH+1

        // NCH+1 <= 1 would imply zero physical channels (NCH = 0), which is
        // meaningless for a resonance range — every spin group must have at
        // least one channel.
        // Reference: ENDF-6 §2.2.1.6; SAMMY rml/mrml01.f.
        if nch_plus_one <= 1 {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "RML spin-group LIST: NCH+1 must be >= 2, got NCH+1={nch_plus_one}"
            )));
        }
        let nch = nch_plus_one - 1;

        let sg_values = parse_list_values(ctx.lines, ctx.pos, npl)?;

        // C3: Validate that the LIST record carries at least 6*(NCH+1) values.
        // NCH is derived from N2 in the LIST header (N2 = NCH+1); the first data row
        // is a dummy/header row of zeros that ENDF evaluators may fill arbitrarily.
        // SAMMY (mrml01.f Scan_File_2/ENDF123) reads NCH from N2 and ignores row[5].
        // Reference: ENDF-6 §2.2.1.6 Table 2.3; SAMMY rml/mrml01.f lines 104-107.
        let expected_npl = 6 * (nch + 1);
        if npl < expected_npl {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "LRF=7 spin-group LIST: NPL={npl} < 6*(NCH+1)={expected_npl}"
            )));
        }

        // First 6 values are the dummy header row (zeros); subsequent NCH×6 values
        // are channel definitions [IPP, L, SCH, BND, APE, APT] per channel.
        let npp = particle_pairs.len();
        let mut channels = Vec::with_capacity(nch);
        for c in 0..nch {
            let b = 6 + c * 6; // skip the 6-value header row
            // C2: IPP is 1-based in ENDF; validate range before converting.
            let ipp_raw = sg_values[b] as usize;
            if ipp_raw == 0 || ipp_raw > npp {
                return Err(EndfParseError::UnsupportedFormat(format!(
                    "LRF=7 spin-group channel IPP={ipp_raw} is out of range 1..={npp}"
                )));
            }
            // Photon channels (MA < 0.5, PNT=0) are stored as regular channels.
            // The physics code sets P_c=1, S_c=B_c, φ_c=0 for PNT=0 channels
            // (rmatrix_limited.rs; SAMMY rml/mrml07.f:118-122 Ymat(2)-=1) and
            // classifies them as capture channels via pp.mt == 102.  Their reduced width amplitudes
            // appear at the corresponding column position in the resonance rows,
            // exactly like any other channel.  Reference: ENDF-6 §2.2.1.6; SAMMY
            // rml/mrml01.f (Ippx test, mrml07.f P=1 convention for massless).
            channels.push(RmlChannel {
                particle_pair_idx: ipp_raw - 1, // convert 1-based ENDF index to 0-based
                l: sg_values[b + 1] as u32,     // L
                channel_spin: sg_values[b + 2], // SCH
                boundary: sg_values[b + 3],     // BND
                effective_radius: sg_values[b + 4] * ENDF_RADIUS_TO_FM, // APE
                true_radius: sg_values[b + 5] * ENDF_RADIUS_TO_FM, // APT
            });
        }

        // Apply global scattering radius for channels where APE/APT == 0
        for ch in &mut channels {
            if ch.effective_radius == 0.0 {
                ch.effective_radius = scattering_radius;
            }
            if ch.true_radius == 0.0 {
                ch.true_radius = scattering_radius;
            }
        }

        // LIST: [0, 0, 0, NRS, 6*NX, NX]  — resonance parameters.
        //
        // ENDF-6 §2.2.1.6 fixes the resonance LIST control fields as
        // [C1=0, C2=0, L1=0, L2=NRS, N1=6*NX, N2=NX]:
        //   NRS lives in L2 (the resonance count for this spin group).
        //   NX  lives in N2 (number of packed 6-float ENDF data rows =
        //       NRS · ceil(stride/6) where stride is NCH+1 for KRM=2 and
        //       NCH+2 for KRM=3), and N1 must equal 6*NX.
        //
        // For spin groups where each resonance fits in one packed row
        // (NCH+1 ≤ 6 for KRM=2, NCH+2 ≤ 6 for KRM=3) NX == NRS and the
        // distinction is invisible; for larger NCH (e.g. F-19 spin groups
        // with NCH≥5) NX > NRS and reading NRS from N2 over-counts the
        // resonances and trips the stride guard below with a misleading
        // "stride too small" error.
        //
        // SAMMY reads NRS via `FORMAT (33X, I11)` which skips C1+C2+L1
        // (3 × 11 chars) and reads L2 (mrml01.f:413-415, also :116-119
        // for the scan pass). OpenScale reads `list.getL2()` and writes
        // `list.setL2(nres) / setN2(nx)` (File2.cpp:415, :686-697).
        //
        // For KRM=3 (e.g. W-184 ENDF/B-VIII.0), evaluators pad each resonance row
        // to a fixed 6 values per ENDF line, so NPL/NRS = 6 even when NCH=1.
        // Using hardcoded nch+1 drifts the offset and misreads zeros as energies.
        // Fix: derive stride directly from NPL/NRS; read only NCH widths per row.
        let res_cont = parse_cont(ctx.lines, ctx.pos)?;
        let nrs = checked_count(res_cont.l2, "NRS")?;
        let nx = checked_count(res_cont.n2, "NX")?;
        let res_npl = checked_count(res_cont.n1, "NPL")?;
        if res_npl != 6 * nx {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "LRF=7 resonance LIST: N1 ({res_npl}) != 6 * N2 ({}); ENDF-6 §2.2.1.6 \
                 requires NPL = 6*NX for the packed-row layout",
                6 * nx
            )));
        }
        // Per ENDF-6 §2.2.1.6, NX is the per-spin-group packed-row count:
        //     NX = NRS · ceil(per_resonance_floats / 6)
        // where the per-resonance float count is layout-dependent:
        //     KRM=2: per_resonance = NCH+1  (ER + NCH reduced widths γ_c)
        //     KRM=3: per_resonance = NCH+2  (ER + Γγ + NCH partial widths Γ_c)
        // SAMMY rml/mrml01.f ENDF123 confirms the KRM=3 layout reads Gamgam
        // at position 1 and (Gamma,I=1,Ichan) at positions 2..NCH+1.
        // Because the per-resonance row count is constant within a spin
        // group, NX is always an integer multiple of NRS. A non-zero NRS
        // with NX not divisible by NRS would yield a fractional stride
        // (`6 * NX / NRS` non-integer) and mis-align resonance reads.
        // Reject up-front rather than rely on the downstream
        // `res_npl % nrs != 0` check, which is a weaker invariant.
        if nrs > 0 && nx % nrs != 0 {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "LRF=7 resonance LIST: N2/NX ({nx}) is not a multiple of L2/NRS ({nrs}); \
                 ENDF-6 §2.2.1.6 requires NX = NRS * ceil(stride/6) where stride is \
                 NCH+1 for KRM=2 and NCH+2 for KRM=3"
            )));
        }
        // Canonical empty spin group per ENDF-6 §2.2.1.6 and OpenScale's
        // writer at File2.cpp:683-697:
        //   list.setL2(spin->getNres());        // L2 = NRS
        //   ...
        //   // nx must be at least 1, even if nres=0
        //   if (spin->getNres() == 0)
        //       nx = 1;
        //   list.setN1(6 * nx);                  // N1 = 6
        //   list.setN2(nx);                      // N2 = 1
        // The LIST body for the empty spin group is a single 6-float zero
        // filler row. Reject any NRS=0 record that does not carry NX=1
        // (NX=0 is malformed by OpenScale; NX>1 would imply phantom rows
        // with no resonance count to anchor them).
        if nrs == 0 && nx != 1 {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "LRF=7 resonance LIST: NRS=0 requires NX=1 (single zero-filler row \
                 per ENDF-6 §2.2.1.6 + OpenScale File2.cpp:683-697); got NX={nx}"
            )));
        }
        let res_values = parse_list_values(ctx.lines, ctx.pos, res_npl)?;

        // C4: Validate stride before use — NPL must divide evenly by NRS, and each row
        // must be at least min_stride values wide.
        //
        // KRM=2: per-resonance layout is [ER, Γ_1, ..., Γ_NCH, <padding>]
        //        → min_stride = NCH+1 (energy + NCH reduced width amplitudes)
        // KRM=3: per-resonance layout is [ER, Γγ, Γ_1, ..., Γ_NCH, <padding>]
        //        → min_stride = NCH+2 (energy + Gamgam + NCH partial widths)
        //
        // Reference: ENDF-6 §2.2.1.6; SAMMY rml/mrml01.f ENDF123 subroutine —
        //   reads Gamgam at position 1 (immediately after ER), then
        //   (Gamma,I=1,Ichan) at positions 2..NCH+1.
        let min_stride = if krm == 3 { nch + 2 } else { nch + 1 };
        let stride = if nrs == 0 {
            min_stride // no resonances; stride unused
        } else {
            if res_npl % nrs != 0 {
                return Err(EndfParseError::UnsupportedFormat(format!(
                    "LRF=7 resonance block NPL={res_npl} is not divisible by NRS={nrs}"
                )));
            }
            let s = res_npl / nrs;
            if s < min_stride {
                return Err(EndfParseError::UnsupportedFormat(format!(
                    "LRF=7 resonance stride={s} < {}={min_stride} \
                     (KRM={krm}, NPL={res_npl}, NRS={nrs})",
                    if krm == 3 { "NCH+2" } else { "NCH+1" }
                )));
            }
            s
        };
        let mut resonances = Vec::with_capacity(nrs);
        for r in 0..nrs {
            let b = r * stride;
            // Parse resonance row according to KRM column order.
            //
            // KRM=2: [ER, γ_1, ..., γ_NCH, <padding>]
            //   widths (reduced amplitudes γ) start at b+1.
            //   No capture width column; gamma_gamma = 0.
            //
            // KRM=3: [ER, Γγ, Γ_1, ..., Γ_NCH, <padding>]
            //   Gamgam (radiation width, eV) is at b+1.
            //   Partial widths Γ_c start at b+2.
            //   Gamgam forms complex pole energies: Ẽ_n = E_n - i·Γγ/2.
            //
            // Reference: ENDF-6 §2.2.1.6; SAMMY rml/mrml01.f ENDF123 subroutine.
            //
            // Bounds safety: stride ≥ min_stride (verified above), b = r·stride,
            // and r < nrs, so b + stride ≤ res_npl = res_values.len().
            // For KRM=3: b+2+nch ≤ b+min_stride ≤ b+stride; guaranteed in bounds.
            // For KRM=2: b+1+nch ≤ b+min_stride ≤ b+stride; guaranteed in bounds.
            // Explicit error checks below make the safety locally verifiable and
            // guard against future changes that might weaken the stride invariant.
            let (widths, gamma_gamma) = if krm == 3 {
                let need = b + 2 + nch;
                if need > res_values.len() {
                    return Err(EndfParseError::UnsupportedFormat(format!(
                        "LRF=7 KRM=3 resonance row {r}: need {need} values, \
                         have {} (stride={stride}, NCH={nch})",
                        res_values.len()
                    )));
                }
                let gamma_gamma = res_values[b + 1]; // Gamgam at position 1
                let widths = res_values[b + 2..b + 2 + nch].to_vec(); // Γ_c at positions 2..NCH+1
                (widths, gamma_gamma)
            } else {
                // KRM=2: widths immediately follow ER; no capture-width column.
                let need = b + 1 + nch;
                if need > res_values.len() {
                    return Err(EndfParseError::UnsupportedFormat(format!(
                        "LRF=7 KRM=2 resonance row {r}: need {need} values, \
                         have {} (stride={stride}, NCH={nch})",
                        res_values.len()
                    )));
                }
                (res_values[b + 1..b + 1 + nch].to_vec(), 0.0)
            };
            resonances.push(RmlResonance {
                energy: res_values[b],
                widths,
                gamma_gamma,
            });
        }

        spin_groups.push(SpinGroup {
            j,
            parity,
            channels,
            resonances,
            // Nonzero KBK/KPS are rejected at the top of this loop iteration
            // (immediately after the spin-group CONT is read), so any spin
            // group that reaches this point has no background correction.
            has_background_correction: false,
        });
    }

    let rml = RmlData {
        target_spin,
        awr,
        scattering_radius,
        krm,
        particle_pairs,
        spin_groups,
    };

    Ok(ResonanceRange {
        energy_low: ctx.energy_low,
        energy_high: ctx.energy_high,
        resolved: true,
        formalism: ResonanceFormalism::RMatrixLimited,
        target_spin,
        scattering_radius,
        naps: ctx.naps,
        ap_table: ctx.ap_table.take(),
        l_groups: Vec::new(),
        rml: Some(Box::new(rml)),
        urr: None,
        r_external: vec![],
    })
}

/// Maximum sane ENDF count value.
///
/// ENDF files in practice never contain more than ~100k resonances per section.
/// Accepting `i32::MAX` would cause enormous allocations (gigabytes) on
/// malformed files.  This cap is generous enough for any real evaluation while
/// protecting against allocation bombs.
const MAX_ENDF_COUNT: i32 = 1_000_000;

/// Validate that an ENDF integer count is non-negative and return as `usize`.
///
/// Malformed records can contain negative counts which, if cast directly to
/// `usize`, wrap to huge values and cause OOM panics in `Vec::with_capacity`
/// or `parse_list_values`.
fn checked_count(value: i32, label: &str) -> Result<usize, EndfParseError> {
    if value < 0 {
        return Err(EndfParseError::UnsupportedFormat(format!(
            "Negative ENDF count: {label}={value}"
        )));
    }
    if value > MAX_ENDF_COUNT {
        return Err(EndfParseError::UnsupportedFormat(format!(
            "ENDF count too large: {label}={value} (maximum {MAX_ENDF_COUNT})"
        )));
    }
    Ok(value as usize)
}

/// Skip an unsupported URR body **with LFW=0** layout.
///
/// Called when LRU=2 has an LRF value other than 1 or 2 so that the records
/// for this range are consumed and subsequent ranges can still be parsed.
///
/// Structure consumed (ENDF-6 §2.2.2, LFW=0):
/// ```text
/// CONT: SPI, AP, 0, 0, NLS, 0
/// For each L (NLS times):
///   CONT: AWRI, 0, L, 0, N1, N2
///   if N2 > 0  -> LRF=1 style: one LIST record of N1 values
///   if N2 == 0 -> LRF=2 style: N1 J-sub-blocks, each = CONT + LIST(N1_j values)
/// ```
fn skip_urr_body(lines: &[&str], pos: &mut usize) -> Result<(), EndfParseError> {
    // CONT: SPI, AP, 0, 0, NLS, 0
    let header = parse_cont(lines, pos)?;
    let nls = checked_count(header.n1, "NLS")?;

    for _ in 0..nls {
        // L CONT: AWRI, 0, L, 0, N1, N2
        let l_cont = parse_cont(lines, pos)?;
        let n1 = checked_count(l_cont.n1, "N1")?;
        let n2 = checked_count(l_cont.n2, "N2")?;

        if n2 > 0 {
            // LRF=1 style: N2=NJS, N1=6*NJS — single LIST record.
            parse_list_values(lines, pos, n1)?;
        } else {
            // LRF=2 style: N1=NJS, N2=0 — N1 J-sub-blocks, each with their
            // own CONT (carrying 6*(NE+1) in N1) followed by a LIST record.
            for _ in 0..n1 {
                let j_cont = parse_cont(lines, pos)?;
                let jn1 = checked_count(j_cont.n1, "N1")?;
                parse_list_values(lines, pos, jn1)?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Low-level ENDF line parsing helpers
// ---------------------------------------------------------------------------

/// Parsed CONT (control) record with 2 floats, 4 integers.
struct ContRecord {
    c1: f64,
    c2: f64,
    l1: i32,
    l2: i32,
    n1: i32,
    n2: i32,
}

/// Parse a CONT record from the current line.
fn parse_cont(lines: &[&str], pos: &mut usize) -> Result<ContRecord, EndfParseError> {
    if *pos >= lines.len() {
        return Err(EndfParseError::UnexpectedEof(
            "Expected CONT record but reached end of data".to_string(),
        ));
    }
    let line = lines[*pos];
    *pos += 1;

    Ok(ContRecord {
        c1: parse_endf_float(line, 0)?,
        c2: parse_endf_float(line, 1)?,
        l1: parse_endf_int(line, 2)?,
        l2: parse_endf_int(line, 3)?,
        n1: parse_endf_int(line, 4)?,
        n2: parse_endf_int(line, 5)?,
    })
}

/// Parse a LIST of floating-point values spanning multiple lines.
///
/// ENDF packs 6 values per line. We read ceil(n/6) lines.
fn parse_list_values(
    lines: &[&str],
    pos: &mut usize,
    n_values: usize,
) -> Result<Vec<f64>, EndfParseError> {
    let mut values = Vec::with_capacity(n_values);
    let n_lines = n_values.div_ceil(6);

    for _ in 0..n_lines {
        if *pos >= lines.len() {
            return Err(EndfParseError::UnexpectedEof(
                "Expected LIST data but reached end".to_string(),
            ));
        }
        let line = lines[*pos];
        *pos += 1;

        let remaining = n_values - values.len();
        let fields_on_line = remaining.min(6);

        for field in 0..fields_on_line {
            values.push(parse_endf_float(line, field)?);
        }
    }

    Ok(values)
}

/// Parse a floating-point value from an 11-character ENDF field.
///
/// ENDF uses Fortran-style floats that may omit 'E', e.g.:
/// - " 1.234567+2" means 1.234567e+2
/// - "-3.456789-1" means -3.456789e-1
/// - " 0.000000+0" means 0.0
fn parse_endf_float(line: &str, field_index: usize) -> Result<f64, EndfParseError> {
    let start = field_index * 11;
    let end = start + 11;

    if line.len() < end {
        // Short line — treat as zero.
        return Ok(0.0);
    }

    let field = &line[start..end];
    let trimmed = field.trim();

    if trimmed.is_empty() {
        return Ok(0.0);
    }

    // Try standard Rust float parsing first.
    if let Ok(v) = trimmed.parse::<f64>() {
        return Ok(v);
    }

    // Handle Fortran-style: "1.234567+2" or "-3.456789-1"
    // Look for +/- that is NOT the first character and NOT preceded by 'e'/'E'/'d'/'D'.
    let bytes = trimmed.as_bytes();
    for i in 1..bytes.len() {
        if (bytes[i] == b'+' || bytes[i] == b'-')
            && bytes[i - 1] != b'e'
            && bytes[i - 1] != b'E'
            && bytes[i - 1] != b'd'
            && bytes[i - 1] != b'D'
            && bytes[i - 1] != b'+'
            && bytes[i - 1] != b'-'
        {
            let mantissa = &trimmed[..i];
            let exp_slice = &trimmed[i..];
            // Strip spaces from the exponent only when present (some ENDF files
            // write "+ 4" not "+4").  Avoid allocation on the common path.
            let with_e = if exp_slice.contains(' ') {
                let exponent: String = exp_slice.chars().filter(|c| !c.is_whitespace()).collect();
                format!("{}E{}", mantissa, exponent)
            } else {
                format!("{}E{}", mantissa, exp_slice)
            };
            if let Ok(v) = with_e.parse::<f64>() {
                return Ok(v);
            }
        }
    }

    Err(EndfParseError::InvalidNumber(format!(
        "Cannot parse ENDF float: '{}'",
        field
    )))
}

/// Parse an integer from an 11-character ENDF field.
fn parse_endf_int(line: &str, field_index: usize) -> Result<i32, EndfParseError> {
    let start = field_index * 11;
    let end = start + 11;

    if line.len() < end {
        return Ok(0);
    }

    let field = &line[start..end];
    let trimmed = field.trim();

    if trimmed.is_empty() {
        return Ok(0);
    }

    // ENDF integers may have a decimal point (e.g., "1.000000+0" for 1).
    // Try integer parse first, then float-with-integral-value.
    if let Ok(v) = trimmed.parse::<i32>() {
        return Ok(v);
    }

    // Try parsing as float, but reject non-integral values (e.g., "1.9e+0")
    // rather than silently truncating.  Use the same ε=1e-6 tolerance as
    // `parse_tab1`'s NBT/INT validation so that integer fields stored as
    // ENDF floats ("1.000000+0") still round-trip, but a malformed
    // "1.900000+0" is surfaced as an InvalidNumber rather than parsed as 1.
    if let Ok(v) = parse_endf_float(line, field_index) {
        if (v - v.round()).abs() > 1e-6 {
            return Err(EndfParseError::InvalidNumber(format!(
                "Non-integral value in ENDF int field: '{}' (={})",
                field, v
            )));
        }
        return Ok(v.round() as i32);
    }

    Err(EndfParseError::InvalidNumber(format!(
        "Cannot parse ENDF int: '{}'",
        field
    )))
}

/// Parse a URR range with LFW=1, LRF=1 (energy-dependent fission widths,
/// single-level BW).
///
/// ## Record layout (ENDF-6 §2.2.2.1 "Case B")
/// ```text
/// CONT: SPI, AP, LSSF, 0, NE, NLS
/// LIST: NE energy values (shared fission width grid)
/// For each L:
///   CONT: AWRI, 0, L, 0, NJS, 0
///   For each J:
///     LIST control: 0.0, 0.0, L, MUF, NE+6, 0
///     LIST body:    [D, AJ, AMUN, GNO, GG, 0] + NE fission widths
/// ```
///
/// Each per-J record is a full ENDF LIST: a control line carrying the data
/// count `N1 = NE+6` and the fission degrees-of-freedom `MUF` (the L2 field),
/// followed by `NE+6` data values.  The control line MUST be consumed before
/// the body — omitting it misaligns the line stream by one record per J-group.
///
/// Other widths (D, GNO, GG) are energy-independent (single values).
/// Only fission widths (GF) are tabulated at the shared energy grid.
///
/// Reference: ENDF-6 §2.2.2.1 Case B; SCALE/openScale `File2.cpp`
/// (`lfw==1 && lrf==1` branch, per-J `list.readData`/`cont.readData`) and
/// `File2Unres.f90` (read loop, per-J `ControlEndf_read` + `ListEndf_read`;
/// `File2Unres_getMuf` reads MUF from the control L2 field, `getNeJ` reads
/// `N1 = NE+6`).
fn parse_urr_lfw1_lrf1(ctx: &mut RangeParseContext<'_>) -> Result<ResonanceRange, EndfParseError> {
    // CONT: SPI, AP, LSSF, 0, NE, NLS
    let header = parse_cont(ctx.lines, ctx.pos)?;
    let spi = header.c1;
    let ap = header.c2 * ENDF_RADIUS_TO_FM;
    let ne = checked_count(header.n1, "NE")?;
    let nls = checked_count(header.n2, "NLS")?;

    // LIST: NE energy values (shared fission width grid)
    let fission_energies = parse_list_values(ctx.lines, ctx.pos, ne)?;

    let mut l_groups = Vec::with_capacity(nls);

    for _ in 0..nls {
        // CONT: AWRI, 0, L, 0, NJS, 0
        let l_cont = parse_cont(ctx.lines, ctx.pos)?;
        let awri = l_cont.c1;
        let l = l_cont.l1 as u32;
        let njs = checked_count(l_cont.n1, "NJS")?;

        let mut j_groups = Vec::with_capacity(njs);

        for _ in 0..njs {
            // Per-J LIST control: 0.0, 0.0, L, MUF, NE+6, 0
            // MUF (fission degrees of freedom) is the L2 field; the data
            // count N1 must equal NE+6.  Consuming this control record keeps
            // the line stream aligned — the body follows on the next lines.
            let j_cont = parse_cont(ctx.lines, ctx.pos)?;
            let muf = j_cont.l2;
            let n1 = checked_count(j_cont.n1, "N1")?;
            // SCALE validates this exact relation (File2.cpp: N1-6 == NE).
            if n1 != ne + 6 {
                return Err(EndfParseError::UnsupportedFormat(format!(
                    "URR LFW=1/LRF=1: per-J N1={n1} ≠ NE+6={} (NE={ne})",
                    ne + 6
                )));
            }

            // LIST body: [D, AJ, AMUN, GNO, GG, 0] + NE fission widths
            let values = parse_list_values(ctx.lines, ctx.pos, ne + 6)?;

            let d = values[0];
            let aj = values[1];
            let amun = values[2];
            let gno = values[3];
            let gg = values[4];
            // values[5] = 0 (unused)

            // Fission widths from the shared energy grid
            let gf: Vec<f64> = values[6..6 + ne].to_vec();

            j_groups.push(UrrJGroup {
                j: aj,
                amun,
                // Case B carries the fission degrees of freedom MUF in the
                // per-J control record's L2 field; store it as AMUF.
                amuf: muf as f64,
                energies: fission_energies.clone(),
                d: vec![d],
                gx: vec![0.0],
                gn: vec![gno],
                gg: vec![gg],
                gf,
                int_code: 2, // Default lin-lin for fission width interpolation
            });
        }

        l_groups.push(UrrLGroup { l, awri, j_groups });
    }

    Ok(ResonanceRange {
        energy_low: ctx.energy_low,
        energy_high: ctx.energy_high,
        resolved: false,
        formalism: ResonanceFormalism::Unresolved,
        target_spin: spi,
        scattering_radius: ap,
        naps: ctx.naps,
        ap_table: ctx.ap_table.take(),
        l_groups: Vec::new(),
        rml: None,
        urr: Some(Box::new(UrrData {
            lrf: 1,
            spi,
            ap,
            e_low: ctx.energy_low,
            e_high: ctx.energy_high,
            l_groups,
        })),
        r_external: vec![],
    })
}

/// Parse an Unresolved Resonance Region (LRU=2) range.
///
/// Handles two routes:
/// * LFW=0 (LRF=1 energy-independent or LRF=2 tabulated widths) — the
///   standard URR case.
/// * LFW=1 / LRF=2 — the LIST record layout is byte-identical to
///   LFW=0/LRF=2 (the LFW=1 fission-width grid is embedded in each
///   per-J LIST row rather than a separate shared-grid block), so the
///   caller routes that combination here.
///
/// LFW=1 / LRF=1 (energy-dependent fission widths with the shared-grid
/// layout) is handled separately by `parse_urr_lfw1_lrf1`.
///
/// Reference: ENDF-6 Formats Manual §2.2.2
fn parse_urr_range(
    ctx: &mut RangeParseContext<'_>,
    lrf: i32,
) -> Result<ResonanceRange, EndfParseError> {
    use crate::resonance::{UrrData, UrrJGroup, UrrLGroup};

    // See function-level rustdoc for the LFW/LRF routing rules.

    // CONT: SPI, AP, 0, 0, NLS, 0
    // ENDF AP is in 10⁻¹² cm; convert to fm (×10).
    let spi_cont = parse_cont(ctx.lines, ctx.pos)?;
    let spi = spi_cont.c1;
    let ap = spi_cont.c2 * ENDF_RADIUS_TO_FM; // scattering radius (fm)
    let nls = checked_count(spi_cont.n1, "NLS")?;

    let mut l_groups = Vec::with_capacity(nls);

    if lrf == 1 {
        // LRF=1: energy-independent widths, one LIST block per L covering all J.
        for _ in 0..nls {
            // CONT: AWRI, 0, L, 0, N1=6*NJS, N2=NJS
            let l_cont = parse_cont(ctx.lines, ctx.pos)?;
            let awri = l_cont.c1;
            if l_cont.l1 < 0 {
                return Err(EndfParseError::UnsupportedFormat(format!(
                    "URR LRF=1: negative L={}",
                    l_cont.l1
                )));
            }
            let l = l_cont.l1 as u32;
            let n1 = checked_count(l_cont.n1, "N1")?; // 6*NJS
            let njs = checked_count(l_cont.n2, "NJS")?;

            if njs == 0 || n1 != 6 * njs {
                return Err(EndfParseError::UnsupportedFormat(format!(
                    "URR LRF=1 L={l}: N1={n1} ≠ 6×NJS={} (NJS={njs})",
                    6 * njs
                )));
            }

            let values = parse_list_values(ctx.lines, ctx.pos, n1)?;

            let mut j_groups = Vec::with_capacity(njs);
            for j_idx in 0..njs {
                let base = j_idx * 6;
                // [D, AJ, AMUN, GNO, GG, GF]
                j_groups.push(UrrJGroup {
                    j: values[base + 1],        // AJ
                    amun: values[base + 2],     // AMUN (neutron DOF)
                    amuf: 0.0,                  // LRF=1 format does not carry AMUF
                    energies: vec![],           // Energy-independent
                    d: vec![values[base]],      // D (level spacing, eV)
                    gx: vec![0.0],              // No competitive width in LRF=1
                    gn: vec![values[base + 3]], // GNO (reduced neutron width, eV)
                    gg: vec![values[base + 4]], // GG (gamma width, eV)
                    gf: vec![values[base + 5]], // GF (fission width, eV)
                    int_code: 2,                // LRF=1 has no table; default lin-lin
                });
            }

            l_groups.push(UrrLGroup { l, awri, j_groups });
        }
    } else {
        // LRF=2: energy-dependent width tables, one LIST per (L, J).
        for _l_idx in 0..nls {
            // CONT: AWRI, 0, L, 0, NJS, 0
            let l_cont = parse_cont(ctx.lines, ctx.pos)?;
            let awri = l_cont.c1;
            if l_cont.l1 < 0 {
                return Err(EndfParseError::UnsupportedFormat(format!(
                    "URR LRF=2: negative L={}",
                    l_cont.l1
                )));
            }
            let l = l_cont.l1 as u32;
            let njs = checked_count(l_cont.n1, "NJS")?; // N1 = NJS for LRF=2

            // Zero NJS means no J-groups for this L-value, which is malformed
            // (ENDF §2.2.2.2 requires at least one J-group per L-group).
            // Consistent with the LRF=1 path which also rejects NJS=0.
            if njs == 0 {
                return Err(EndfParseError::UnsupportedFormat(format!(
                    "URR LRF=2 L={l}: NJS=0 (at least one J-group required)"
                )));
            }

            let mut j_groups = Vec::with_capacity(njs);
            for _j_idx in 0..njs {
                // CONT: AJ, 0, INT, 0, N1=6*(NE+1), N2=NE
                let j_cont = parse_cont(ctx.lines, ctx.pos)?;
                let aj = j_cont.c1;
                let int_code = j_cont.l1; // interpolation law (L1 field)
                // ENDF-6 §0.5 defines INT codes 1..=5 (1=histogram, 2=lin-lin,
                // 3=log-x/lin-y, 4=lin-x/log-y, 5=log-log). Anything outside
                // that range — including negative values and INT=0 or INT≥6 —
                // is a malformed record, not merely an unimplemented mode.
                if !(1..=5).contains(&int_code) {
                    return Err(EndfParseError::UnsupportedFormat(format!(
                        "URR LRF=2 J={aj}: INT={int_code} out of spec (expected 1..=5)"
                    )));
                }
                let n1 = checked_count(j_cont.n1, "N1")?; // 6*(NE+1)
                let ne = checked_count(j_cont.n2, "NE")?; // NE (number of energy points)

                // Validate N1 = 6*(NE+1) before consuming any LIST body.
                // This catches malformed records regardless of whether the INT
                // code is supported, preventing over-/under-consumption of lines.
                // SAMMY only validates this for LFW=1/LRF=1 (File2.cpp line 1031);
                // we validate unconditionally since we actually parse URR data.
                let expected_n1 = 6 * (ne + 1);
                if n1 != expected_n1 {
                    return Err(EndfParseError::UnsupportedFormat(format!(
                        "URR LRF=2 J={aj}: N1={n1} ≠ 6*(NE+1)={expected_n1} (NE={ne})"
                    )));
                }

                // All ENDF interpolation laws (INT=1..5) are now supported
                // in the URR physics module (urr.rs).
                // INT=1: histogram, INT=2: lin-lin, INT=3: log-x/lin-y,
                // INT=4: lin-x/log-y, INT=5: log-log.
                // ENDF-6 §2.2.2.2.

                let values = parse_list_values(ctx.lines, ctx.pos, n1)?;

                // Row 0 (DOF): [0, 0, 0, AMUN, 0, AMUF]
                let amun = values[3];
                let amuf = values[5];

                // Rows 1..NE: [E_i, D_i, GX_i, GN_i, GG_i, GF_i]
                let mut energies = Vec::with_capacity(ne);
                let mut d = Vec::with_capacity(ne);
                let mut gx = Vec::with_capacity(ne);
                let mut gn = Vec::with_capacity(ne);
                let mut gg = Vec::with_capacity(ne);
                let mut gf = Vec::with_capacity(ne);

                for row in 0..ne {
                    let base = (row + 1) * 6; // +1 to skip the DOF row
                    energies.push(values[base]);
                    d.push(values[base + 1]);
                    gx.push(values[base + 2]);
                    gn.push(values[base + 3]);
                    gg.push(values[base + 4]);
                    gf.push(values[base + 5]);
                }

                // Deduplicate energy grid: some evaluations (e.g., JENDL-5
                // Eu-151, Eu-153) contain duplicate energy points. SAMMY
                // silently accepts these. We keep the LAST occurrence of
                // each duplicate energy, matching SAMMY behavior.
                // Exact f64 equality is correct: ENDF duplicates are
                // bitwise-identical copies in the same file record.
                // Issue: #402
                {
                    let n = energies.len();
                    if n > 1 {
                        // O(n) backwards compaction: keep last of each run.
                        let mut write = n - 1;
                        let mut last_e = energies[n - 1];
                        let mut read = n - 1;
                        while read > 0 {
                            read -= 1;
                            if energies[read] == last_e {
                                continue;
                            }
                            write -= 1;
                            energies[write] = energies[read];
                            d[write] = d[read];
                            gx[write] = gx[read];
                            gn[write] = gn[read];
                            gg[write] = gg[read];
                            gf[write] = gf[read];
                            last_e = energies[read];
                        }
                        let new_len = n - write;
                        energies.copy_within(write..n, 0);
                        d.copy_within(write..n, 0);
                        gx.copy_within(write..n, 0);
                        gn.copy_within(write..n, 0);
                        gg.copy_within(write..n, 0);
                        gf.copy_within(write..n, 0);
                        energies.truncate(new_len);
                        d.truncate(new_len);
                        gx.truncate(new_len);
                        gn.truncate(new_len);
                        gg.truncate(new_len);
                        gf.truncate(new_len);
                    }
                }

                // Validate that the (now deduplicated) URR energy grid is
                // strictly ascending (precondition of table_interp).
                for i in 0..energies.len().saturating_sub(1) {
                    if energies[i] >= energies[i + 1] {
                        return Err(EndfParseError::UnsupportedFormat(format!(
                            "URR energy grid must be strictly ascending \
                             (AJ={aj}, index {i}: {} >= {})",
                            energies[i],
                            energies[i + 1]
                        )));
                    }
                }

                j_groups.push(UrrJGroup {
                    j: aj,
                    amun,
                    amuf,
                    energies,
                    d,
                    gx,
                    gn,
                    gg,
                    gf,
                    // INT was validated to be in 1..=5 immediately after
                    // parsing the per-J CONT record, so this cast is safe.
                    // (urr.rs:130-136 dispatches on the full INT=1..=5 set.)
                    int_code: int_code as u32,
                });
            }

            l_groups.push(UrrLGroup { l, awri, j_groups });
        }
    }

    // ENDF-6 §2.2.2: LRF for URR is 1 or 2. Guard before i32→u32 cast.
    debug_assert!(lrf == 1 || lrf == 2, "URR LRF must be 1 or 2, got: {lrf}");

    let urr = UrrData {
        lrf: lrf as u32,
        spi,
        ap,
        e_low: ctx.energy_low,
        e_high: ctx.energy_high,
        l_groups,
    };

    Ok(ResonanceRange {
        energy_low: ctx.energy_low,
        energy_high: ctx.energy_high,
        resolved: false,
        formalism: ResonanceFormalism::Unresolved,
        target_spin: spi,
        scattering_radius: ap,
        naps: ctx.naps,
        ap_table: ctx.ap_table.take(),
        l_groups: Vec::new(),
        rml: None,
        urr: Some(Box::new(urr)),
        r_external: vec![],
    })
}

/// Parse a TAB1 record into a `Tab1` interpolation table.
///
/// ENDF TAB1 layout (Reference: ENDF-6 Formats Manual §0.5):
/// ```text
/// CONT: [C1, C2, L1, L2, NR, NP]
/// NR×2 integer values: (NBT_i, INT_i) pairs  — 6 per line
/// NP×2 float values:   (x_i,   y_i)   pairs  — 6 per line
/// ```
///
/// INT codes: 1=histogram, 2=lin-lin, 3=log-x/lin-y, 4=lin-x/log-y, 5=log-log.
fn parse_tab1(lines: &[&str], pos: &mut usize) -> Result<Tab1, EndfParseError> {
    let cont = parse_cont(lines, pos)?;
    let nr = checked_count(cont.n1, "NR")?; // number of interpolation regions
    let np = checked_count(cont.n2, "NP")?; // number of data points

    // NR=0 is valid ENDF: it means a single implicit interpolation region
    // covering all NP points with no explicit boundary record.  The
    // evaluate() call will fall through to the `unwrap_or(2)` default in
    // interp_code_for_interval(), which correctly returns INT=2 (lin-lin).
    // When NR=0, the loop below is a no-op and the interp_raw vec stays empty.

    // Read NR×2 integers: (NBT, INT) pairs packed as ENDF floats.
    // Validate that values are integers, INT codes are in 1..=5, boundaries
    // are strictly increasing, and the last boundary equals NP.
    let interp_raw = parse_list_values(lines, pos, nr * 2)?;
    let mut boundaries = Vec::with_capacity(nr);
    let mut interp_codes = Vec::with_capacity(nr);
    for i in 0..nr {
        let nbt_raw = interp_raw[i * 2];
        let int_raw = interp_raw[i * 2 + 1];

        // ENDF stores integers as floats (e.g. "2.000000+0").  They must be
        // exact whole numbers.  Use a small epsilon (1e-6) rather than the
        // half-unit tolerance 0.5, which would silently accept 1.4 or 2.49.
        // NBT is a 1-based index (ENDF §0.5), so 0 is invalid.
        if (nbt_raw - nbt_raw.round()).abs() > 1e-6 || nbt_raw < 1.0 {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "TAB1 NBT[{}] is not a positive integer: {}",
                i, nbt_raw
            )));
        }
        if (int_raw - int_raw.round()).abs() > 1e-6 {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "TAB1 INT[{}] is not an integer: {}",
                i, int_raw
            )));
        }
        let int_code = int_raw.round() as u32;
        if !(1..=5).contains(&int_code) {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "TAB1 INT[{}]={} is out of range 1..=5",
                i, int_code
            )));
        }
        let nbt = nbt_raw.round() as usize;

        // Boundaries must be strictly increasing (ENDF §0.5).
        if let Some(&prev) = boundaries.last()
            && nbt <= prev
        {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "TAB1 NBT[{}]={} is not greater than NBT[{}]={}",
                i,
                nbt,
                i - 1,
                prev
            )));
        }
        boundaries.push(nbt);
        interp_codes.push(int_code);
    }

    // The final boundary must equal NP (ENDF §0.5: last NBT is 1-based index of last point).
    if nr > 0 {
        let last_nbt = *boundaries.last().unwrap();
        if last_nbt != np {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "TAB1 last NBT={} does not equal NP={}",
                last_nbt, np
            )));
        }
    }

    if np == 0 {
        return Err(EndfParseError::UnsupportedFormat(
            "TAB1 NP=0: table must have at least one point".to_string(),
        ));
    }

    // Read NP×2 floats: (E, AP) pairs.
    let data_raw = parse_list_values(lines, pos, np * 2)?;
    let mut points = Vec::with_capacity(np);
    for i in 0..np {
        let x = data_raw[i * 2];
        let y = data_raw[i * 2 + 1];
        // x-values must be strictly increasing; Tab1::evaluate() relies on this.
        if let Some(&(x_prev, _)) = points.last()
            && x <= x_prev
        {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "TAB1 x[{}]={} is not greater than x[{}]={} (x must be strictly increasing)",
                i,
                x,
                i - 1,
                x_prev
            )));
        }
        points.push((x, y));
    }

    Ok(Tab1 {
        boundaries,
        interp_codes,
        points,
    })
}

/// Errors from ENDF parsing.
#[derive(Debug, thiserror::Error)]
pub enum EndfParseError {
    #[error("Missing section: {0}")]
    MissingSection(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Invalid number: {0}")]
    InvalidNumber(String),

    #[error("Unexpected end of file: {0}")]
    UnexpectedEof(String),

    #[error("Invalid isotope: {0}")]
    InvalidIsotope(#[from] nereids_core::error::NereidsError),
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: Every ENDF test fixture line must be at least 75 characters long.
    // The MF/MT filter in `parse_endf_file2` checks `line.len() < 75` and
    // discards shorter lines.  ENDF lines are exactly 80 characters in the
    // real format.  If a test fixture line is truncated below 75 chars, it
    // will be silently dropped and the test will fail with "No MF=2, MT=151
    // data found" rather than a useful error.

    #[test]
    fn test_parse_endf_float_standard() {
        // ENDF fields are exactly 11 chars wide, no separators.
        // " 1.23456+2" in 11 chars = " 1.23456+02" (Fortran E11.4 style)
        //  01234567890  (field 0: cols 0-10, field 1: cols 11-21, etc.)
        let line = " 1.23456+02 2.34567-01 0.00000+00                                            ";
        assert!((parse_endf_float(line, 0).unwrap() - 123.456).abs() < 0.01);
        assert!((parse_endf_float(line, 1).unwrap() - 0.234567).abs() < 1e-6);
        assert!((parse_endf_float(line, 2).unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_endf_float_with_e() {
        // 11-char fields: "1.23456E+02" "2.34567E-01"
        let line = "1.23456E+022.34567E-01                                                       ";
        assert!((parse_endf_float(line, 0).unwrap() - 123.456).abs() < 0.01);
        assert!((parse_endf_float(line, 1).unwrap() - 0.234567).abs() < 1e-6);
    }

    #[test]
    fn test_parse_endf_float_negative() {
        let line = "-1.23456+02-2.34567-01                                                       ";
        assert!((parse_endf_float(line, 0).unwrap() - (-123.456)).abs() < 0.01);
        assert!((parse_endf_float(line, 1).unwrap() - (-0.234567)).abs() < 1e-6);
    }

    /// Fortran exponents with a space between the sign and digit — e.g. "9.22330+ 4"
    /// — appear in some older ENDF evaluations (observed in SAMMY tr149/t149a.endf
    /// for U-233).  The parser strips the space before parsing the exponent.
    #[test]
    fn test_parse_endf_float_spaced_exponent() {
        // " 9.22330+ 4" occupies 11 chars: space before mantissa, space before digit
        let line =
            " 9.22330+ 4 1.23400- 2                                                         ";
        assert!((parse_endf_float(line, 0).unwrap() - 92_233.0).abs() < 1.0);
        assert!((parse_endf_float(line, 1).unwrap() - 0.01234).abs() < 1e-6);
    }

    #[test]
    fn test_parse_endf_int() {
        let line = "          0          1          2          3          4          5            ";
        assert_eq!(parse_endf_int(line, 0).unwrap(), 0);
        assert_eq!(parse_endf_int(line, 1).unwrap(), 1);
        assert_eq!(parse_endf_int(line, 2).unwrap(), 2);
    }

    /// Parse the vendored U-238 ENDF file (Reich-Moore, LRF=3).
    ///
    /// This test validates against the public-domain U-238 ENDF/B-VIII.0
    /// evaluation shipped at `examples/data/u238_ex027.endf` (the same
    /// file SAMMY distributes as `samexm_new/ex027_new/ex027.endf`). The
    /// first positive-energy resonance of U-238 is at 6.674 eV.
    ///
    /// Vendored under public-domain ENDF/B redistribution, so this gate
    /// runs unconditionally on CI — no `Skipping…` fall-through.
    #[test]
    fn test_parse_u238_sammy_endf() {
        // Crate-local copy so the test works when nereids-endf is built
        // standalone (outside the workspace, where `examples/data/` is
        // not packaged).  The original `examples/data/u238_ex027.endf`
        // is kept for end-user example code.
        let endf_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/u238_ex027.endf");

        let endf_text = std::fs::read_to_string(&endf_path)
            .unwrap_or_else(|e| panic!("vendored U-238 fixture missing at {endf_path:?}: {e}"));
        let data = parse_endf_file2(&endf_text).unwrap();

        // Basic structure checks.
        assert_eq!(data.za, 92238, "Should be U-238");
        assert!((data.awr - 236.006).abs() < 0.01, "AWR should be ~236");
        assert!(!data.ranges.is_empty(), "Should have at least one range");

        let range = &data.ranges[0];
        assert!(range.resolved, "First range should be resolved");
        assert_eq!(
            range.formalism,
            ResonanceFormalism::ReichMoore,
            "U-238 ENDF uses Reich-Moore (LRF=3)"
        );
        assert!(
            (range.target_spin - 0.0).abs() < 1e-10,
            "U-238 target spin I=0"
        );
        assert!(
            (range.scattering_radius - 9.4285).abs() < 0.01,
            "Scattering radius ~9.4285 fm (ENDF 0.94285 × 10)"
        );
        assert_eq!(range.l_groups.len(), 2, "Should have L=0 and L=1 groups");

        // Check first L-group (L=0).
        let l0 = &range.l_groups[0];
        assert_eq!(l0.l, 0, "First group should be L=0");
        assert!(
            l0.resonances.len() > 500,
            "L=0 should have hundreds of resonances"
        );

        // Find the famous 6.674 eV resonance of U-238.
        let first_positive = l0
            .resonances
            .iter()
            .find(|r| r.energy > 0.0)
            .expect("Should have positive-energy resonances");
        assert!(
            (first_positive.energy - 6.674).abs() < 0.01,
            "First positive resonance should be at 6.674 eV, got {}",
            first_positive.energy
        );
        assert!(
            (first_positive.j - 0.5).abs() < 1e-10,
            "6.674 eV resonance has J=0.5"
        );

        // The 6.674 eV resonance neutron width: ~1.493e-3 eV
        assert!(
            (first_positive.gn - 1.493e-3).abs() < 1e-5,
            "Neutron width should be ~1.493e-3 eV, got {}",
            first_positive.gn
        );
        // Gamma width: ~2.3e-2 eV
        assert!(
            (first_positive.gg - 2.3e-2).abs() < 1e-3,
            "Gamma width should be ~2.3e-2 eV, got {}",
            first_positive.gg
        );

        // The aggregate accessor must cover at least the resolved range's
        // resonances (it sums over every range, including any unresolved
        // range the file carries).
        let total = data.total_resonance_count();
        let resolved_sum: usize = range.l_groups.iter().map(|g| g.resonances.len()).sum();
        assert!(
            total >= resolved_sum && resolved_sum > 500,
            "total_resonance_count ({total}) must cover the resolved range's \
             {resolved_sum} resonances"
        );
    }

    /// Pin the Ta-181 ENDF/B-VIII.0 resonance count at 76 (genuinely-sparse RRR).
    ///
    /// Ta-181 (MAT 7328) in ENDF/B-VIII.0 has NER=2 ranges:
    ///   range 0: LRU=1 resolved, LRF=2 (MLBW), one L-group, 76 discrete
    ///            resonances, resolved region only to 330 eV;
    ///   range 1: LRU=2 unresolved (URR), 0 discrete resonances.
    /// So `total_resonance_count()` == 76 is **faithful**, not a dropped range —
    /// the parser reads every NER range and errors on unconsumed MF2/MT151 data.
    /// ENDF/B-VIII.1 later extended the resolved region (RRR to 2554 eV, 565
    /// resonances); a low VIII.0 count reflects that evaluation's resolved-region
    /// extent, not a parser bug. This test is a regression guard for that fact.
    ///
    /// Vendored under public-domain ENDF/B redistribution (73-Ta-181 LLNL EVAL,
    /// same evaluation NNDC/IAEA distribute) at the workspace-root
    /// `tests/data/endf/Ta-181.endf` (Hf-177 precedent). Inside the full NEREIDS
    /// workspace — where CI runs — the fixture is always present and the gate
    /// asserts fully; a standalone crate build (no workspace fixtures) skips.
    #[test]
    fn test_parse_ta181_endf8_0_resonance_count() {
        let endf_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tests/data/endf/Ta-181.endf");
        let endf_text = match std::fs::read_to_string(&endf_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!(
                    "skipping test_parse_ta181_endf8_0_resonance_count: \
                     fixture not available at {endf_path:?}: {e}. \
                     Run from the full NEREIDS workspace to exercise this gate."
                );
                return;
            }
        };
        let data = parse_endf_file2(&endf_text).unwrap();

        assert_eq!(data.za, 73181, "Should be Ta-181");
        assert!(
            (data.awr - 179.3936).abs() < 0.01,
            "AWR should be ~179.3936, got {}",
            data.awr
        );

        // Faithful count: VIII.0's RRR is genuinely sparse (76), not a drop.
        assert_eq!(
            data.total_resonance_count(),
            76,
            "Ta-181 VIII.0 has 76 discrete resonances (sparse RRR to 330 eV)"
        );
        assert_eq!(
            data.ranges.len(),
            2,
            "Ta-181 VIII.0 has NER=2 ranges (resolved MLBW + unresolved URR)"
        );

        // Range 0: resolved MLBW (LRF=2), all 76 resonances live here.
        let resolved = &data.ranges[0];
        assert!(resolved.resolved, "Range 0 must be resolved (LRU=1)");
        assert_eq!(
            resolved.formalism,
            ResonanceFormalism::MLBW,
            "Ta-181 VIII.0 resolved range uses MLBW (LRF=2)"
        );
        assert!(resolved.rml.is_none(), "MLBW range has no LRF=7 RML data");
        assert!(resolved.urr.is_none(), "Resolved range has no URR data");
        assert_eq!(
            resolved.resonance_count(),
            76,
            "All 76 resonances belong to the resolved MLBW range"
        );

        // Range 1: unresolved (LRU=2, URR), zero discrete resonances.
        let unresolved = &data.ranges[1];
        assert!(!unresolved.resolved, "Range 1 must be unresolved (LRU=2)");
        assert_eq!(
            unresolved.formalism,
            ResonanceFormalism::Unresolved,
            "Ta-181 VIII.0 second range is the URR (LRU=2)"
        );
        assert!(
            unresolved.urr.is_some(),
            "URR range must carry unresolved data"
        );
        assert_eq!(
            unresolved.resonance_count(),
            0,
            "URR range carries no discrete resonances"
        );
    }

    /// Verify KRM=3 resonance column order (offline fixture — no network needed).
    ///
    /// For KRM=3 the per-resonance ENDF layout is [ER, Γγ, Γ_1, ..., Γ_NCH, padding].
    /// The regression checks that `gamma_gamma` comes from position b+1 (Γγ) and
    /// `widths[0]` from position b+2 (Γ_1), NOT the other way round.
    ///
    /// Constructed values:
    ///   res0: ER=10 eV, Γγ=0.025 eV, Γ_1=0.001 eV
    ///   res1: ER=20 eV, Γγ=0.030 eV, Γ_1=0.002 eV
    ///
    /// The fixture is a minimal but fully valid ENDF MF=2/MT=151 block:
    ///   1 isotope, 1 energy range, LRF=7, KRM=3, 1 particle pair, 1 spin group,
    ///   2 resonances, NCH=1 (single elastic neutron channel).
    #[test]
    fn test_krm3_resonance_column_order() {
        // Each ENDF line is exactly 80 chars:
        //   positions  0-65: six 11-char data fields
        //   positions 66-69: MAT (4 chars)
        //   positions 70-71: MF (2 chars)
        //   positions 72-74: MT (3 chars)
        //   positions 75-79: NS (5 chars)
        //
        // Floats use Fortran notation, e.g. "1.000000+1" = 1e1 = 10.0.
        // Integer fields written as right-justified 11-char strings.
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf7_krm3_resonance_column_order.endf");

        let data = parse_endf_file2(ENDF).expect("fixture must parse without error");
        let rml = data.ranges[0]
            .rml
            .as_ref()
            .expect("LRF=7 range must have RmlData");
        let sg = &rml.spin_groups[0];

        assert_eq!(sg.resonances.len(), 2, "spin group must have 2 resonances");

        let res0 = &sg.resonances[0];
        assert!(
            (res0.energy - 10.0).abs() < 1e-10,
            "res0 energy must be 10.0 eV, got {}",
            res0.energy
        );
        // The critical assertions: Γγ must come from column b+1, Γ_1 from column b+2.
        // With the old (buggy) code these two values were swapped.
        assert!(
            (res0.gamma_gamma - 0.025).abs() < 1e-10,
            "res0 gamma_gamma must be 0.025 eV (Gamgam at b+1), got {}",
            res0.gamma_gamma
        );
        assert_eq!(res0.widths.len(), 1, "NCH=1 so widths must have 1 element");
        assert!(
            (res0.widths[0] - 0.001).abs() < 1e-10,
            "res0 widths[0] must be 0.001 eV (Γ_1 at b+2), got {}",
            res0.widths[0]
        );

        let res1 = &sg.resonances[1];
        assert!(
            (res1.energy - 20.0).abs() < 1e-10,
            "res1 energy must be 20.0 eV"
        );
        assert!(
            (res1.gamma_gamma - 0.030).abs() < 1e-10,
            "res1 gamma_gamma must be 0.030 eV, got {}",
            res1.gamma_gamma
        );
        assert!(
            (res1.widths[0] - 0.002).abs() < 1e-10,
            "res1 widths[0] must be 0.002 eV, got {}",
            res1.widths[0]
        );
    }

    /// KRM=2 spin group with an explicit photon capture channel (IPP=2, MA=0).
    ///
    /// Before issue #45 the parser rejected MA<0.5 channels with UnsupportedFormat.
    /// This test verifies that photon channels are now parsed and stored correctly:
    ///   - channels[1] points to the photon particle pair (MT=102)
    ///   - res.widths has two entries: [γ_elastic, γ_photon]
    #[test]
    fn test_krm2_explicit_photon_channel() {
        // Minimal synthetic LRF=7, KRM=2, NJS=1 ENDF snippet.
        // Two particle pairs: pair 1 = n+W184 (MT=2), pair 2 = γ+W185 (MT=102, MA=0).
        // One spin group with 2 channels (elastic + photon); one resonance.
        //
        // Each ENDF line is 80 chars: 6×11-char fields + MAT(4)+MF(2)+MT(3)+NS(5).
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf7_krm2_explicit_photon_channel.endf");

        let data = parse_endf_file2(ENDF).expect("KRM=2 photon channel must parse without error");
        let rml = data.ranges[0]
            .rml
            .as_ref()
            .expect("LRF=7 range must have RmlData");

        assert_eq!(rml.krm, 2, "KRM must be 2");
        assert_eq!(rml.particle_pairs.len(), 2, "must have 2 particle pairs");
        assert!(
            rml.particle_pairs[1].ma < 0.5,
            "pair 2 must be massless (photon)"
        );
        assert_eq!(
            rml.particle_pairs[1].mt, 102,
            "pair 2 must be MT=102 capture"
        );

        let sg = &rml.spin_groups[0];
        assert_eq!(sg.channels.len(), 2, "spin group must have 2 channels");
        assert_eq!(
            sg.channels[0].particle_pair_idx, 0,
            "channel 0 must point to pair 0 (elastic)"
        );
        assert_eq!(
            sg.channels[1].particle_pair_idx, 1,
            "channel 1 must point to pair 1 (photon)"
        );

        assert_eq!(sg.resonances.len(), 1, "must have 1 resonance");
        let res = &sg.resonances[0];
        assert!((res.energy - 10.0).abs() < 1e-10, "energy must be 10 eV");
        assert_eq!(res.widths.len(), 2, "widths must have 2 entries (NCH=2)");
        assert!(
            (res.widths[0] - 0.001).abs() < 1e-10,
            "widths[0] (elastic) must be 0.001, got {}",
            res.widths[0]
        );
        assert!(
            (res.widths[1] - 0.004).abs() < 1e-10,
            "widths[1] (photon) must be 0.004, got {}",
            res.widths[1]
        );
    }

    /// Parse a minimal hand-crafted ENDF snippet with NRO=1 (energy-dependent
    /// scattering radius).
    ///
    /// The fixture encodes:
    /// - LRF=3 (Reich-Moore), NRO=1
    /// - AP TAB1: 2 points — ENDF values 8.0 and 10.0 (10⁻¹² cm),
    ///   which become 80.0 fm and 100.0 fm after ×10 conversion
    /// - One L-group (L=0) with one resonance at 6.674 eV
    ///
    /// Verifies:
    /// - ap_table is Some after parsing
    /// - ap_table.evaluate(1.0) ≈ 80.0 fm (8.0 × ENDF_RADIUS_TO_FM)
    /// - ap_table.evaluate(500.5) ≈ 90.0 fm (midpoint, lin-lin)
    /// - ap_table.evaluate(1000.0) ≈ 100.0 fm
    /// - scattering_radius_at() delegates to the table
    #[test]
    fn test_parse_nro1_tab1() {
        // Each ENDF line is exactly 80 chars: 66 data chars + 14 MAT/MF/MT/SEQ.
        // Cols 67-70: MAT=9237, Cols 71-72: MF=2, Cols 73-75: MT=151, Cols 76-80: seq
        //
        // Line layout (11 chars per field × 6 fields = 66 chars, then 14 control chars):
        //   HEAD:  ZA=92238  AWR=236.006  0  0  NIS=1  0
        //   CONT:  ZAI=92238 ABN=1.0      0  LFW=0 NER=1  0
        //   CONT:  EL=1e-5   EH=1e4    LRU=1  LRF=3  NRO=1  NAPS=0
        //   TAB1 CONT: 0  0  0  0  NR=1  NP=2
        //   TAB1 interp: NBT=2, INT=2  (plus 4 padding zeros)
        //   TAB1 data:   (1.0, 8.0), (1000.0, 10.0)
        //   RM CONT:  SPI=0.0  AP=9.0  0  0  NLS=1  0
        //   L CONT:  AWRI=236.006  0  L=0  0  6*NRS=6  NRS=1
        //   Resonance: ER=6.674  AJ=0.5  GN=1.493e-3  GG=23e-3  GFA=0  GFB=0
        //   SEND: all zeros
        // Each ENDF line: 66 data chars + 4-char MAT(9237) + 2-char MF(" 2")
        //   + 3-char MT("151") + 5-char SEQ = 80 chars total.
        let endf = include_str!("../../../tests/data/synthetic/lrf3_nro1_tab1.endf");

        let data = parse_endf_file2(endf).expect("NRO=1 fixture must parse cleanly");
        assert_eq!(data.ranges.len(), 1, "one energy range");

        let range = &data.ranges[0];
        assert_eq!(
            range.formalism,
            ResonanceFormalism::ReichMoore,
            "must be LRF=3"
        );

        let table = range
            .ap_table
            .as_ref()
            .expect("NRO=1 range must have ap_table");
        assert_eq!(table.points.len(), 2, "TAB1 must have 2 points");

        // Exact boundary values (ENDF 8.0 × 10 = 80.0 fm).
        assert!(
            (table.evaluate(1.0) - 80.0).abs() < 1e-10,
            "AP(1 eV) = 80.0 fm"
        );
        assert!(
            (table.evaluate(1000.0) - 100.0).abs() < 1e-10,
            "AP(1000 eV) = 100.0 fm"
        );
        // Lin-lin midpoint: AP(500.5 eV) ≈ 90.0 fm.
        let mid = table.evaluate(500.5);
        assert!((mid - 90.0).abs() < 0.1, "AP midpoint ≈ 90.0 fm, got {mid}");

        // scattering_radius_at delegates to the table.
        assert!(
            (range.scattering_radius_at(1.0) - 80.0).abs() < 1e-10,
            "scattering_radius_at(1 eV) = 80.0"
        );
        assert!(
            (range.scattering_radius_at(1000.0) - 100.0).abs() < 1e-10,
            "scattering_radius_at(1000 eV) = 100.0"
        );

        // Resonance is still parsed correctly.
        assert_eq!(range.l_groups.len(), 1, "one L-group");
        let res = &range.l_groups[0].resonances[0];
        assert!((res.energy - 6.674).abs() < 1e-6);
    }

    /// LFW=1 with LRF=2 (tabulated widths, U-233-style record).
    ///
    /// SAMMY test tr149 (`t149a.endf`, MAT=9222, ZA=92233) has two ranges:
    ///   - Range 0: LRU=1 (resolved, Reich-Moore / LRF=3)
    ///   - Range 1: LRU=2, LRF=2, **LFW=1** (energy-dependent fission widths)
    ///
    /// ENDF-6 §2.2.2.2: for LFW=1/LRF=2 the per-(L,J) LIST layout is
    /// **identical to LFW=0/LRF=2** (the fission widths are already
    /// per-energy-point in the LIST tail), so the parser dispatches to
    /// the shared `parse_urr_range` path and produces full URR data.
    ///
    /// We previously gated this assertion on a `../SAMMY/...t149a.endf`
    /// sibling checkout; on CI (and on any clean clone) the file was
    /// absent and the test silently reported `ok` after a `Skipping…`
    /// print. Vendoring the full tr149 ENDF would be heavy; instead we
    /// synthesise a minimal but record-shape-faithful LFW=1/LRF=2
    /// fixture so the assertion runs unconditionally.
    #[test]
    fn test_parse_u233_lfw1_lrf2_urr_parsed() {
        // Minimal MF=2/MT=151 with two ranges, mirroring tr149 layout:
        //   Range 0: LRU=1, LRF=3 (Reich-Moore) — one trivial resonance.
        //   Range 1: LRU=2, LRF=2, LFW=1       — NLS=1, NJS=1, NE=2.
        // LFW=1 is flagged on the isotope CONT (L2 field).
        const ENDF: &str = include_str!("../../../tests/data/synthetic/u233_lfw1_lrf2_urr.endf");

        let data = parse_endf_file2(ENDF)
            .expect("U-233 LFW=1/LRF=2 fixture must parse (record layout = LFW=0/LRF=2)");

        // Both ranges must round-trip: resolved + URR.
        assert_eq!(data.ranges.len(), 2, "must have 2 ranges (resolved + URR)");

        let resolved_count = data.ranges.iter().filter(|r| r.resolved).count();
        assert_eq!(resolved_count, 1, "exactly one resolved range");

        let urr_count = data.ranges.iter().filter(|r| r.urr.is_some()).count();
        assert_eq!(urr_count, 1, "LFW=1/LRF=2 URR range must be parsed");

        // Select the URR range by predicate rather than by index — the
        // fixture happens to place the URR at index 1, but the assertion
        // should remain valid if the resolved/URR ordering ever changes.
        let urr = data
            .ranges
            .iter()
            .find_map(|r| r.urr.as_deref())
            .expect("URR range must exist (already asserted by urr_count above)");
        assert_eq!(urr.lrf, 2, "URR LRF must be 2");
        assert_eq!(urr.l_groups.len(), 1, "one L-group");
        let jg = &urr.l_groups[0].j_groups[0];
        assert_eq!(
            jg.gf.len(),
            2,
            "LFW=1/LRF=2 must carry NE per-energy fission widths"
        );
        assert!((jg.gf[0] - 1e-3).abs() < 1e-14, "GF[0]={}", jg.gf[0]);
        assert!((jg.gf[1] - 2e-3).abs() < 1e-14, "GF[1]={}", jg.gf[1]);
        assert!((jg.amuf - 1.0).abs() < 1e-14, "AMUF must round-trip as 1");
    }

    /// Hand-crafted LRF=1 URR roundtrip test.
    ///
    /// Verifies that a minimal synthetic ENDF snippet with LRU=2, LRF=1 is
    /// parsed correctly: one L-group (L=0), two J-groups with known D, AJ,
    /// AMUN, GNO, GG, GF values.
    #[test]
    fn test_parse_lrf1_urr_roundtrip() {
        // Minimal ENDF MF=2/MT=151 with one resolved range followed by one
        // LRU=2 LRF=1 unresolved range.
        //
        // Resolved range: a simple RM LRF=3 with one resonance (gives the
        // parser something valid to consume before the URR section).
        //
        // URR range: LRU=2, LRF=1, NLS=1 (L=0), NJS=2 J-groups.
        //   J=2.0: D=0.5 eV, AMUN=1, GNO=3e-4 eV, GG=3.5e-2 eV, GF=0
        //   J=3.0: D=0.4 eV, AMUN=1, GNO=2e-4 eV, GG=3.0e-2 eV, GF=1e-3 eV
        //
        // Each ENDF line: 66 data chars + MAT(4) MF(2) MT(3) SEQ(5) = 80 chars.
        const ENDF: &str = include_str!("../../../tests/data/synthetic/lrf1_urr_roundtrip.endf");

        let data = parse_endf_file2(ENDF).expect("LRF=1 URR fixture must parse cleanly");

        // Should have 2 ranges: one resolved + one URR.
        assert_eq!(data.ranges.len(), 2, "must have 2 ranges");

        let urr_range = &data.ranges[1];
        assert!(!urr_range.resolved, "URR range must not be resolved");
        assert_eq!(
            urr_range.formalism,
            ResonanceFormalism::Unresolved,
            "formalism must be Unresolved"
        );

        let urr = urr_range
            .urr
            .as_ref()
            .expect("URR range must have urr data");
        assert_eq!(urr.lrf, 1, "LRF must be 1");
        assert!((urr.spi - 2.5).abs() < 1e-10, "SPI must be 2.5");
        assert!((urr.e_low - 600.0).abs() < 1.0, "e_low must be 600 eV");
        assert!(
            (urr.e_high - 30_000.0).abs() < 1.0,
            "e_high must be 30 000 eV"
        );

        assert_eq!(urr.l_groups.len(), 1, "must have 1 L-group");
        let lg = &urr.l_groups[0];
        assert_eq!(lg.l, 0, "L must be 0");
        assert!((lg.awri - 231.038).abs() < 0.001, "AWRI must be 231.038");
        assert_eq!(lg.j_groups.len(), 2, "must have 2 J-groups");

        let jg0 = &lg.j_groups[0];
        assert!((jg0.j - 2.0).abs() < 1e-10, "first J must be 2.0");
        assert!(jg0.energies.is_empty(), "LRF=1 energies must be empty");
        assert!((jg0.d[0] - 0.5).abs() < 1e-10, "D must be 0.5 eV");
        assert!((jg0.amun - 1.0).abs() < 1e-10, "AMUN must be 1.0");
        assert!((jg0.gn[0] - 3e-4).abs() < 1e-14, "GNO must be 3e-4 eV");
        assert!((jg0.gg[0] - 3.5e-2).abs() < 1e-12, "GG must be 3.5e-2 eV");
        assert!((jg0.gf[0] - 0.0).abs() < 1e-14, "GF must be 0");

        let jg1 = &lg.j_groups[1];
        assert!((jg1.j - 3.0).abs() < 1e-10, "second J must be 3.0");
        assert!((jg1.d[0] - 0.4).abs() < 1e-10, "D must be 0.4 eV");
        assert!((jg1.gn[0] - 2e-4).abs() < 1e-14, "GNO must be 2e-4 eV");
        assert!((jg1.gf[0] - 1e-3).abs() < 1e-14, "GF must be 1e-3 eV");
    }

    /// LRF=2 URR with INT=3 (log-x / lin-y) parses successfully.
    ///
    /// Pins issue #553 / M2: between commit 9d7c6bb (which removed the
    /// INT=1/3/4 early-return guard in the URR LRF=2 path and wired the
    /// full INT=1..=5 dispatch in urr.rs) and this PR, a stale
    /// `debug_assert!(int_code == 2 || int_code == 5, …)` survived in
    /// the LIST consumer block. Debug builds therefore panicked on
    /// otherwise valid INT=1, 3, or 4 evaluations; release builds — used
    /// for `cargo test --release` and for end-user binaries — worked
    /// correctly because `debug_assert!` is compiled out.
    ///
    /// This test is **explicitly written to fail in debug builds against
    /// the pre-fix parser** and to pass under both `cargo test` and
    /// `cargo test --release` once the assertion is removed and the INT
    /// code is validated up-front (1..=5).
    #[test]
    fn test_parse_lrf2_urr_int3_roundtrip() {
        // Minimal ENDF MF=2/MT=151 with one LRU=2/LRF=2 range:
        // NLS=1 (L=0), NJS=1, NE=2 energy points, INT=3 (log-x/lin-y).
        // LIST layout: row 0 = [0, 0, 0, AMUN, 0, AMUF]; rows 1..=NE = (E,
        // D, GX, GN, GG, GF). Total = 6*(NE+1) = 18 floats = 3 lines.
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf2_urr_int3_roundtrip.endf");

        let data = parse_endf_file2(ENDF).expect("LRF=2 URR with INT=3 must parse");
        assert_eq!(data.ranges.len(), 1, "must have one URR range");
        let urr = data.ranges[0]
            .urr
            .as_ref()
            .expect("URR data must be present");
        assert_eq!(urr.lrf, 2);
        assert_eq!(urr.l_groups.len(), 1);
        let jg = &urr.l_groups[0].j_groups[0];
        assert_eq!(jg.int_code, 3, "INT code must round-trip as 3");
        assert_eq!(jg.energies.len(), 2);
        assert!((jg.energies[0] - 1e3).abs() < 1e-6);
        assert!((jg.energies[1] - 1e5).abs() < 1e-3);
        assert!((jg.amun - 1.0).abs() < 1e-14);
        assert!((jg.gn[0] - 1e-3).abs() < 1e-14);
        assert!((jg.gn[1] - 2e-3).abs() < 1e-14);
    }

    /// LRF=2 URR with INT=0 is rejected as a hard error.
    ///
    /// ENDF-6 §0.5 defines INT codes 1..=5 only. INT=0 is malformed,
    /// not merely unsupported, so the parser surfaces it as
    /// `UnsupportedFormat("INT=0 out of spec (expected 1..=5)")` rather
    /// than panicking or silently defaulting to lin-lin.
    #[test]
    fn test_parse_lrf2_urr_int0_rejected() {
        // Same skeleton as INT=3 test, but with INT=0 in the J CONT.
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf2_urr_int0_rejected.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        // Match on the error variant structurally so the test only
        // depends on the parser surfacing `UnsupportedFormat`, not on
        // the exact `Display` wording of `EndfParseError`.  Then check
        // the payload string for the diagnostic substrings.
        match err {
            EndfParseError::UnsupportedFormat(ref msg) => {
                assert!(
                    msg.contains("INT=0") && msg.contains("out of spec"),
                    "expected INT-out-of-spec rejection in UnsupportedFormat payload, \
                     got: {msg}"
                );
            }
            other => panic!("expected EndfParseError::UnsupportedFormat for INT=0, got: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Issue #123: robustness tests for malformed input validation
    // -----------------------------------------------------------------------

    /// `checked_count` rejects negative values.
    #[test]
    fn test_checked_count_negative() {
        let err = checked_count(-1, "NLS").unwrap_err();
        assert!(
            err.to_string().contains("Negative"),
            "expected negative error, got: {err}"
        );
    }

    /// `checked_count` rejects values above `MAX_ENDF_COUNT` to prevent
    /// allocation bombs from malformed files.
    #[test]
    fn test_checked_count_upper_bound() {
        // Just above the limit.
        let err = checked_count(MAX_ENDF_COUNT + 1, "NRS").unwrap_err();
        assert!(
            err.to_string().contains("too large"),
            "expected upper-bound error, got: {err}"
        );

        // At the limit: should succeed.
        assert_eq!(checked_count(MAX_ENDF_COUNT, "NRS").unwrap(), 1_000_000);

        // i32::MAX: should be rejected.
        let err = checked_count(i32::MAX, "NPL").unwrap_err();
        assert!(
            err.to_string().contains("too large"),
            "expected upper-bound error for i32::MAX, got: {err}"
        );
    }

    /// `parse_endf_int` rejects non-integral float values rather than
    /// silently truncating.
    ///
    /// Without the strict check, an INT-field value stored as
    /// `"1.900000+0"` would be cast as `1.9_f64 as i32 == 1`, masking
    /// a malformed evaluation.  After the strict check, the parser
    /// surfaces `InvalidNumber` immediately.
    ///
    /// The two field layouts:
    ///   • field 0: integral float "1.000000+0" → returns Ok(1)
    ///   • field 1: non-integral  "1.900000+0" → returns Err(InvalidNumber)
    /// must both round-trip the standard ENDF integer-as-float encoding.
    #[test]
    fn test_parse_endf_int_rejects_non_integral_float() {
        // 11-char fields, padded with leading space to match the ENDF column
        // width.  Field 0 is integral; field 1 is non-integral.
        const LINE: &str = " 1.000000+0 1.900000+0";

        // Integral float must parse cleanly.
        let ok = parse_endf_int(LINE, 0).expect("integral float must parse");
        assert_eq!(ok, 1);

        // Non-integral float must be rejected, not truncated to 1.
        let err = parse_endf_int(LINE, 1).expect_err("non-integral must be rejected");
        match err {
            EndfParseError::InvalidNumber(ref msg) => {
                assert!(
                    msg.contains("Non-integral"),
                    "expected Non-integral diagnostic, got: {msg}"
                );
            }
            other => panic!("expected InvalidNumber, got: {other:?}"),
        }
    }

    /// Negative L-value in a Breit-Wigner range is rejected.
    ///
    /// Constructs a minimal SLBW fixture with L=-1 in the L-group CONT record.
    /// Without the validation, `l_cont.l1 as u32` would wrap to `u32::MAX`.
    #[test]
    fn test_bw_negative_l_rejected() {
        // Minimal SLBW fixture: HEAD + isotope CONT + range CONT + SPI/AP CONT +
        // L-group CONT with L=-1 (field 3 = -1).
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf1_bw_negative_l_rejected.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        assert!(
            err.to_string().contains("negative L"),
            "expected negative L error, got: {err}"
        );
    }

    /// Negative L-value in a Reich-Moore range is rejected.
    #[test]
    fn test_rm_negative_l_rejected() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf3_rm_negative_l_rejected.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        assert!(
            err.to_string().contains("negative L"),
            "expected negative L error, got: {err}"
        );
    }

    /// LRU=0 range with non-zero NLS is rejected.
    ///
    /// ENDF-6 §2.2 says the SPI/AP CONT after an LRU=0 range must have
    /// NLS=0 (no L-groups for scattering-radius-only ranges).
    #[test]
    fn test_lru0_nonzero_nls_rejected() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lru0_nonzero_nls_rejected.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        assert!(
            err.to_string().contains("NLS=3"),
            "expected LRU=0 NLS validation error, got: {err}"
        );
    }

    /// LFW=1/LRF=1 (energy-dependent fission widths) URR is fully parsed.
    ///
    /// ENDF-6 §2.2.2.1 Case B: a shared NE-point energy grid is followed,
    /// for each (L, J), by a full LIST record — a control line
    /// `[0.0, 0.0, L, MUF, NE+6, 0]` and then a body
    /// `[D, AJ, AMUN, GNO, GG, 0] + GF(1..NE)`.  The per-J control line MUST
    /// be consumed before the body; otherwise the line stream misaligns by
    /// one record per J-group and the wrong values are read.
    ///
    /// This fixture is standards-compliant (it includes the per-J control
    /// line), so it fails against a parser that omits that read: the body
    /// `[D, AJ, ...]` would be misread from the control line, yielding
    /// `AJ=0` and `GF=[10, 3]` instead of the values asserted below.
    ///
    /// The fixture has NE=2, NLS=1, NJS=1, MUF=1.
    #[test]
    fn test_lfw1_lrf1_urr_fully_parsed() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lfw1_urr_gracefully_skipped.endf");

        let data = parse_endf_file2(ENDF).expect("LFW=1 URR parse should succeed");
        // LFW=1/LRF=1 is fully parsed — URR data should be present.
        let urr_count = data.ranges.iter().filter(|r| r.urr.is_some()).count();
        assert_eq!(urr_count, 1, "LFW=1/LRF=1 URR should be parsed");
        let urr = data.ranges.iter().find(|r| r.urr.is_some()).unwrap();
        let urr_data = urr.urr.as_ref().unwrap();
        assert_eq!(urr_data.lrf, 1);
        assert_eq!(urr_data.l_groups.len(), 1);
        assert_eq!(urr_data.l_groups[0].l, 0, "L-value");
        assert_eq!(urr_data.l_groups[0].j_groups.len(), 1, "one J-group");

        let jg = &urr_data.l_groups[0].j_groups[0];
        // Scalar (energy-independent) parameters from the LIST body row 0.
        assert!((jg.j - 3.0).abs() < 1e-6, "AJ = {}", jg.j);
        assert!((jg.amun - 1.0).abs() < 1e-6, "AMUN = {}", jg.amun);
        // MUF (fission degrees of freedom) comes from the per-J control L2.
        assert!((jg.amuf - 1.0).abs() < 1e-6, "AMUF (MUF) = {}", jg.amuf);
        assert_eq!(jg.d.len(), 1);
        assert!((jg.d[0] - 10.0).abs() < 1e-6, "D = {}", jg.d[0]);
        assert_eq!(jg.gn.len(), 1);
        assert!((jg.gn[0] - 0.05).abs() < 1e-6, "GNO = {}", jg.gn[0]);
        assert_eq!(jg.gg.len(), 1);
        assert!((jg.gg[0] - 0.04).abs() < 1e-6, "GG = {}", jg.gg[0]);

        // Fission widths are energy-dependent (NE=2 values on the shared grid).
        assert_eq!(jg.energies.len(), 2, "shared energy grid has NE points");
        assert!(
            (jg.energies[0] - 600.0).abs() < 1e-3,
            "E[0] = {}",
            jg.energies[0]
        );
        assert!(
            (jg.energies[1] - 30000.0).abs() < 1e-3,
            "E[1] = {}",
            jg.energies[1]
        );
        assert_eq!(jg.gf.len(), 2, "LFW=1 should have NE fission widths");
        assert!((jg.gf[0] - 0.1).abs() < 1e-6, "GF[0] = {}", jg.gf[0]);
        assert!((jg.gf[1] - 0.2).abs() < 1e-6, "GF[1] = {}", jg.gf[1]);
    }

    /// A per-J LIST control whose `N1 != NE+6` is a malformed Case-B record.
    /// The parser must reject it — this covers the per-J N1 validation guard
    /// (the SCALE `list.getN1()-6 == ener.getNtot()` relation). The fixture is
    /// byte-identical to the valid one except the per-J control N1 (8 -> 7).
    #[test]
    fn test_lfw1_lrf1_urr_rejects_bad_perj_n1() {
        const ENDF: &str = include_str!("../../../tests/data/synthetic/lfw1_urr_bad_perj_n1.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        assert!(
            err.to_string().contains("N1=") && err.to_string().contains("NE+6"),
            "expected per-J N1 != NE+6 rejection, got: {err}"
        );
    }

    /// LRU=0 range with non-zero L1 in SPI/AP CONT is rejected.
    ///
    /// ENDF-6 §2.2: the SPI/AP CONT record for LRU=0 must be
    /// [SPI, AP, 0, 0, NLS=0, 0].  Non-zero L1 or L2 indicates a
    /// malformed or mis-identified record.
    #[test]
    fn test_lru0_nonzero_l1_rejected() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lru0_nonzero_l1_rejected.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        assert!(
            err.to_string().contains("L1=5"),
            "expected LRU=0 L1 validation error, got: {err}"
        );
    }

    /// N1 != 6*NRS in a BW range CONT is rejected.
    #[test]
    fn test_bw_n1_nrs_mismatch_rejected() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf1_bw_n1_nrs_mismatch_rejected.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        assert!(
            err.to_string().contains("N1=7"),
            "expected N1/NRS mismatch error, got: {err}"
        );
    }

    /// Multi-MAT detection: unconsumed MF=2/MT=151 lines after the first
    /// material are rejected.
    #[test]
    fn test_multi_mat_detection() {
        // A valid single-range SLBW file with an extra trailing data line
        // that still carries MF=2/MT=151 tags.
        const ENDF: &str = include_str!("../../../tests/data/synthetic/multi_mat_detection.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        assert!(
            err.to_string().contains("Multiple materials"),
            "expected multi-MAT error, got: {err}"
        );
    }

    /// Verify URR energy deduplication keeps the last occurrence.
    #[test]
    #[allow(clippy::useless_vec)] // Vecs needed for mutation (truncate/copy_within)
    fn test_urr_energy_dedup_keeps_last() {
        // Simulate the dedup logic on mock parallel arrays.
        let mut energies = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0];
        let mut d = vec![10.0, 20.0, 21.0, 30.0, 31.0, 32.0, 40.0];
        let mut gx = vec![0.0; 7];
        let mut gn = vec![0.0; 7];
        let mut gg = vec![0.0; 7];
        let mut gf = vec![0.0; 7];

        let n = energies.len();
        if n > 1 {
            let mut write = n - 1;
            let mut last_e = energies[n - 1];
            let mut read = n - 1;
            while read > 0 {
                read -= 1;
                if energies[read] == last_e {
                    continue;
                }
                write -= 1;
                energies[write] = energies[read];
                d[write] = d[read];
                gx[write] = gx[read];
                gn[write] = gn[read];
                gg[write] = gg[read];
                gf[write] = gf[read];
                last_e = energies[read];
            }
            let new_len = n - write;
            energies.copy_within(write..n, 0);
            d.copy_within(write..n, 0);
            energies.truncate(new_len);
            d.truncate(new_len);
        }

        assert_eq!(energies, [1.0, 2.0, 3.0, 4.0]);
        // d[1]=21.0 (last of the 2.0 pair), d[2]=32.0 (last of the 3.0 triple)
        assert_eq!(d, [10.0, 21.0, 32.0, 40.0]);
    }

    /// MF=2 NIS>1 multi-isotope materials are rejected.
    ///
    /// ENDF-6 §2.1 allows a single material to carry several isotopes, each
    /// with its own ZAI/ABN/NER subsection. NEREIDS's `ResonanceData` cannot
    /// represent that hierarchy without losing per-isotope abundance weights;
    /// rather than silently flatten the ranges into one isotope, the parser
    /// returns `UnsupportedFormat` so a downstream consumer cannot be tricked
    /// into computing an abundance-blind cross section.
    ///
    /// The synthetic fixture is a minimal two-isotope SLBW material: Cu-63
    /// (ZAI=29063, ABN=0.6917) and Cu-65 (ZAI=29065, ABN=0.3083), each with
    /// a single L=0 resonance. The parser should reject the file as soon as
    /// it reads the HEAD record (NIS=2 > 1) and never advance into either
    /// isotope subsection.
    #[test]
    fn test_parse_endf_rejects_nis_gt_1() {
        // Minimal NIS=2 fixture. Both isotope subsections use LRF=1 SLBW so
        // that, if the NIS guard were ever removed, the parser would have a
        // valid stream to walk and the assertion below would still pin the
        // guard. The HEAD's ZA is set to the natural-element identifier
        // ZA=29000 (nat-Cu); for NIS=1 callers that would be rejected by
        // `isotope_from_za` with an "A=0" error, but the NIS=2 check runs
        // first and returns the expected UnsupportedFormat.
        const ENDF: &str = include_str!("../../../tests/data/synthetic/nis_gt_1_rejected.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        match &err {
            EndfParseError::UnsupportedFormat(msg) => {
                assert!(
                    msg.contains("NIS=2"),
                    "expected NIS=2 in error message, got: {msg}"
                );
                assert!(
                    msg.contains("multi-isotope"),
                    "expected 'multi-isotope' in error message, got: {msg}"
                );
            }
            other => panic!("expected UnsupportedFormat for NIS>1, got {other:?}"),
        }
    }

    /// LRF=7 resonance LIST carries NRS in L2 and NX (packed-row count) in N2.
    ///
    /// ENDF-6 §2.2.1.6: the resonance LIST control record is
    /// `[C1=0, C2=0, L1=0, L2=NRS, N1=6*NX, N2=NX]`. For spin groups whose
    /// per-resonance row fits in one 6-float ENDF line (NCH+1 ≤ 6 for KRM=2,
    /// NCH+2 ≤ 6 for KRM=3), NX numerically equals NRS, so the field
    /// confusion is invisible. This fixture stresses the case where the
    /// per-resonance row requires *more than one* 6-float row, giving
    /// NX > NRS and L2 ≠ N2.
    ///
    /// Construction: KRM=3, NCH=5 (5 elastic channels in a single spin
    /// group), NRS=2.
    ///   per-resonance values = NCH+2 = 7 → 2 packed rows of 6 floats (12
    ///     values per resonance, last 5 are padding zeros).
    ///   NX = NRS · 2 = 4 packed rows.
    ///   NPL = 6·NX = 24 floats.
    ///   resonance LIST control = `[0.0, 0.0, 0, NRS=2, 24, NX=4]`.
    ///
    /// Under the (pre-fix) buggy reader that took NRS from N2, this fixture
    /// would set NRS=4, recompute stride = NPL/NRS = 6, and trip the
    /// min_stride guard (6 < NCH+2 = 7) with a misleading
    /// "stride too small" UnsupportedFormat error. With the fix it reads
    /// NRS=2 (from L2) and stride = NPL/NRS = 12 = 2·6, parses exactly
    /// two resonances at the intended energies, and validates that
    /// each resonance carries 5 partial widths.
    #[test]
    fn test_parse_lrf7_l2_holds_nrs_with_nx_neq_nrs() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf7_l2_holds_nrs_with_nx_neq_nrs.endf");

        let data = parse_endf_file2(ENDF).expect(
            "LRF=7 fixture with NX != NRS must parse without error after the NRS-from-L2 fix",
        );
        let rml = data.ranges[0]
            .rml
            .as_ref()
            .expect("LRF=7 range must have RmlData");
        let sg = &rml.spin_groups[0];

        assert_eq!(
            sg.resonances.len(),
            2,
            "must parse exactly NRS=2 resonances (from L2), not NX=4 (from N2)"
        );
        assert_eq!(sg.channels.len(), 5, "NCH must be 5");
        assert_eq!(
            sg.resonances[0].widths.len(),
            5,
            "each resonance carries NCH=5 partial widths"
        );

        let res0 = &sg.resonances[0];
        assert!(
            (res0.energy - 10.0).abs() < 1e-10,
            "res0 energy must be 10.0 eV, got {}",
            res0.energy
        );
        assert!(
            (res0.gamma_gamma - 0.025).abs() < 1e-10,
            "res0 gamma_gamma must be 0.025 eV (KRM=3, at b+1), got {}",
            res0.gamma_gamma
        );
        assert!(
            (res0.widths[4] - 0.005).abs() < 1e-10,
            "res0 widths[4] must be 0.005 (last channel of multi-row resonance), got {}",
            res0.widths[4]
        );

        let res1 = &sg.resonances[1];
        assert!(
            (res1.energy - 20.0).abs() < 1e-10,
            "res1 energy must be 20.0 eV, got {}",
            res1.energy
        );
        assert!(
            (res1.widths[4] - 0.050).abs() < 1e-10,
            "res1 widths[4] must be 0.050, got {}",
            res1.widths[4]
        );
    }

    /// LRF=7 spin group with nonzero KBK (R-external background) is rejected.
    ///
    /// The ENDF-6 §2.2.1.6 manual prose treats KBK as a nonzero flag with NCH
    /// background records; OpenScale's reference reader
    /// (File2.cpp:444-524) treats KBK as a sparse record count with each
    /// subrecord carrying the channel index in L1 and the LBK formalism flag
    /// in L2. The two conventions disagree on loop bound, per-subrecord
    /// control-field positions, and payload shape per LBK value. No local
    /// ENDF/B-VIII.0 evaluation has nonzero KBK to validate against.
    /// Until a policy decision resolves the dispute, NEREIDS hard-rejects
    /// nonzero KBK so the parser cannot silently misalign the stream past
    /// the offending spin group.
    #[test]
    fn test_parse_lrf7_rejects_nonzero_kbk() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf7_nonzero_kbk_rejected.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        match &err {
            EndfParseError::UnsupportedFormat(msg) => {
                assert!(
                    msg.contains("KBK=1"),
                    "expected KBK=1 in error message, got: {msg}"
                );
            }
            other => panic!("expected UnsupportedFormat for KBK != 0, got {other:?}"),
        }
    }

    /// LRF=7 spin group with nonzero KPS (tabulated phase-shift override) is
    /// rejected. Same documentation-vs-implementation dispute as KBK;
    /// OpenScale itself refuses to read KPS > 0
    /// (File2.cpp:439-441 throws "kps > 0 for lrf=7 not yet supported"), so
    /// NEREIDS adopts the same behaviour rather than guess at a layout.
    #[test]
    fn test_parse_lrf7_rejects_nonzero_kps() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf7_nonzero_kps_rejected.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        match &err {
            EndfParseError::UnsupportedFormat(msg) => {
                assert!(
                    msg.contains("KPS=1"),
                    "expected KPS=1 in error message, got: {msg}"
                );
            }
            other => panic!("expected UnsupportedFormat for KPS != 0, got {other:?}"),
        }
    }

    /// LRF=7 particle pair with PNT=2 ("ASSIGN") is rejected at parse time.
    /// SAMMY's Check_Quantum (rml/mrml03.f:22) rejects Lpent outside {0,1};
    /// neither SAMMY nor NEREIDS implements PNT=2.  Validating up front keeps
    /// the unknown penetrability flag out of the physics evaluator.
    #[test]
    fn test_parse_lrf7_rejects_pnt_two() {
        const ENDF: &str = include_str!("../../../tests/data/synthetic/lrf7_pnt2_rejected.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        match &err {
            EndfParseError::UnsupportedFormat(msg) => {
                assert!(
                    msg.contains("PNT=2"),
                    "expected PNT=2 in error message, got: {msg}"
                );
            }
            other => panic!("expected UnsupportedFormat for PNT=2, got {other:?}"),
        }
    }

    /// LRF=7 massless particle pair (MA=0, photon/eliminated channel) declared
    /// with PNT=1 is rejected: SAMMY always assigns Lpent=0 to the photon
    /// channel, and the physics evaluator's PNT=1 penetrability branch would
    /// otherwise divide by a zero reduced mass.
    #[test]
    fn test_parse_lrf7_rejects_massless_pnt_one() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf7_massless_pnt1_rejected.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        match &err {
            EndfParseError::UnsupportedFormat(msg) => {
                assert!(
                    msg.contains("must have PNT=0"),
                    "expected massless-PNT consistency message, got: {msg}"
                );
            }
            other => panic!("expected UnsupportedFormat for massless PNT=1, got {other:?}"),
        }
    }

    /// LRF=7 particle pair with a fractional PNT (1.5) is rejected before the
    /// f64→i32 narrowing can truncate it into a spurious 0/1 that would bypass
    /// the {0,1} range check.
    #[test]
    fn test_parse_lrf7_rejects_fractional_pnt() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf7_pnt_fractional_rejected.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        match &err {
            EndfParseError::UnsupportedFormat(msg) => {
                assert!(
                    msg.contains("PNT=1.5") && msg.contains("not a finite integer"),
                    "expected fractional-PNT message, got: {msg}"
                );
            }
            other => panic!("expected UnsupportedFormat for fractional PNT, got {other:?}"),
        }
    }

    /// LRF=7 PNT=1 pair with a non-positive mass (MB=0) is rejected up front:
    /// the penetrability path would otherwise form a non-finite reduced mass.
    #[test]
    fn test_parse_lrf7_rejects_pnt1_zero_mass() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf7_pnt1_zero_mass_rejected.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        match &err {
            EndfParseError::UnsupportedFormat(msg) => {
                assert!(
                    msg.contains("finite positive masses"),
                    "expected PNT=1 mass-validation message, got: {msg}"
                );
            }
            other => panic!("expected UnsupportedFormat for PNT=1 zero mass, got {other:?}"),
        }
    }

    /// MF=2 NIS=0 (no isotopes declared) is rejected up-front.
    ///
    /// ENDF-6 §2.1 requires NIS >= 1 for a valid resonance evaluation.
    /// Without the explicit reject, NIS=0 would fall through the per-isotope
    /// loop (zero iterations), leave the resonance section empty, and trip a
    /// confusing downstream "unconsumed data lines" / empty-range failure
    /// far from the actual root cause. The reject mirrors the NIS>1 guard
    /// pattern so both invalid extremes return a clear UnsupportedFormat.
    #[test]
    fn test_parse_endf_rejects_nis_zero() {
        // Minimal NIS=0 fixture: just the HEAD line with NIS=0. The HEAD's
        // ZA=74184 (W-184) is a valid identifier, so any error must come
        // from the NIS=0 guard, not from `isotope_from_za`.
        // HEAD: ZA=74184, AWR=182, NIS=0
        const ENDF: &str = include_str!("../../../tests/data/synthetic/nis_zero_rejected.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        match &err {
            EndfParseError::UnsupportedFormat(msg) => {
                assert!(
                    msg.contains("NIS=0"),
                    "expected NIS=0 in error message, got: {msg}"
                );
                assert!(
                    msg.contains("NIS >= 1"),
                    "expected 'NIS >= 1' guidance in error message, got: {msg}"
                );
            }
            other => panic!("expected UnsupportedFormat for NIS=0, got {other:?}"),
        }
    }

    /// LRF=7 resonance LIST with N2/NX not divisible by L2/NRS is rejected.
    ///
    /// ENDF-6 §2.2.1.6 fixes NX = NRS · ceil(stride/6) where stride is NCH+1
    /// for KRM=2 and NCH+2 for KRM=3, so NX must be an integer multiple of
    /// NRS (the per-resonance packed-row count is constant within a spin
    /// group). A fixture with NRS=4 and NX=2 yields a fractional stride
    /// 6·NX/NRS = 3 floats per resonance, which would mis-align the
    /// resonance reads. Without the divisibility check, the existing
    /// `res_npl == 6*nx` guard passes (12 == 6·2) and the downstream
    /// `res_npl % nrs != 0` would also pass (12 % 4 == 0), producing the
    /// bogus stride. The new guard catches this directly.
    #[test]
    fn test_parse_lrf7_rejects_nx_not_multiple_of_nrs() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf7_nx_not_multiple_of_nrs_rejected.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        match &err {
            EndfParseError::UnsupportedFormat(msg) => {
                assert!(
                    msg.contains("not a multiple"),
                    "expected 'not a multiple' in error message, got: {msg}"
                );
                assert!(
                    msg.contains("NX (2)") || msg.contains("(2)"),
                    "expected NX=2 to appear in error message, got: {msg}"
                );
                assert!(
                    msg.contains("NRS (4)") || msg.contains("(4)"),
                    "expected NRS=4 to appear in error message, got: {msg}"
                );
            }
            other => {
                panic!("expected UnsupportedFormat for NX not multiple of NRS, got {other:?}")
            }
        }
    }

    /// LRF=7 spin group with zero resonances must be accepted when written in
    /// the canonical ENDF-6 §2.2.1.6 form NRS=0, NX=1, NPL=6 (a single
    /// six-float zero-filler row in the LIST body).
    ///
    /// OpenScale's reference writer at
    /// `external/openScale/repo/packages/ScaleUtils/EndfLib/endf/File2.cpp:683-697`
    /// pads the resonance LIST for empty spin groups:
    ///
    /// ```cpp
    /// list.setL2(spin->getNres());        // L2 = NRS = 0
    /// ...
    /// // nx must be at least 1, even if nres=0
    /// if (spin->getNres() == 0)
    ///     nx = 1;
    /// list.setN1(6 * nx);                  // N1 = 6
    /// list.setN2(nx);                      // N2 = 1
    /// ```
    ///
    /// A naive guard `if nrs == 0 && nx != 0 { reject }` would reject this
    /// canonical pattern (NX=1 ≠ 0). The relaxed guard
    /// `if nrs == 0 && nx != 1 { reject }` accepts it while still rejecting
    /// malformed shapes such as NRS=0/NX=2.
    #[test]
    fn test_parse_lrf7_accepts_nrs_zero_nx_one_canonical_empty() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf7_nrs_zero_nx_one_canonical_empty.endf");

        let data = parse_endf_file2(ENDF)
            .expect("LRF=7 fixture with NRS=0/NX=1 canonical empty spin group must parse cleanly");
        let rml = data.ranges[0]
            .rml
            .as_ref()
            .expect("LRF=7 range must have RmlData");
        assert_eq!(
            rml.spin_groups.len(),
            1,
            "must parse exactly one spin group"
        );
        let sg = &rml.spin_groups[0];
        assert!(
            sg.resonances.is_empty(),
            "empty spin group must contain zero resonances, got {}",
            sg.resonances.len()
        );
        assert_eq!(
            sg.channels.len(),
            1,
            "empty spin group still carries its NCH channel definitions"
        );
    }

    /// LRF=7 spin group with NRS=0 but NX≠1 is rejected as malformed.
    ///
    /// OpenScale's writer (File2.cpp:683-697) explicitly pads NX to 1 when
    /// NRS=0, so any NRS=0 record with NX=0 (no filler row) or NX>1 (phantom
    /// filler rows with nothing to anchor them) is not a valid ENDF-6 emission.
    /// The previous over-permissive guard accepted NRS=0/NX=2 silently,
    /// leaving the parser to read two zero-filled rows as "no resonances"
    /// while the LIST body did contain data that some other reader might
    /// interpret as resonance parameters.
    #[test]
    fn test_parse_lrf7_rejects_nrs_zero_nx_two_malformed() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf7_nrs_zero_nx_two_malformed.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        match &err {
            EndfParseError::UnsupportedFormat(msg) => {
                assert!(
                    msg.contains("NRS=0"),
                    "expected 'NRS=0' in error message, got: {msg}"
                );
                assert!(
                    msg.contains("NX=2"),
                    "expected 'NX=2' in error message, got: {msg}"
                );
            }
            other => panic!("expected UnsupportedFormat for NRS=0/NX!=1, got {other:?}"),
        }
    }
}
