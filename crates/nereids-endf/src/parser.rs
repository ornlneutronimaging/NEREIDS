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
                // Unresolved resonance region (LRU=2): parsed and SKIPPED.
                //
                // NEREIDS does not compute URR average cross sections — the
                // Hauser-Feshbach path was removed (it lacked the ENDF
                // width-fluctuation correction, a systematically wrong average).
                // We still consume the URR body to keep the line stream aligned
                // so resolved ranges in the same evaluation remain accessible,
                // and tag the range Unresolved (non-evaluable → Skip in the
                // physics layer). Structural guards inside the skip helpers
                // detect malformed records that would otherwise misalign the
                // cursor. Reference: ENDF-6 §2.2.2.
                let nro_urr = range_cont.n1; // energy-dependent scattering radius flag
                let naps_urr = range_cont.n2; // scattering radius calculation flag
                // A TAB1 AP(E) record precedes the URR body when NRO != 0.
                // Store it (converted to fm) on the placeholder, mirroring the
                // LRF=7 skip: the placeholder preserves everything the header
                // carries even though the range is never evaluated.
                let ap_table_urr = if nro_urr != 0 {
                    let mut tab = parse_tab1(&lines, &mut pos)?;
                    for pt in &mut tab.points {
                        pt.1 *= ENDF_RADIUS_TO_FM;
                    }
                    Some(tab)
                } else {
                    None
                };

                // LRF must be 1 or 2 for the URR (ENDF-6 §2.2.2). Any other
                // value is a malformed URR record; reject it as an unsupported
                // format — consistent with the sibling malformed-record guards
                // (INT range, N1 relations) — rather than heuristically
                // consuming a guessed body and silently dropping the span.
                if lrf != 1 && lrf != 2 {
                    return Err(EndfParseError::UnsupportedFormat(format!(
                        "LRU=2 (URR) with LRF={lrf}: ENDF-6 §2.2.2 restricts \
                         the unresolved region to LRF=1 or LRF=2"
                    )));
                }

                // LFW=1/LRF=1 uses the "Case B" shared-energy-grid layout
                // (ENDF-6 §2.2.2.1); LFW=1/LRF=2 is byte-identical to LFW=0/LRF=2.
                let (spi_urr, ap_urr_fm) = if lfw != 0 && lrf == 1 {
                    skip_urr_lfw1_lrf1(&lines, &mut pos)?
                } else {
                    skip_urr_range(&lines, &mut pos, lrf)?
                };

                all_ranges.push(ResonanceRange {
                    energy_low,
                    energy_high,
                    resolved: false,
                    formalism: ResonanceFormalism::Unresolved,
                    target_spin: spi_urr,
                    scattering_radius: ap_urr_fm,
                    naps: naps_urr,
                    ap_table: ap_table_urr,
                    l_groups: Vec::new(),
                    r_external: vec![],
                });
                continue;
            }

            if lru == 0 {
                // LRU=0: scattering-radius-only range (no resonance parameters).
                // ENDF-6 §2.1: after the range CONT (and optional TAB1 if
                // NRO!=0), a single CONT record follows: [SPI, AP, 0, 0,
                // NLS=0, 0]. We consume the NRO TAB1 if present, then the CONT,
                // and store a non-evaluable placeholder (rather than dropping
                // the range) so the LRU=0 span is named by skip_description and
                // the no-evaluable-content error — a file whose only range is
                // LRU=0 is still rejected (NEREIDS cannot evaluate a
                // resonance-parameter-free stanza, and loading it would yield
                // silent zero cross-sections), but with an accurate message.
                let nro_lru0 = range_cont.n1;
                let naps_lru0 = range_cont.n2;
                let ap_table_lru0 = if nro_lru0 != 0 {
                    // TAB1 AP(E) precedes the SPI/AP CONT; consume and store it.
                    let mut tab = parse_tab1(&lines, &mut pos)?;
                    for pt in &mut tab.points {
                        pt.1 *= ENDF_RADIUS_TO_FM;
                    }
                    Some(tab)
                } else {
                    None
                };
                // CONT: SPI, AP, 0, 0, NLS=0, 0
                // Validate NLS=0 (#123): a non-zero NLS in an LRU=0 range is
                // malformed and would cause the parser to look for L-groups that
                // don't exist, misaligning the cursor for subsequent ranges.
                let spi_cont = parse_cont(&lines, &mut pos)?;
                // ENDF-6 §2.1: the SPI/AP CONT is [SPI, AP, 0, 0, NLS=0, 0].
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
                all_ranges.push(ResonanceRange {
                    energy_low,
                    energy_high,
                    resolved: false,
                    formalism: ResonanceFormalism::ScatteringRadiusOnly,
                    target_spin: spi_cont.c1,
                    scattering_radius: spi_cont.c2 * ENDF_RADIUS_TO_FM,
                    naps: naps_lru0,
                    ap_table: ap_table_lru0,
                    l_groups: Vec::new(),
                    r_external: vec![],
                });
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
                ResonanceFormalism::RMatrixLimited => skip_rmatrix_limited_range(&mut ctx)?,
                ResonanceFormalism::Unresolved => {
                    // Unreachable: Unresolved is only assigned in the LRU=2 branch above.
                    unreachable!("Unresolved formalism should not appear in LRU=1 dispatch");
                }
                ResonanceFormalism::ScatteringRadiusOnly => {
                    // Unreachable: ScatteringRadiusOnly is only assigned in the
                    // LRU=0 branch above.
                    unreachable!(
                        "ScatteringRadiusOnly formalism should not appear in LRU=1 dispatch"
                    );
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

    let data = ResonanceData {
        isotope,
        za,
        awr,
        ranges: all_ranges,
    };

    // Fail loudly when the evaluation carries NO evaluable content. A file
    // whose every range is a parse-and-skip placeholder (LRF=7, LRU=2, or an
    // LRU=0 scattering-radius-only stanza) would otherwise "load" with zero
    // resonances, return zero cross-sections everywhere, and produce
    // transmission ≡ 1 with no signal to the caller. Files with at least one
    // evaluable range still load: parse-and-skip exists for real mixed tapes
    // (e.g. Ta-181, U-238, Pu-240) whose resolved ranges remain fully usable.
    if !data.has_evaluable_range() {
        let detail = if data.ranges.is_empty() {
            "the file carries no resonance-parameter ranges at all".to_string()
        } else {
            let skipped: Vec<String> = data.ranges.iter().map(|r| r.skip_description()).collect();
            format!(
                "every range in this file is a parse-and-skip placeholder: {}",
                skipped.join("; ")
            )
        };
        return Err(EndfParseError::UnsupportedFormat(format!(
            "No evaluable resonance ranges: NEREIDS evaluates resolved LRF=1/2/3 \
             (SLBW/MLBW/Reich-Moore) only, and {detail}. Cross-sections would be \
             identically zero (transmission = 1) over the full energy grid."
        )));
    }

    Ok(data)
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

    // Zero NLS means a resolved range with no L-groups: out of spec
    // (ENDF-6 §2.2.1.1 requires NLS >= 1; NLS=0 belongs to LRU=0
    // scattering-radius-only stanzas) and silently inert if accepted —
    // the range would count as evaluable yet contribute zero cross-section
    // everywhere. Mirrors the NJS=0 guards in the URR skip paths.
    if nls == 0 {
        return Err(EndfParseError::UnsupportedFormat(format!(
            "{formalism:?} range: NLS=0 (ENDF-6 §2.2.1.1 requires at least one \
             L-group in a resolved range; NLS=0 is reserved for LRU=0)"
        )));
    }

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

    // A resolved range whose L-groups are all empty (every NRS=0) is the
    // NLS=0 case in disguise: it would count as evaluable yet contribute
    // zero cross-section everywhere, including potential scattering.
    // Reject it loudly, like the NLS=0 guard above.
    if l_groups.iter().all(|lg| lg.resonances.is_empty()) {
        return Err(EndfParseError::UnsupportedFormat(format!(
            "{formalism:?} range carries no resonances (every L-group has \
             NRS=0): cross-sections would be identically zero over its span"
        )));
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

    // Zero NLS means a resolved range with no L-groups: out of spec
    // (ENDF-6 §2.2.1.1 requires NLS >= 1; NLS=0 belongs to LRU=0
    // scattering-radius-only stanzas) and silently inert if accepted —
    // the range would count as evaluable yet contribute zero cross-section
    // everywhere. Mirrors the NJS=0 guards in the URR skip paths.
    if nls == 0 {
        return Err(EndfParseError::UnsupportedFormat(
            "Reich-Moore range: NLS=0 (ENDF-6 §2.2.1.1 requires at least one \
             L-group in a resolved range; NLS=0 is reserved for LRU=0)"
                .to_string(),
        ));
    }

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

    // A resolved range whose L-groups are all empty (every NRS=0) is the
    // NLS=0 case in disguise: it would count as evaluable yet contribute
    // zero cross-section everywhere, including potential scattering.
    // Reject it loudly, like the NLS=0 guard above.
    if l_groups.iter().all(|lg| lg.resonances.is_empty()) {
        return Err(EndfParseError::UnsupportedFormat(
            "Reich-Moore range carries no resonances (every L-group has \
             NRS=0): cross-sections would be identically zero over its span"
                .to_string(),
        ));
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
        r_external: vec![],
    })
}

/// Consume (skip) an R-Matrix Limited (LRF=7) resolved resonance range,
/// advancing the line cursor past the particle-pair, channel, and resonance
/// LIST records so any following range/material stays aligned.
///
/// NEREIDS does not evaluate LRF=7 cross sections — the RML physics was removed
/// (its closed-channel treatment was incomplete: the Coulomb/SHF=1 closed-channel
/// shift was unimplemented, and the evaluator was never validated against SAMMY)
/// — so the parameters are discarded and the range is tagged
/// `ResonanceFormalism::RMatrixLimited` (non-evaluable → Skip).
///
/// Record advancement and every structural/quantum-flag guard are preserved
/// verbatim (IFG/KRM/KRL; the PNT/SHF/mass particle-pair guards; KBK/KPS; the
/// NCH, IPP-range, NRS/NX/NPL and KRM-dependent stride guards): they reject
/// malformed or unsupported records that would otherwise misalign the stream.
/// Only the RmlData/ParticlePair/RmlChannel/RmlResonance/SpinGroup construction
/// is gone.
///
/// Reference: ENDF-6 Formats Manual §2.2.1.6; SAMMY rml/mrml01.f
fn skip_rmatrix_limited_range(
    ctx: &mut RangeParseContext<'_>,
) -> Result<ResonanceRange, EndfParseError> {
    // CONT: [SPI, AP, IFG, KRM, NJS, KRL]
    // ENDF AP is in 10⁻¹² cm; convert to fm (×10).
    let cont = parse_cont(ctx.lines, ctx.pos)?;
    let target_spin = cont.c1;
    let scattering_radius = cont.c2 * ENDF_RADIUS_TO_FM;
    // IFG (L1): GAM representation flag (ENDF-6 §2.2.1.6). IFG=0 means channel
    // widths GAM are given in eV; IFG=1 means reduced-width amplitudes (√eV).
    // Only IFG=0 is supported; SAMMY's ENDF reader makes the same distinction.
    let ifg = cont.l1;
    if ifg != 0 {
        return Err(EndfParseError::UnsupportedFormat(format!(
            "LRF=7 IFG={ifg} (reduced-width amplitudes) is not supported (only \
             IFG=0, channel widths in eV)"
        )));
    }
    let krm = cont.l2 as u32; // R-matrix type: 2=standard, 3=Reich-Moore approx
    // KRM=0/1/4 are defined in the ENDF spec but not supported here.
    if krm != 2 && krm != 3 {
        return Err(EndfParseError::UnsupportedFormat(format!(
            "LRF=7 KRM={krm} is not supported (only KRM=2 and KRM=3)"
        )));
    }
    let njs = checked_count(cont.n1, "NJS")?; // number of spin groups
    // KRL (N2): kinematics flag. KRL=0 (non-relativistic) is universal;
    // KRL=1 (relativistic) is not supported. SAMMY always writes 0.
    let krl = cont.n2;
    if krl != 0 {
        return Err(EndfParseError::UnsupportedFormat(format!(
            "LRF=7 KRL={krl} (relativistic kinematics) is not supported (only KRL=0)"
        )));
    }

    // LIST: [0, 0, NPP, 0, 12*NPP, NPP]  — particle pair definitions.
    // NPP is authoritative in L1. Reference: ENDF-6 §2.2.1.6 Table 2.1.
    let pp_cont = parse_cont(ctx.lines, ctx.pos)?;
    let npp = checked_count(pp_cont.l1, "NPP")?;
    let pp_values = parse_list_values(ctx.lines, ctx.pos, npp * 12)?;

    // Validate-and-narrow an ENDF integer-coded particle-pair flag (PNT/SHF).
    // A fractional or non-finite f64 is a malformed record and must not be
    // silently truncated/saturated (PNT=1.7→1, PNT=NaN→0 would bypass the {0,1}
    // check below).
    fn pp_int_flag(value: f64, field: &str, idx: usize) -> Result<i32, EndfParseError> {
        if !value.is_finite() || value.fract() != 0.0 {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "LRF=7 particle pair {idx}: {field}={value} is not a finite integer"
            )));
        }
        Ok(value as i32)
    }

    // Particle-pair validation. Values are discarded (the pair defs are not
    // stored), but every quantum-flag guard is applied so malformed records are
    // rejected exactly as before.
    for i in 0..npp {
        let b = i * 12;
        let ma = pp_values[b];
        let mb = pp_values[b + 1];
        let za = pp_values[b + 2];
        let zb = pp_values[b + 3];
        let pnt = pp_int_flag(pp_values[b + 7], "PNT", i)?;
        let shf = pp_int_flag(pp_values[b + 8], "SHF", i)?;
        // PNT (Lpent) must be 0 or 1 (SAMMY rml/mrml03.f:22 Check_Quantum rejects
        // Lpent ∉ {0,1}; PNT=2 "ASSIGN" is unimplemented).
        if pnt != 0 && pnt != 1 {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "LRF=7 particle pair {i}: PNT={pnt} is not supported (only PNT=0 \
                 and PNT=1; SAMMY rejects Lpent outside {{0,1}})"
            )));
        }
        // A massless pair (photon/eliminated channel, MA=0) must carry PNT=0.
        if ma < 0.5 && pnt != 0 {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "LRF=7 particle pair {i}: massless pair (MA={ma}) with PNT={pnt} is \
                 invalid; a photon/eliminated channel must have PNT=0"
            )));
        }
        // PNT=1 requires finite positive masses yielding a finite reduced mass
        // μ = MA·MB/(MA+MB) (computed exactly as the physics did); catches
        // overflow of MA·MB to ∞ and the MA+MB=0 / sign cases.
        if pnt == 1 {
            let reduced_mass = ma * mb / (ma + mb);
            if !(ma.is_finite()
                && mb.is_finite()
                && ma > 0.0
                && mb > 0.0
                && reduced_mass.is_finite()
                && reduced_mass > 0.0)
            {
                return Err(EndfParseError::UnsupportedFormat(format!(
                    "LRF=7 particle pair {i}: PNT=1 requires finite positive masses \
                     yielding a finite reduced mass (MA={ma}, MB={mb})"
                )));
            }
        }
        // Coulomb + SHF=1: closed-channel Coulomb shift at imaginary argument is
        // unimplemented (SAMMY rml/mrml07.f Pghcou only for open channels).
        if za.abs() > 0.5 && zb.abs() > 0.5 && shf == 1 {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "LRF=7 particle pair {i}: Coulomb channel (za={za}, zb={zb}) with \
                 SHF=1 is not supported; closed-channel Coulomb shift at \
                 imaginary rho is not yet implemented"
            )));
        }
    }

    for _ in 0..njs {
        // LIST: [AJ, PJ, KBK, KPS, 6*(NCH+1), NCH+1]
        let sg_cont = parse_cont(ctx.lines, ctx.pos)?;
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

        let npl = checked_count(sg_cont.n1, "NPL")?; // 6*(NCH+1)
        let nch_plus_one = checked_count(sg_cont.n2, "NCH+1")?; // NCH+1

        // NCH+1 <= 1 would imply zero physical channels (NCH = 0), which is
        // meaningless for a resonance range — every spin group must have at
        // least one channel. Reference: ENDF-6 §2.2.1.6; SAMMY rml/mrml01.f.
        if nch_plus_one <= 1 {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "RML spin-group LIST: NCH+1 must be >= 2, got NCH+1={nch_plus_one}"
            )));
        }
        let nch = nch_plus_one - 1;

        let sg_values = parse_list_values(ctx.lines, ctx.pos, npl)?;

        // The LIST must carry at least 6*(NCH+1) values (a dummy header row of
        // zeros followed by NCH channel definitions).
        // Reference: ENDF-6 §2.2.1.6 Table 2.3; SAMMY rml/mrml01.f lines 104-107.
        let expected_npl = 6 * (nch + 1);
        if npl < expected_npl {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "LRF=7 spin-group LIST: NPL={npl} < 6*(NCH+1)={expected_npl}"
            )));
        }

        // Validate each channel's IPP (1-based particle-pair index) is in range;
        // the channel definitions [IPP, L, SCH, BND, APE, APT] are otherwise
        // discarded (LRF=7 is not evaluated). The first 6 values are the dummy
        // header row. Reference: ENDF-6 §2.2.1.6; SAMMY rml/mrml01.f (Ippx test).
        for c in 0..nch {
            let b = 6 + c * 6; // skip the 6-value header row
            let ipp_raw = sg_values[b] as usize;
            if ipp_raw == 0 || ipp_raw > npp {
                return Err(EndfParseError::UnsupportedFormat(format!(
                    "LRF=7 spin-group channel IPP={ipp_raw} is out of range 1..={npp}"
                )));
            }
        }

        // LIST: [0, 0, 0, NRS, 6*NX, NX]  — resonance parameters.
        //
        // ENDF-6 §2.2.1.6 fixes the control fields as [0, 0, 0, L2=NRS, N1=6*NX,
        // N2=NX]: NRS (resonance count) is in L2; NX (packed 6-float row count =
        // NRS·ceil(stride/6), stride NCH+1 for KRM=2 / NCH+2 for KRM=3) is in N2;
        // N1 = 6*NX. For wide spin groups (e.g. F-19, NCH≥5) NX > NRS, so NRS
        // MUST be read from L2 and NX from N2 — mixing them mis-sizes the block.
        // SAMMY mrml01.f:413-415; OpenScale File2.cpp:415,686-697.
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
        // The per-resonance row count is constant within a spin group, so NX is
        // always an integer multiple of NRS; a non-multiple would yield a
        // fractional stride and mis-align resonance reads.
        if nrs > 0 && nx % nrs != 0 {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "LRF=7 resonance LIST: N2/NX ({nx}) is not a multiple of L2/NRS ({nrs}); \
                 ENDF-6 §2.2.1.6 requires NX = NRS * ceil(stride/6) where stride is \
                 NCH+1 for KRM=2 and NCH+2 for KRM=3"
            )));
        }
        // Canonical empty spin group (ENDF-6 §2.2.1.6 + OpenScale File2.cpp:
        // 683-697): NRS=0 carries a single zero-filler row, so NX must be 1.
        if nrs == 0 && nx != 1 {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "LRF=7 resonance LIST: NRS=0 requires NX=1 (single zero-filler row \
                 per ENDF-6 §2.2.1.6 + OpenScale File2.cpp:683-697); got NX={nx}"
            )));
        }
        // Consume the resonance block, then validate the KRM-dependent stride so
        // a malformed row layout is rejected rather than silently accepted.
        // KRM=2 per-resonance row = [ER, γ_1..γ_NCH, pad] → stride ≥ NCH+1;
        // KRM=3 = [ER, Γγ, Γ_1..Γ_NCH, pad] → stride ≥ NCH+2.
        // Reference: ENDF-6 §2.2.1.6; SAMMY rml/mrml01.f ENDF123.
        parse_list_values(ctx.lines, ctx.pos, res_npl)?;
        if nrs > 0 {
            let min_stride = if krm == 3 { nch + 2 } else { nch + 1 };
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
        }
    }

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

/// Consume (skip) a URR LFW=1/LRF=1 "Case B" range body (ENDF-6 §2.2.2.1),
/// advancing the line cursor past the shared fission-width energy LIST and each
/// per-(L,J) LIST record. NEREIDS does not evaluate URR cross sections, so the
/// parameter values are discarded; only the record structure is read, which is
/// what keeps the line stream aligned for any following range or SEND record.
/// Returns the header's `(SPI, AP)` — AP converted to fm — so the caller can
/// preserve them on the placeholder range.
///
/// The per-J control line (carrying `N1 = NE+6`) MUST be consumed before its
/// body — omitting it misaligns the stream by one record per J-group. The
/// `N1 = NE+6` guard is preserved to catch malformed records.
///
/// Reference: ENDF-6 §2.2.2.1 Case B; OpenScale `File2.cpp`
/// (`lfw==1 && lrf==1` branch) and AMPX `File2Unres.f90` — both OpenScale/AMPX
/// readers vendored under SAMMY's external tree, not SAMMY proper.
fn skip_urr_lfw1_lrf1(lines: &[&str], pos: &mut usize) -> Result<(f64, f64), EndfParseError> {
    // CONT: SPI, AP, LSSF, 0, NE, NLS
    let header = parse_cont(lines, pos)?;
    let spi = header.c1;
    let ap_fm = header.c2 * ENDF_RADIUS_TO_FM;
    let ne = checked_count(header.n1, "NE")?;
    let nls = checked_count(header.n2, "NLS")?;

    // LIST: NE shared fission-width-grid energies.
    parse_list_values(lines, pos, ne)?;

    for _ in 0..nls {
        // CONT: AWRI, 0, L, 0, NJS, 0
        let l_cont = parse_cont(lines, pos)?;
        let njs = checked_count(l_cont.n1, "NJS")?;

        // Zero NJS means no J-groups for this L-value, which is malformed
        // (ENDF §2.2.2.1 requires at least one J-group per L-group) and would
        // under-consume the body, surfacing as a confusing downstream error.
        // Mirrors the NJS=0 guards in `skip_urr_range`.
        if njs == 0 {
            return Err(EndfParseError::UnsupportedFormat(format!(
                "URR LFW=1/LRF=1 L={}: NJS=0 (at least one J-group required)",
                l_cont.l1
            )));
        }

        for _ in 0..njs {
            // Per-J LIST control: 0.0, 0.0, L, MUF, NE+6, 0.  Consuming this
            // control record keeps the line stream aligned — the body follows.
            let j_cont = parse_cont(lines, pos)?;
            let n1 = checked_count(j_cont.n1, "N1")?;
            // SCALE validates this exact relation (File2.cpp: N1-6 == NE).
            if n1 != ne + 6 {
                return Err(EndfParseError::UnsupportedFormat(format!(
                    "URR LFW=1/LRF=1: per-J N1={n1} ≠ NE+6={} (NE={ne})",
                    ne + 6
                )));
            }
            // LIST body: [D, AJ, AMUN, GNO, GG, 0] + NE fission widths (discarded).
            parse_list_values(lines, pos, n1)?;
        }
    }

    Ok((spi, ap_fm))
}

/// Consume (skip) a URR (LRU=2) range body of the given LRF, advancing the line
/// cursor past every L- and (L,J)-record. NEREIDS does not evaluate URR cross
/// sections, so the parameter values are discarded; only the record structure
/// is read, which keeps the line stream aligned for any following range or SEND
/// record. Returns the header's `(SPI, AP)` — AP converted to fm — so the
/// caller can preserve them on the placeholder range.
///
/// Handles LFW=0/LRF=1 (energy-independent widths) and LRF=2 (tabulated
/// widths); LFW=1/LRF=2 is byte-identical to LFW=0/LRF=2 and routes here too.
/// LFW=1/LRF=1 (shared-grid "Case B") is handled by `skip_urr_lfw1_lrf1`.
///
/// The structural guards (N1=6*NJS for LRF=1; INT∈1..=5 and N1=6*(NE+1) for
/// LRF=2; NJS>0) are preserved: they catch malformed records that would
/// otherwise over-/under-consume lines and misalign the cursor.
///
/// Reference: ENDF-6 Formats Manual §2.2.2
fn skip_urr_range(lines: &[&str], pos: &mut usize, lrf: i32) -> Result<(f64, f64), EndfParseError> {
    // CONT: SPI, AP, 0, 0, NLS, 0
    let spi_cont = parse_cont(lines, pos)?;
    let spi = spi_cont.c1;
    let ap_fm = spi_cont.c2 * ENDF_RADIUS_TO_FM;
    let nls = checked_count(spi_cont.n1, "NLS")?;

    if lrf == 1 {
        // LRF=1: energy-independent widths, one LIST block per L covering all J.
        for _ in 0..nls {
            // CONT: AWRI, 0, L, 0, N1=6*NJS, N2=NJS
            let l_cont = parse_cont(lines, pos)?;
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
            // LIST body: [D, AJ, AMUN, GNO, GG, GF] × NJS (discarded).
            parse_list_values(lines, pos, n1)?;
        }
    } else {
        // LRF=2: energy-dependent width tables, one LIST per (L, J).
        for _ in 0..nls {
            // CONT: AWRI, 0, L, 0, NJS, 0
            let l_cont = parse_cont(lines, pos)?;
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
            if njs == 0 {
                return Err(EndfParseError::UnsupportedFormat(format!(
                    "URR LRF=2 L={l}: NJS=0 (at least one J-group required)"
                )));
            }

            for _ in 0..njs {
                // CONT: AJ, 0, INT, 0, N1=6*(NE+1), N2=NE
                let j_cont = parse_cont(lines, pos)?;
                let int_code = j_cont.l1; // interpolation law (L1 field)
                // ENDF-6 §0.5 defines INT codes 1..=5; anything outside that
                // range — including negative values and INT=0 or INT≥6 — is a
                // malformed record.
                if !(1..=5).contains(&int_code) {
                    return Err(EndfParseError::UnsupportedFormat(format!(
                        "URR LRF=2: INT={int_code} out of spec (expected 1..=5)"
                    )));
                }
                let n1 = checked_count(j_cont.n1, "N1")?; // 6*(NE+1)
                let ne = checked_count(j_cont.n2, "NE")?; // NE (number of energy points)

                // Validate N1 = 6*(NE+1) before consuming the LIST body so a
                // malformed record cannot over-/under-consume lines.
                let expected_n1 = 6 * (ne + 1);
                if n1 != expected_n1 {
                    return Err(EndfParseError::UnsupportedFormat(format!(
                        "URR LRF=2: N1={n1} ≠ 6*(NE+1)={expected_n1} (NE={ne})"
                    )));
                }
                // LIST body: DOF row + NE width rows (discarded).
                parse_list_values(lines, pos, n1)?;
            }
        }
    }

    Ok((spi, ap_fm))
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
        // The fixture is committed to the repo, so a missing file is a
        // packaging/path regression that MUST fail CI — not a skip that would
        // silently disable this #638 resonance-count guard.
        let endf_text = std::fs::read_to_string(&endf_path).unwrap_or_else(|e| {
            panic!(
                "vendored Ta-181 regression fixture must be present at \
                 {endf_path:?} (committed test data): {e}"
            )
        });
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
        assert_eq!(
            unresolved.resonance_count(),
            0,
            "URR range carries no discrete resonances"
        );

        // Mixed evaluation: the resolved MLBW range keeps the file loadable,
        // and the skipped URR range must be flagged as unevaluated so callers
        // can warn that its span contributes zero cross-section.
        assert!(
            data.has_unevaluated_ranges(),
            "Ta-181's URR range must be flagged as unevaluated"
        );
        assert_eq!(
            data.unevaluated_ranges().len(),
            1,
            "exactly the URR range is unevaluated"
        );
    }

    /// Assert that a fixture whose every range is a parse-and-skip placeholder
    /// is rejected with the no-evaluable-ranges error.
    ///
    /// This still pins cursor advancement: a misaligned skip surfaces as a
    /// *different* error (multi-material guard, unexpected EOF, or a
    /// structural-guard message), not this one — so reaching this specific
    /// message proves the skip consumed exactly the range body.
    #[track_caller]
    fn assert_rejected_no_evaluable_ranges(endf: &str) {
        let err = parse_endf_file2(endf).unwrap_err();
        match err {
            EndfParseError::UnsupportedFormat(ref msg) => {
                assert!(
                    msg.contains("No evaluable resonance ranges"),
                    "expected no-evaluable-ranges rejection, got: {msg}"
                );
            }
            other => panic!("expected UnsupportedFormat, got: {other:?}"),
        }
    }

    /// A valid KRM=3 LRF=7 range is parsed-and-skipped, and because it is the
    /// file's ONLY range the parse is rejected with the no-evaluable-ranges
    /// error (a pure-LRF=7 evaluation would yield zero cross-sections
    /// everywhere). The fixture is a minimal but fully valid ENDF MF=2/MT=151
    /// block (1 isotope, 1 range, LRF=7, KRM=3, 1 particle pair, 1 spin group,
    /// 2 resonances, NCH=1); reaching the no-evaluable-ranges error (rather
    /// than a misalignment error) exercises the KRM=3 stride (NCH+2) in the
    /// skip's resonance-block consumption.
    #[test]
    fn test_lrf7_krm3_rejected_no_evaluable_ranges() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf7_krm3_resonance_column_order.endf");
        assert_rejected_no_evaluable_ranges(ENDF);
    }

    /// The no-evaluable-ranges rejection must NAME each skipped range's
    /// formalism and energy span so the user can see exactly what was in the
    /// file and why it cannot be evaluated.
    #[test]
    fn test_lrf7_rejection_names_formalism_and_span() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf7_krm3_resonance_column_order.endf");
        let err = parse_endf_file2(ENDF).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("LRF=7 (R-Matrix Limited)"),
            "rejection must name the skipped formalism, got: {msg}"
        );
        assert!(
            msg.contains("eV"),
            "rejection must state the skipped energy span, got: {msg}"
        );
        assert!(
            msg.contains("LRF=1/2/3"),
            "rejection must state what NEREIDS evaluates, got: {msg}"
        );
    }

    /// A KRM=2 LRF=7 range with an explicit photon capture channel (IPP=2,
    /// MA=0, MT=102) passes every particle-pair, IPP-range, and stride guard
    /// during the skip — then the file is rejected because the LRF=7 range is
    /// its only range (no evaluable content).
    #[test]
    fn test_lrf7_krm2_photon_channel_rejected_no_evaluable_ranges() {
        // Two particle pairs: pair 1 = n+W184 (MT=2), pair 2 = γ+W185 (MT=102, MA=0);
        // one spin group with 2 channels (elastic + photon); one resonance.
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf7_krm2_explicit_photon_channel.endf");
        assert_rejected_no_evaluable_ranges(ENDF);
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

        // The URR range is parsed-and-skipped: it must appear as a
        // non-resolved Unresolved range. LFW=1/LRF=2 shares the LFW=0/LRF=2
        // record layout, so a mis-sized skip would trip the multi-material
        // guard or drop the resolved range — the counts above pin alignment.
        let urr_count = data
            .ranges
            .iter()
            .filter(|r| r.formalism == ResonanceFormalism::Unresolved)
            .count();
        assert_eq!(
            urr_count, 1,
            "LFW=1/LRF=2 URR range must be present (skipped)"
        );
        let urr_range = data
            .ranges
            .iter()
            .find(|r| r.formalism == ResonanceFormalism::Unresolved)
            .expect("URR range must exist");
        assert!(!urr_range.resolved, "URR range must not be resolved");

        // Mixed evaluation: the resolved range keeps the file loadable, and
        // the skipped URR range must be flagged as unevaluated.
        assert!(data.has_unevaluated_ranges());
        assert_eq!(data.unevaluated_ranges().len(), 1);
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

        // The LRF=1 URR body is parsed-and-skipped; the range is present as
        // Unresolved (asserted above). ranges.len()==2 confirms the skip
        // consumed exactly the URR body and left the line stream aligned.

        // Mixed evaluation: the resolved range keeps the file loadable, and
        // the skipped URR range must be flagged as unevaluated.
        assert!(data.has_unevaluated_ranges());
        assert_eq!(data.unevaluated_ranges().len(), 1);
    }

    /// An LRF=2 URR range with INT=3 (log-x/lin-y) passes the per-J INT guard
    /// (§0.5: INT∈1..=5 — a stricter guard would spuriously reject valid
    /// evaluations), then the file is rejected because the URR range is its
    /// only range (no evaluable content).
    #[test]
    fn test_parse_lrf2_urr_int3_rejected_no_evaluable_ranges() {
        // Minimal ENDF MF=2/MT=151 with one LRU=2/LRF=2 range:
        // NLS=1 (L=0), NJS=1, NE=2 energy points, INT=3 (log-x/lin-y).
        // LIST layout: row 0 = [0, 0, 0, AMUN, 0, AMUF]; rows 1..=NE = (E,
        // D, GX, GN, GG, GF). Total = 6*(NE+1) = 18 floats = 3 lines.
        // Reaching the no-evaluable-ranges error (not an INT rejection or a
        // misalignment error) proves the INT=3 guard accepted the record and
        // the skip consumed exactly the URR body.
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf2_urr_int3_roundtrip.endf");
        assert_rejected_no_evaluable_ranges(ENDF);
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

    /// A resolved (LRU=1) MLBW range with NLS=0 is rejected.
    ///
    /// ENDF-6 §2.2.1.1 requires NLS >= 1 in a resolved range (NLS=0 is
    /// reserved for LRU=0 scattering-radius-only stanzas). Without the guard
    /// this file loads as "evaluable" with zero resonances — zero
    /// cross-section everywhere and no warning.
    #[test]
    fn test_mlbw_nls0_rejected() {
        const ENDF: &str = include_str!("../../../tests/data/synthetic/mlbw_nls0_rejected.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        assert!(
            err.to_string().contains("NLS=0"),
            "expected resolved NLS=0 rejection, got: {err}"
        );
    }

    /// A resolved range with NLS=1 but NRS=0 in every L-group is rejected —
    /// the NLS=0 case in disguise (zero resonances, zero cross-section
    /// everywhere, including potential scattering).
    #[test]
    fn test_mlbw_nrs0_everywhere_rejected() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/mlbw_nrs0_everywhere_rejected.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        assert!(
            err.to_string().contains("carries no resonances"),
            "expected all-empty-L-group rejection, got: {err}"
        );
    }

    /// A resolved (LRU=1) Reich-Moore range with NLS=0 is rejected.
    #[test]
    fn test_rm_nls0_rejected() {
        const ENDF: &str = include_str!("../../../tests/data/synthetic/lrf3_rm_nls0_rejected.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        assert!(
            err.to_string().contains("NLS=0"),
            "expected resolved NLS=0 rejection, got: {err}"
        );
    }

    /// LFW=1/LRF=1 (energy-dependent fission widths) URR is skipped with the
    /// line stream left aligned.
    ///
    /// ENDF-6 §2.2.2.1 Case B: a shared NE-point energy grid is followed,
    /// for each (L, J), by a full LIST record — a control line
    /// `[0.0, 0.0, L, MUF, NE+6, 0]` and then a body
    /// `[D, AJ, AMUN, GNO, GG, 0] + GF(1..NE)`.  The per-J control line MUST
    /// be consumed before the body; otherwise the line stream misaligns by
    /// one record per J-group and the file fails to parse (the stray lines
    /// trip the multi-material guard).
    ///
    /// This fixture is standards-compliant (it includes the per-J control
    /// line); the assertion that it parses end-to-end is the regression guard
    /// for the Case-B skip's cursor advancement.
    ///
    /// The fixture has NE=2, NLS=1, NJS=1, MUF=1.
    #[test]
    fn test_lfw1_lrf1_urr_rejected_no_evaluable_ranges() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lfw1_urr_gracefully_skipped.endf");

        // The Case-B URR body is parsed-and-skipped; consuming the per-J
        // control line (N1=NE+6) keeps the stream aligned. The fixture's only
        // range is the URR, so the aligned parse then hits the
        // no-evaluable-ranges rejection — a misaligned skip would instead trip
        // the multi-material guard or EOF (see the fixture note above), so
        // reaching this specific error pins the cursor advancement.
        assert_rejected_no_evaluable_ranges(ENDF);
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

    /// An LRU=2 (URR) range with LRF ∉ {1, 2} is a malformed record and is
    /// rejected with an error naming the offending LRF.
    ///
    /// ENDF-6 §2.2.2 restricts the unresolved region to LRF=1 or LRF=2. The
    /// parser previously heuristically consumed such a body and dropped the
    /// span silently; it now hard-errors like the sibling malformed-record
    /// guards.
    #[test]
    fn test_urr_lrf3_rejected() {
        const ENDF: &str = include_str!("../../../tests/data/synthetic/urr_lrf3_rejected.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("LRF=3") && msg.contains("URR"),
            "expected URR LRF=3 rejection naming the LRF, got: {msg}"
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

    /// A well-formed LRU=0-only evaluation (scattering radius, no resonance
    /// parameters) is rejected — NEREIDS cannot evaluate it and loading it
    /// would yield silent zero cross-sections — but the message NAMES the
    /// LRU=0 span rather than misreporting an empty file.
    ///
    /// ENDF-6 §2.1: LRU=0 gives a scattering radius but no resonances. The
    /// fixture is byte-identical to `lru0_nonzero_nls_rejected` except the
    /// SPI/AP CONT carries NLS=0 (well-formed), so parsing reaches the
    /// no-evaluable-content guard rather than the NLS validation guard.
    #[test]
    fn test_lru0_only_rejected_names_lru0_span() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lru0_only_rejected_no_evaluable.endf");

        let err = parse_endf_file2(ENDF).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("No evaluable resonance ranges"),
            "expected no-evaluable-ranges rejection, got: {msg}"
        );
        assert!(
            msg.contains("LRU=0 (scattering-radius-only, no resonance parameters)"),
            "rejection must name the LRU=0 span, got: {msg}"
        );
    }

    /// A mixed evaluation carrying an LRU=0 range plus a resolved range still
    /// loads: the resolved range keeps the file evaluable, and the LRU=0 span
    /// is surfaced as a non-evaluable parse-and-skip placeholder (same handling
    /// as LRF=7 / LRU=2).
    #[test]
    fn test_lru0_plus_resolved_mixed_loads_and_flags_lru0() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lru0_plus_resolved_mixed.endf");

        let data = parse_endf_file2(ENDF).expect("mixed LRU=0 + resolved file must load");
        assert_eq!(data.ranges.len(), 2, "LRU=0 range + resolved range");
        assert!(
            data.has_evaluable_range(),
            "resolved range keeps it evaluable"
        );
        assert!(data.has_unevaluated_ranges(), "LRU=0 range is unevaluated");

        let lru0 = data
            .ranges
            .iter()
            .find(|r| r.formalism == ResonanceFormalism::ScatteringRadiusOnly)
            .expect("the LRU=0 range must be present");
        assert!(!lru0.resolved, "LRU=0 range must not be resolved");
        assert!(
            lru0.skip_description()
                .contains("LRU=0 (scattering-radius-only, no resonance parameters)"),
            "LRU=0 skip_description must name the span, got: {}",
            lru0.skip_description()
        );

        // The fixture gives the placeholder DIFFERENT SPI/AP (1.5 / 5.0 fm)
        // than the resolved range (2.5 / 9.6931 fm), so consumers that must
        // prefer the evaluable range are distinguishable from ones reading
        // the placeholder.
        assert_eq!(lru0.target_spin, 1.5);
        assert_eq!(lru0.scattering_radius, 5.0);
        let resolved = data
            .ranges
            .iter()
            .find(|r| r.is_evaluable())
            .expect("the resolved range must be evaluable");
        assert_eq!(resolved.target_spin, 2.5);
        assert!((resolved.scattering_radius - 9.6931).abs() < 1e-9);
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
    /// "stride too small" UnsupportedFormat error. With the fix the skip reads
    /// NRS=2 (from L2), so stride = NPL/NRS = 24/2 = 12 ≥ NCH+2 = 7 and the
    /// range parses end-to-end (loading as a skipped RMatrixLimited range).
    #[test]
    fn test_parse_lrf7_l2_holds_nrs_with_nx_neq_nrs() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf7_l2_holds_nrs_with_nx_neq_nrs.endf");

        // Reading NRS from L2 keeps the stride guard satisfied, so the skip
        // consumes the wide-spin-group range cleanly and the parse reaches the
        // no-evaluable-ranges rejection (the LRF=7 range is the file's only
        // range) instead of the misleading "stride too small" error the
        // NRS-from-N2 misread would produce.
        assert_rejected_no_evaluable_ranges(ENDF);
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
    fn test_parse_lrf7_nrs_zero_nx_one_canonical_empty_passes_guard() {
        const ENDF: &str =
            include_str!("../../../tests/data/synthetic/lrf7_nrs_zero_nx_one_canonical_empty.endf");

        // The canonical empty spin group (NRS=0/NX=1) passes the relaxed
        // guard, so the skip consumes the range cleanly and the parse reaches
        // the no-evaluable-ranges rejection (the LRF=7 range is the file's
        // only range) rather than the NRS=0/NX guard error a stricter check
        // would raise.
        assert_rejected_no_evaluable_ranges(ENDF);
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
