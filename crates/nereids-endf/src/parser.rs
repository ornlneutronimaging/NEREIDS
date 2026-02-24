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
    let nis = head.n1 as usize; // number of isotopes (usually 1)

    let isotope = isotope_from_za(za);
    let mut all_ranges = Vec::new();

    for _ in 0..nis {
        // Isotope CONT: ZAI, ABN, 0, LFW, NER, 0
        let iso_cont = parse_cont(&lines, &mut pos)?;
        let _zai = iso_cont.c1 as u32;
        let _abn = iso_cont.c2; // abundance
        let _lfw = iso_cont.l2; // fission width flag
        let ner = iso_cont.n1 as usize; // number of energy ranges

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
                // ENDF-6 §2.2.2; SAMMY unr/munr01.f90.
                let nro_urr = range_cont.n1;
                let ap_table_urr = if nro_urr != 0 {
                    Some(parse_tab1(&lines, &mut pos)?)
                } else {
                    None
                };

                // LRF=1 and LRF=2 are fully supported.
                // Other values (LRF=3/4 are obsolete ENDF formats) are skipped so
                // the rest of the file still parses.
                if lrf != 1 && lrf != 2 {
                    skip_urr_body(&lines, &mut pos)?;
                    continue;
                }

                let urr_range =
                    parse_urr_range(&lines, &mut pos, lrf, energy_low, energy_high, ap_table_urr)?;
                all_ranges.push(urr_range);
                continue;
            }

            if lru != 1 {
                return Err(EndfParseError::UnsupportedFormat(format!(
                    "LRU={} not supported (expected 1=resolved)",
                    lru
                )));
            }

            let nro = range_cont.n1; // energy-dependent scattering radius flag
            let _naps = range_cont.n2; // scattering radius calculation flag

            // If NRO != 0, a TAB1 record immediately follows giving AP(E).
            // Parse and store it; scattering_radius_at(E) will interpolate it
            // at each energy point.  Reference: ENDF-6 §2.2.1; SAMMY mlb/mmlb1.f90.
            let ap_table = if nro != 0 {
                Some(parse_tab1(&lines, &mut pos)?)
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

            let mut range = match formalism {
                ResonanceFormalism::MLBW | ResonanceFormalism::SLBW => {
                    parse_bw_range(&lines, &mut pos, energy_low, energy_high, formalism)?
                }
                ResonanceFormalism::ReichMoore => {
                    parse_reich_moore_range(&lines, &mut pos, energy_low, energy_high)?
                }
                ResonanceFormalism::RMatrixLimited => {
                    parse_rmatrix_limited_range(&lines, &mut pos, energy_low, energy_high, awr)?
                }
                ResonanceFormalism::Unresolved => {
                    // Unreachable: Unresolved is only assigned in the LRU=2 branch above.
                    unreachable!("Unresolved formalism should not appear in LRU=1 dispatch");
                }
            };
            range.ap_table = ap_table;
            all_ranges.push(range);
        }
    }

    Ok(ResonanceData {
        isotope,
        za,
        awr,
        ranges: all_ranges,
    })
}

/// Parse a Breit-Wigner (SLBW or MLBW) resolved resonance range.
///
/// ENDF-6 File 2, LRF=1:
/// - CONT: SPI, AP, 0, 0, NLS, 0
/// - For each L-value:
///   - CONT: AWRI, 0.0, L, 0, 6*NRS, NRS
///   - LIST: NRS resonances, each 6 values: ER, AJ, GT, GN, GG, GF
///
/// Reference: ENDF-6 Formats Manual Section 2.2.1.1
fn parse_bw_range(
    lines: &[&str],
    pos: &mut usize,
    energy_low: f64,
    energy_high: f64,
    formalism: ResonanceFormalism,
) -> Result<ResonanceRange, EndfParseError> {
    // CONT: SPI, AP, 0, 0, NLS, 0
    let cont = parse_cont(lines, pos)?;
    let target_spin = cont.c1;
    let scattering_radius = cont.c2;
    let nls = cont.n1 as usize; // number of L-values

    let mut l_groups = Vec::with_capacity(nls);

    for _ in 0..nls {
        // CONT: AWRI, 0.0, L, 0, 6*NRS, NRS
        let l_cont = parse_cont(lines, pos)?;
        let awr_l = l_cont.c1;
        let l_val = l_cont.l1 as u32;
        let nrs = l_cont.n2 as usize; // number of resonances

        let mut resonances = Vec::with_capacity(nrs);

        // Each resonance is 6 values on one line (or spanning lines).
        // In ENDF format, LIST records pack 6 values per line.
        let total_values = nrs * 6;
        let values = parse_list_values(lines, pos, total_values)?;

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
            resonances,
        });
    }

    Ok(ResonanceRange {
        energy_low,
        energy_high,
        resolved: true,
        formalism,
        target_spin,
        scattering_radius,
        ap_table: None, // set by caller from NRO TAB1 if present
        l_groups,
        rml: None,
        urr: None,
    })
}

/// Parse a Reich-Moore resolved resonance range.
///
/// ENDF-6 File 2, LRF=2:
/// - CONT: SPI, AP, 0, 0, NLS, 0
/// - For each L-value:
///   - CONT: AWRI, APL, L, 0, 6*NRS, NRS
///   - LIST: NRS resonances, each 6 values: ER, AJ, GN, GG, GFA, GFB
///
/// Reference: ENDF-6 Formats Manual Section 2.2.1.2
/// Reference: SAMMY manual Section 2 (R-matrix theory)
fn parse_reich_moore_range(
    lines: &[&str],
    pos: &mut usize,
    energy_low: f64,
    energy_high: f64,
) -> Result<ResonanceRange, EndfParseError> {
    // CONT: SPI, AP, 0, 0, NLS, 0
    let cont = parse_cont(lines, pos)?;
    let target_spin = cont.c1;
    let scattering_radius = cont.c2;
    let nls = cont.n1 as usize; // number of L-values

    let mut l_groups = Vec::with_capacity(nls);

    for _ in 0..nls {
        // CONT: AWRI, APL, L, 0, 6*NRS, NRS
        let l_cont = parse_cont(lines, pos)?;
        let awr_l = l_cont.c1;
        let apl = l_cont.c2; // L-dependent scattering radius
        let l_val = l_cont.l1 as u32;
        let nrs = l_cont.n2 as usize; // number of resonances

        let mut resonances = Vec::with_capacity(nrs);

        // Each resonance is 6 values: ER, AJ, GN, GG, GFA, GFB
        let total_values = nrs * 6;
        let values = parse_list_values(lines, pos, total_values)?;

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
            resonances,
        });
    }

    Ok(ResonanceRange {
        energy_low,
        energy_high,
        resolved: true,
        formalism: ResonanceFormalism::ReichMoore,
        target_spin,
        scattering_radius,
        ap_table: None, // set by caller from NRO TAB1 if present
        l_groups,
        rml: None,
        urr: None,
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
///   LIST: [0, 0, 0, 0, NPL, NRS]                    ← resonance parameters
///         KRM=2: stride ≥ NCH+1; per resonance: [ER, γ_1, ..., γ_NCH, <padding>]
///         KRM=3: stride ≥ NCH+2; per resonance: [ER, Γγ, Γ_1, ..., Γ_NCH, <padding>]
/// ```
///
/// Reference: ENDF-6 Formats Manual §2.2.1.6; SAMMY rml/mrml01.f
fn parse_rmatrix_limited_range(
    lines: &[&str],
    pos: &mut usize,
    energy_low: f64,
    energy_high: f64,
    awr: f64,
) -> Result<ResonanceRange, EndfParseError> {
    // CONT: [SPI, AP, IFG, KRM, NJS, KRL]
    let cont = parse_cont(lines, pos)?;
    let target_spin = cont.c1;
    let scattering_radius = cont.c2;
    // IFG (L1): radius unit flag.
    //   IFG=0: AP, APE, APT are in fm (10⁻¹² cm) — universal in ENDF/B-VIII.0.
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
    let njs = cont.n1 as usize; // number of spin groups
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
    let pp_cont = parse_cont(lines, pos)?;
    let npp = pp_cont.l1 as usize;
    let pp_values = parse_list_values(lines, pos, npp * 12)?;

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
            pnt: pp_values[b + 7] as i32,
            shf: pp_values[b + 8] as i32,
            mt: pp_values[b + 9] as u32,
            pa: pp_values[b + 10],
            pb: pp_values[b + 11],
        });
    }

    // Coulomb + SHF=1: closed-channel Coulomb shift at imaginary argument is
    // unimplemented.  Reject at parse time rather than silently producing wrong
    // dispersive terms near threshold.
    // Reference: SAMMY rml/mrml07.f — Pghcou is only called for open channels.
    for (i, pp) in particle_pairs.iter().enumerate() {
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
    // - PNT 0/1: distinguished by pp.ma in rmatrix_limited.rs.
    // - SHF 0/1: respected by the shf field in rmatrix_limited.rs.
    // - Coulomb channels (pp.za > 0 && pp.zb > 0): routed through
    //   nereids_physics::coulomb (Steed's CF1+CF2, SAMMY coulomb/mrml08.f90).
    let mut spin_groups = Vec::with_capacity(njs);

    for _ in 0..njs {
        // LIST: [AJ, PJ, KBK, KPS, 6*(NCH+1), NCH+1]
        // First 6*(NCH+1) values: header row [0,0,0,0,0,NCH] then NCH×6 channel defs.
        let sg_cont = parse_cont(lines, pos)?;
        let aj = sg_cont.c1;
        let pj = sg_cont.c2; // explicit parity field; may be 0.0 when parity is in sign(AJ)
        let kbk = sg_cont.l1; // background R-matrix flag
        let kps = sg_cont.l2; // phase shift flag

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
        let npl = sg_cont.n1 as usize; // 6*(NCH+1)
        let nch_plus_one = sg_cont.n2 as usize; // NCH+1
        let nch = nch_plus_one.saturating_sub(1);

        let sg_values = parse_list_values(lines, pos, npl)?;

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
            // The physics code sets P_c=1, S_c=0, φ_c=0 for massless particles
            // (rmatrix_limited.rs, ENDF-6 §2.2.1.6 Note 4) and classifies them as
            // capture channels via pp.mt == 102.  Their reduced width amplitudes
            // appear at the corresponding column position in the resonance rows,
            // exactly like any other channel.  Reference: ENDF-6 §2.2.1.6; SAMMY
            // rml/mrml01.f (Ippx test, mrml07.f P=1 convention for massless).
            channels.push(RmlChannel {
                particle_pair_idx: ipp_raw - 1, // convert 1-based ENDF index to 0-based
                l: sg_values[b + 1] as u32,     // L
                channel_spin: sg_values[b + 2], // SCH
                boundary: sg_values[b + 3],     // BND
                effective_radius: sg_values[b + 4], // APE (fm)
                true_radius: sg_values[b + 5],  // APT (fm)
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

        // LIST: [0, 0, 0, 0, NPL, NRS]  — resonance parameters
        // NPL = total values = NRS × (values-per-resonance).
        // For standard LRF=7, values-per-resonance = NCH+1.
        // For KRM=3 (e.g. W-184 ENDF/B-VIII.0), evaluators pad each resonance row
        // to a fixed 6 values per ENDF line, so NPL/NRS = 6 even when NCH=1.
        // Using hardcoded nch+1 drifts the offset and misreads zeros as energies.
        // Fix: derive stride directly from NPL/NRS; read only NCH widths per row.
        // Reference: ENDF-6 §2.2.1.6; SAMMY rml/mrml01.f (Scan_File_2 resonance loop).
        let res_cont = parse_cont(lines, pos)?;
        let nrs = res_cont.n2 as usize;
        let res_npl = res_cont.n1 as usize;
        let res_values = parse_list_values(lines, pos, res_npl)?;

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
            if !res_npl.is_multiple_of(nrs) {
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

        // KBK: background R-matrix correction (pole-free or smooth background terms).
        // Per ENDF-6 §2.2.1.6: when KBK > 0 there are NCH background sub-records
        // per spin group (one per channel), each consisting of a CONT+LIST pair;
        // if LBK==1 in that CONT, two TAB1 records (real and imaginary parts) follow.
        // Records are consumed to advance the file position; the background correction
        // is NOT applied — matching SAMMY behaviour (mrml10.f is a matrix factorisation
        // utility, not a background R-matrix reader).
        // Reference: OpenScale File2Lrf7.f90 lines 269–298; ENDF-6 §2.2.1.6 Table 2.4.
        if kbk != 0 {
            for _ in 0..nch {
                skip_background_subrecord(lines, pos)?;
            }
        }

        // KPS: tabulated penetrability/phase-shift override.
        // Same record structure as KBK: NCH CONT+LIST pairs (one per channel), with
        // optional TAB1s when LPS==1.  SAMMY always computes penetrabilities and phase
        // shifts analytically (mrml07.f, Sinsix subroutine) and ignores KPS entirely.
        // We match that behaviour: consume the records, do not apply them.
        // Reference: OpenScale File2Lrf7.f90 lines 301–331; ENDF-6 §2.2.1.6 Table 2.5.
        if kps != 0 {
            for _ in 0..nch {
                skip_background_subrecord(lines, pos)?;
            }
        }

        spin_groups.push(SpinGroup {
            j,
            parity,
            channels,
            resonances,
            has_background_correction: kbk != 0 || kps != 0,
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
        energy_low,
        energy_high,
        resolved: true,
        formalism: ResonanceFormalism::RMatrixLimited,
        target_spin,
        scattering_radius,
        ap_table: None, // set by caller from NRO TAB1 if present
        l_groups: Vec::new(),
        rml: Some(Box::new(rml)),
        urr: None,
    })
}

/// Skip a TAB1 record (CONT + NR interpolation pairs + NP data pairs).
fn skip_tab1(lines: &[&str], pos: &mut usize) -> Result<(), EndfParseError> {
    let cont = parse_cont(lines, pos)?;
    let nr = checked_count(cont.n1, "NR")?; // number of interpolation regions
    let np = checked_count(cont.n2, "NP")?; // number of data points
    let nr_lines = (nr * 2).div_ceil(6); // NR×2 integer values (NBT, INT pairs)
    let np_lines = (np * 2).div_ceil(6); // NP×2 float values (x, y pairs)
    let needed = nr_lines + np_lines;
    if *pos + needed > lines.len() {
        return Err(EndfParseError::UnexpectedEof(format!(
            "TAB1 skip needs {needed} lines but only {} remain",
            lines.len() - *pos
        )));
    }
    *pos += needed;
    Ok(())
}

/// Skip an unsupported URR body (CONT + NLS L-groups with their LIST data).
///
/// Called when LRU=2 has an LRF value other than 1 or 2 so that the records
/// for this range are consumed and subsequent ranges can still be parsed.
///
/// Structure consumed (ENDF-6 §2.2.2, all LRF variants share this skeleton):
/// ```text
/// CONT: SPI, AP, 0, 0, NLS, 0
/// For each L (NLS times):
///   CONT: AWRI, 0, L, 0, N1, N2
///   if N2 > 0  → LRF=1 style: one LIST record of N1 values
///   if N2 == 0 → LRF=2 style: N1 J-sub-blocks, each = CONT + LIST(N1_j values)
/// ```
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
    Ok(value as usize)
}

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

/// Skip the tail of an LRF=2 URR section after an unsupported INT code is
/// encountered mid-parse.
///
/// `remaining_j` is the number of J-blocks still to consume in the current
/// L-group (each is CONT+LIST).  `remaining_l` is the number of full
/// L-groups still to consume after the current one (each is CONT + NJS
/// J-blocks of CONT+LIST).
fn skip_remaining_lrf2(
    lines: &[&str],
    pos: &mut usize,
    remaining_j: usize,
    remaining_l: usize,
) -> Result<(), EndfParseError> {
    // Finish the current L-group's remaining J-blocks.
    for _ in 0..remaining_j {
        let j_cont = parse_cont(lines, pos)?;
        let jn1 = checked_count(j_cont.n1, "N1")?;
        parse_list_values(lines, pos, jn1)?;
    }
    // Consume subsequent L-groups in full.
    for _ in 0..remaining_l {
        let l_cont = parse_cont(lines, pos)?;
        let njs = checked_count(l_cont.n1, "NJS")?;
        for _ in 0..njs {
            let j_cont = parse_cont(lines, pos)?;
            let jn1 = checked_count(j_cont.n1, "N1")?;
            parse_list_values(lines, pos, jn1)?;
        }
    }
    Ok(())
}

/// Skip one background sub-record: a CONT+LIST pair plus (if LBK/LPS == 1)
/// two TAB1 records for the real and imaginary tabulated parts.
///
/// Used to consume KBK and KPS background blocks in LRF=7 spin groups.
///
/// Per ENDF-6 §2.2.1.6 and OpenScale File2Lrf7.f90:
/// - CONT: [ED, EU, LBK_or_LPS, <unused>, N1, N2]
///   where L1 (LBK_or_LPS) is the type flag: LBK for KBK blocks, LPS for KPS blocks.
/// - LIST: N1 data values
/// - If LBK_or_LPS == 1: real TAB1 + imaginary TAB1
fn skip_background_subrecord(lines: &[&str], pos: &mut usize) -> Result<(), EndfParseError> {
    let cont = parse_cont(lines, pos)?;
    let lbk_or_lps = cont.l1;
    let n1 = checked_count(cont.n1, "N1")?;
    let list_lines = n1.div_ceil(6);
    if *pos + list_lines > lines.len() {
        return Err(EndfParseError::UnexpectedEof(format!(
            "Background sub-record LIST needs {list_lines} lines but only {} remain",
            lines.len() - *pos
        )));
    }
    *pos += list_lines;
    if lbk_or_lps == 1 {
        skip_tab1(lines, pos)?;
        skip_tab1(lines, pos)?;
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

    Err(EndfParseError::InvalidFloat(format!(
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
    // Try integer parse first, then float-to-int.
    if let Ok(v) = trimmed.parse::<i32>() {
        return Ok(v);
    }

    // Try parsing as float then truncating.
    if let Ok(v) = parse_endf_float(line, field_index) {
        return Ok(v as i32);
    }

    Err(EndfParseError::InvalidFloat(format!(
        "Cannot parse ENDF int: '{}'",
        field
    )))
}

/// Parse an Unresolved Resonance Region (LRU=2) range.
///
/// Supports LRF=1 (energy-independent widths) and LRF=2 (tabulated widths).
///
/// ## Units
/// AP (scattering radius) is stored as-is from the ENDF file.  With IFG=0
/// (the universal convention in ENDF/B-VIII.0), AP is in units of 10⁻¹² cm
/// which is identically 1 fm.  No conversion is needed; the physics layer
/// (`channel::rho`, `urr_cross_sections`) expects fm throughout.
///
/// ## LRF=1 record layout (ENDF-6 §2.2.2.1)
/// ```text
/// CONT: SPI, AP, 0, 0, NLS, 0
/// For each L:
///   CONT: AWRI, 0, L, 0, N1=6*NJS, N2=NJS
///   LIST: NJS × [D, AJ, AMUN, GNO, GG, GF]
/// ```
///
/// ## LRF=2 record layout (ENDF-6 §2.2.2.2)
/// ```text
/// CONT: SPI, AP, 0, 0, NLS, 0
/// For each L:
///   CONT: AWRI, 0, L, 0, NJS, 0
///   For each J:
///     CONT: AJ, 0, INT, 0, N1=6*(NE+1), N2=NE
///     LIST: row 0 = [0,0,0,AMUN,0,AMUF]   (DOF)
///           rows 1..NE = [E,D,GX,GN,GG,GF]
/// ```
///
/// Reference: ENDF-6 Formats Manual §2.2.2; SAMMY `unr/munr03.f90`
fn parse_urr_range(
    lines: &[&str],
    pos: &mut usize,
    lrf: i32,
    energy_low: f64,
    energy_high: f64,
    ap_table: Option<Tab1>,
) -> Result<ResonanceRange, EndfParseError> {
    use crate::resonance::{UrrData, UrrJGroup, UrrLGroup};

    // CONT: SPI, AP, 0, 0, NLS, 0
    // AP is in 10⁻¹² cm ≡ fm (IFG=0); no conversion needed.
    let spi_cont = parse_cont(lines, pos)?;
    let spi = spi_cont.c1;
    let ap = spi_cont.c2; // scattering radius (fm)
    let nls = checked_count(spi_cont.n1, "NLS")?;

    let mut l_groups = Vec::with_capacity(nls);

    if lrf == 1 {
        // LRF=1: energy-independent widths, one LIST block per L covering all J.
        for _ in 0..nls {
            // CONT: AWRI, 0, L, 0, N1=6*NJS, N2=NJS
            let l_cont = parse_cont(lines, pos)?;
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

            let values = parse_list_values(lines, pos, n1)?;

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
        for l_idx in 0..nls {
            // CONT: AWRI, 0, L, 0, NJS, 0
            let l_cont = parse_cont(lines, pos)?;
            let awri = l_cont.c1;
            if l_cont.l1 < 0 {
                return Err(EndfParseError::UnsupportedFormat(format!(
                    "URR LRF=2: negative L={}",
                    l_cont.l1
                )));
            }
            let l = l_cont.l1 as u32;
            let njs = checked_count(l_cont.n1, "NJS")?; // N1 = NJS for LRF=2

            let mut j_groups = Vec::with_capacity(njs);
            for j_idx in 0..njs {
                // CONT: AJ, 0, INT, 0, N1=6*(NE+1), N2=NE
                let j_cont = parse_cont(lines, pos)?;
                let aj = j_cont.c1;
                let int_code = j_cont.l1; // interpolation law (L1 field)
                // Negative INT is a malformed ENDF record, not merely an
                // unimplemented mode — reject it outright.
                if int_code < 0 {
                    return Err(EndfParseError::UnsupportedFormat(format!(
                        "URR LRF=2 J={aj}: negative INT={int_code}"
                    )));
                }
                let n1 = checked_count(j_cont.n1, "N1")?; // 6*(NE+1)
                let ne = checked_count(j_cont.n2, "NE")?; // NE (number of energy points)

                // Supported interpolation laws: INT=2 (lin-lin) and INT=5 (log-log).
                // INT=1/3/4 are valid ENDF but not yet implemented.  Rather
                // than aborting the whole file parse (which would hide usable
                // resolved ranges), consume the remaining LRF=2 body and
                // return the range with urr=None so physics falls back to
                // zero for this energy band.
                // ENDF-6 §2.2.2.2; SAMMY unr/munr01.f90.
                if int_code != 2 && int_code != 5 {
                    // Consume this J-block's LIST record.
                    parse_list_values(lines, pos, n1)?;
                    // Consume any remaining J-blocks in this L-group and
                    // all subsequent L-groups so `pos` is correctly advanced.
                    skip_remaining_lrf2(lines, pos, njs - (j_idx + 1), nls - (l_idx + 1))?;
                    return Ok(ResonanceRange {
                        energy_low,
                        energy_high,
                        resolved: false,
                        formalism: ResonanceFormalism::Unresolved,
                        target_spin: spi,
                        scattering_radius: ap,
                        ap_table,
                        l_groups: Vec::new(),
                        rml: None,
                        urr: None,
                    });
                }

                let expected_n1 = 6 * (ne + 1);
                if n1 != expected_n1 {
                    return Err(EndfParseError::UnsupportedFormat(format!(
                        "URR LRF=2 J={aj}: N1={n1} ≠ 6*(NE+1)={expected_n1} (NE={ne})"
                    )));
                }

                let values = parse_list_values(lines, pos, n1)?;

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

                // Validate that URR energy grid is strictly ascending to satisfy
                // the precondition of table_interp ("xs must be strictly ascending").
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
                    int_code: int_code as u32,
                });
            }

            l_groups.push(UrrLGroup { l, awri, j_groups });
        }
    }

    let urr = UrrData {
        lrf: lrf as u32,
        spi,
        ap,
        e_low: energy_low,
        e_high: energy_high,
        l_groups,
    };

    Ok(ResonanceRange {
        energy_low,
        energy_high,
        resolved: false,
        formalism: ResonanceFormalism::Unresolved,
        target_spin: spi,
        scattering_radius: ap,
        ap_table,
        l_groups: Vec::new(),
        rml: None,
        urr: Some(Box::new(urr)),
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

    #[error("Invalid float: {0}")]
    InvalidFloat(String),

    #[error("Unexpected end of file: {0}")]
    UnexpectedEof(String),
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Parse the SAMMY ex027 ENDF file for U-238 (Reich-Moore, LRF=3).
    ///
    /// This test validates against the SAMMY-distributed ENDF file.
    /// The first positive-energy resonance of U-238 is at 6.674 eV.
    #[test]
    fn test_parse_u238_sammy_endf() {
        let endf_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("../SAMMY/SAMMY/samexm_new/ex027_new/ex027.endf");

        if !endf_path.exists() {
            eprintln!(
                "Skipping test: SAMMY ENDF file not found at {:?}",
                endf_path
            );
            return;
        }

        let endf_text = std::fs::read_to_string(&endf_path).unwrap();
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
            (range.scattering_radius - 0.94285).abs() < 0.001,
            "Scattering radius ~0.94285 fm"
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

        let total = data.total_resonance_count();
        println!(
            "U-238 ENDF parsed successfully: {} total resonances across {} L-groups",
            total,
            range.l_groups.len()
        );
        println!(
            "  L=0: {} resonances, L=1: {} resonances",
            l0.resonances.len(),
            range.l_groups[1].resonances.len()
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
        const ENDF: &str = concat!(
            // ── HEAD: ZA=74184, AWR=182, NIS=1 ──────────────────────────────
            " 7.418400+4 1.820000+2          0          0          1          07437 2151    1\n",
            // ── Isotope CONT: NER=1 ──────────────────────────────────────────
            " 7.418400+4 1.000000+0          0          0          1          07437 2151    2\n",
            // ── Range CONT: EL=1e-5, EH=1e3, LRU=1, LRF=7, NRO=0, NAPS=0 ──
            " 1.000000-5 1.000000+3          1          7          0          07437 2151    3\n",
            // ── LRF=7 CONT: SPI=0, AP=0.7, IFG=0, KRM=3, NJS=1, KRL=0 ─────
            " 0.000000+0 7.000000-1          0          3          1          07437 2151    4\n",
            // ── Particle-pair LIST CONT: NPP=1, NPL=12 ───────────────────────
            " 0.000000+0 0.000000+0          1          0         12          17437 2151    5\n",
            // Particle pair 1: MA=1, MB=182, ZA=0, ZB=0, IA=0.5, IB=0
            " 1.000000+0 1.820000+2 0.000000+0 0.000000+0 5.000000-1 0.000000+07437 2151    6\n",
            // Q=0, PNT=1, SHF=0, MT=2, PA=1, PB=1
            " 0.000000+0 1.000000+0 0.000000+0 2.000000+0 1.000000+0 1.000000+07437 2151    7\n",
            // ── Spin-group LIST CONT: AJ=0.5, KBK=0, KPS=0, NPL=12, NCH+1=2
            " 5.000000-1 0.000000+0          0          0         12          27437 2151    8\n",
            // Header row (6 zeros, ignored by parser)
            " 0.000000+0 0.000000+0 0.000000+0 0.000000+0 0.000000+0 0.000000+07437 2151    9\n",
            // Channel 0: IPP=1, L=0, SCH=0.5, BND=0, APE=0.7, APT=0.7
            " 1.000000+0 0.000000+0 5.000000-1 0.000000+0 7.000000-1 7.000000-17437 2151   10\n",
            // ── Resonance LIST CONT: NPL=12, NRS=2 ───────────────────────────
            " 0.000000+0 0.000000+0          0          0         12          27437 2151   11\n",
            // res0: ER=10 eV, Γγ=0.025 eV, Γ_1=0.001 eV, (3 padding zeros)
            " 1.000000+1 2.500000-2 1.000000-3 0.000000+0 0.000000+0 0.000000+07437 2151   12\n",
            // res1: ER=20 eV, Γγ=0.030 eV, Γ_1=0.002 eV, (3 padding zeros)
            " 2.000000+1 3.000000-2 2.000000-3 0.000000+0 0.000000+0 0.000000+07437 2151   13\n",
        );

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
        const ENDF: &str = concat!(
            // ── HEAD: ZA=74184, AWR=182, NIS=1 ─────────────────────────────────
            " 7.418400+4 1.820000+2          0          0          1          07437 2151    1\n",
            // ── Isotope CONT: NER=1 ─────────────────────────────────────────────
            " 7.418400+4 1.000000+0          0          0          1          07437 2151    2\n",
            // ── Range CONT: LRU=1, LRF=7, NRO=0 ────────────────────────────────
            " 1.000000-5 1.000000+3          1          7          0          07437 2151    3\n",
            // ── LRF=7 CONT: SPI=0, AP=0.7, IFG=0, KRM=2, NJS=1, KRL=0 ─────────
            " 0.000000+0 7.000000-1          0          2          1          07437 2151    4\n",
            // ── Particle-pair LIST CONT: NPP=2 in L1, N1=24, N2=2 ───────────────
            " 0.000000+0 0.000000+0          2          0         24          27437 2151    5\n",
            // Pair 1 (neutron+W184): MA=1, MB=182, ZA=0, ZB=0, IA=0.5, IB=0
            " 1.000000+0 1.820000+2 0.000000+0 0.000000+0 5.000000-1 0.000000+07437 2151    6\n",
            // Q=0, PNT=1, SHF=0, MT=2, PA=1, PB=1
            " 0.000000+0 1.000000+0 0.000000+0 2.000000+0 1.000000+0 1.000000+07437 2151    7\n",
            // Pair 2 (photon+W185): MA=0 (massless), MB=183, ZA=0, ZB=0, IA=0, IB=0.5
            " 0.000000+0 1.830000+2 0.000000+0 0.000000+0 0.000000+0 5.000000-17437 2151    8\n",
            // Q=6e6 eV (binding), PNT=0, SHF=0, MT=102 (capture), PA=1, PB=1
            " 6.000000+6 0.000000+0 0.000000+0 1.020000+2 1.000000+0 1.000000+07437 2151    9\n",
            // ── Spin-group LIST CONT: AJ=0.5, KBK=0, KPS=0, NPL=18, NCH+1=3 ────
            " 5.000000-1 0.000000+0          0          0         18          37437 2151   10\n",
            // Header row (6 zeros)
            " 0.000000+0 0.000000+0 0.000000+0 0.000000+0 0.000000+0 0.000000+07437 2151   11\n",
            // Channel 0 (elastic): IPP=1, L=0, SCH=0.5, BND=0, APE=0.7, APT=0.7
            " 1.000000+0 0.000000+0 5.000000-1 0.000000+0 7.000000-1 7.000000-17437 2151   12\n",
            // Channel 1 (photon): IPP=2, L=0, SCH=0, BND=0, APE=0, APT=0
            " 2.000000+0 0.000000+0 0.000000+0 0.000000+0 0.000000+0 0.000000+07437 2151   13\n",
            // ── Resonance LIST CONT: NPL=6, NRS=1 ───────────────────────────────
            " 0.000000+0 0.000000+0          0          0          6          17437 2151   14\n",
            // res0: ER=10 eV, γ_elastic=0.001, γ_photon=0.004, 3 padding zeros
            " 1.000000+1 1.000000-3 4.000000-3 0.000000+0 0.000000+0 0.000000+07437 2151   15\n",
        );

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

    /// Parse a real LRF=7 ENDF file (W-184) downloaded from IAEA.
    ///
    /// Run with: cargo test -p nereids-endf -- --ignored test_parse_w184_rml
    ///
    /// Validates: formalism == RMatrixLimited, spin groups non-empty,
    /// first positive resonance energy in plausible range (W-184: ~101.9 eV).
    #[test]
    #[ignore = "requires network: downloads W-184 ENDF from IAEA (~50 kB)"]
    fn test_parse_w184_rml() {
        use crate::retrieval::{EndfLibrary, EndfRetriever};
        use nereids_core::types::Isotope;

        let retriever = EndfRetriever::new();
        let isotope = Isotope::new(74, 184);
        let (_, text) = retriever
            .get_endf_file(&isotope, EndfLibrary::EndfB8_0, 7437)
            .expect("Failed to download W-184 ENDF/B-VIII.0");

        let data = parse_endf_file2(&text).expect("Failed to parse W-184 ENDF");

        assert!(
            !data.ranges.is_empty(),
            "W-184 should have at least one energy range"
        );
        let range = &data.ranges[0];
        assert_eq!(
            range.formalism,
            crate::resonance::ResonanceFormalism::RMatrixLimited,
            "W-184 uses LRF=7 (R-Matrix Limited) in ENDF/B-VIII.0"
        );

        let rml = range.rml.as_ref().expect("LRF=7 range should have RmlData");
        assert!(!rml.particle_pairs.is_empty(), "Should have particle pairs");
        assert!(!rml.spin_groups.is_empty(), "Should have spin groups");

        let total_resonances: usize = rml.spin_groups.iter().map(|sg| sg.resonances.len()).sum();
        assert!(
            total_resonances > 10,
            "W-184 should have many resonances, got {total_resonances}"
        );

        // First positive-energy resonance in W-184 ENDF/B-VIII.0 (KRM=3, J=1/2+ group)
        // is at ~101.95 eV.  The previously assumed 7.6 eV belongs to W-182, not W-184.
        let first_pos_e = rml
            .spin_groups
            .iter()
            .flat_map(|sg| &sg.resonances)
            .map(|r| r.energy)
            .filter(|&e| e > 0.0)
            .fold(f64::MAX, f64::min);
        assert!(
            first_pos_e > 50.0 && first_pos_e < 200.0,
            "First W-184 positive resonance expected ~101.95 eV, got {first_pos_e:.2} eV"
        );

        println!(
            "W-184 LRF=7 parsed: {} spin groups, {} total resonances, \
             first resonance at {:.2} eV",
            rml.spin_groups.len(),
            total_resonances,
            first_pos_e
        );
    }

    /// Parse a minimal hand-crafted ENDF snippet with NRO=1 (energy-dependent
    /// scattering radius).
    ///
    /// The fixture encodes:
    /// - LRF=3 (Reich-Moore), NRO=1
    /// - AP TAB1: 2 points — AP(1 eV)=8.0 fm, AP(1000 eV)=10.0 fm (lin-lin)
    /// - One L-group (L=0) with one resonance at 6.674 eV
    ///
    /// Verifies:
    /// - ap_table is Some after parsing
    /// - ap_table.evaluate(1.0) ≈ 8.0 fm
    /// - ap_table.evaluate(500.5) ≈ 9.0 fm (midpoint, lin-lin)
    /// - ap_table.evaluate(1000.0) ≈ 10.0 fm
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
        let endf = concat!(
            // HEAD: ZA=92238, AWR=236.006, 0, 0, NIS=1, 0
            " 9.223800+4 2.360060+2          0          0          1          09237 2151    1\n",
            // Isotope CONT: ZAI=92238, ABN=1.0, 0, LFW=0, NER=1, 0
            " 9.223800+4 1.000000+0          0          0          1          09237 2151    2\n",
            // Range CONT: EL=1e-5, EH=1e4, LRU=1, LRF=3, NRO=1, NAPS=0
            " 1.000000-5 1.000000+4          1          3          1          09237 2151    3\n",
            // TAB1 CONT: C1=0, C2=0, L1=0, L2=0, NR=1, NP=2
            " 0.000000+0 0.000000+0          0          0          1          29237 2151    4\n",
            // TAB1 interp: NBT=2, INT=2 (4 padding zeros fill the 6-field line)
            "          2          2          0          0          0          09237 2151    5\n",
            // TAB1 data: (1.0, 8.0), (1000.0, 10.0); remaining 2 slots are padding
            " 1.000000+0 8.000000+0 1.000000+3 1.000000+1 0.000000+0 0.000000+09237 2151    6\n",
            // RM CONT: SPI=0.0, AP=9.0, 0, 0, NLS=1, 0
            " 0.000000+0 9.000000+0          0          0          1          09237 2151    7\n",
            // L CONT: AWRI=236.006, 0, L=0, 0, 6*NRS=6, NRS=1
            " 2.360060+2 0.000000+0          0          0          6          19237 2151    8\n",
            // Resonance: ER=6.674, AJ=0.5, GN=1.493e-3, GG=23e-3, GFA=0, GFB=0
            " 6.674000+0 5.000000-1 1.493000-3 2.300000-2 0.000000+0 0.000000+09237 2151    9\n",
        );

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

        // Exact boundary values.
        assert!(
            (table.evaluate(1.0) - 8.0).abs() < 1e-10,
            "AP(1 eV) = 8.0 fm"
        );
        assert!(
            (table.evaluate(1000.0) - 10.0).abs() < 1e-10,
            "AP(1000 eV) = 10.0 fm"
        );
        // Lin-lin midpoint: AP(500.5 eV) ≈ 9.0 fm.
        let mid = table.evaluate(500.5);
        assert!((mid - 9.0).abs() < 0.01, "AP midpoint ≈ 9.0 fm, got {mid}");

        // scattering_radius_at delegates to the table.
        assert!(
            (range.scattering_radius_at(1.0) - 8.0).abs() < 1e-10,
            "scattering_radius_at(1 eV) = 8.0"
        );
        assert!(
            (range.scattering_radius_at(1000.0) - 10.0).abs() < 1e-10,
            "scattering_radius_at(1000 eV) = 10.0"
        );

        // Resonance is still parsed correctly.
        assert_eq!(range.l_groups.len(), 1, "one L-group");
        let res = &range.l_groups[0].resonances[0];
        assert!((res.energy - 6.674).abs() < 1e-6);
    }

    /// Parse the U-233 URR section (LRU=2, LRF=2) from the SAMMY test file tr149.
    ///
    /// Validates the full LRF=2 record structure against the known U-233 file:
    /// - Two L-groups (L=0, L=1)
    /// - Two J-groups per L
    /// - NE=21 energy points per J-group
    /// - First energy = 600 eV, last ≈ 30 000 eV
    ///
    /// Test data: ../SAMMY/SAMMY/sammy/samtry/tr149/t149a.endf (MAT=9222, ZA=92233)
    #[test]
    fn test_parse_u233_urr_lrf2() {
        let endf_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("../SAMMY/SAMMY/sammy/samtry/tr149/t149a.endf");

        if !endf_path.exists() {
            eprintln!(
                "Skipping test: U-233 ENDF file not found at {:?}",
                endf_path
            );
            return;
        }

        let text = std::fs::read_to_string(&endf_path).unwrap();
        let data = parse_endf_file2(&text).expect("U-233 ENDF must parse without error");

        // U-233 has two energy ranges in MAT=9222:
        //   range 0: LRU=1 (resolved, LRF=3 or LRF=2)
        //   range 1: LRU=2 (unresolved, LRF=2)
        let urr_range = data
            .ranges
            .iter()
            .find(|r| r.urr.is_some())
            .expect("U-233 must have at least one URR range");

        let urr = urr_range.urr.as_ref().unwrap();
        assert_eq!(urr.lrf, 2, "LRF must be 2 for U-233 URR");
        assert!((urr.spi - 2.5).abs() < 1e-6, "SPI must be 2.5 for U-233");
        assert!((urr.e_low - 600.0).abs() < 1.0, "e_low must be ~600 eV");
        assert!(
            (urr.e_high - 30_000.0).abs() < 100.0,
            "e_high must be ~30 000 eV"
        );

        assert_eq!(
            urr.l_groups.len(),
            2,
            "U-233 URR must have 2 L-groups (L=0, L=1)"
        );

        let lg0 = &urr.l_groups[0];
        assert_eq!(lg0.l, 0, "First L-group must have L=0");
        assert_eq!(lg0.j_groups.len(), 2, "L=0 must have 2 J-groups");

        let jg0 = &lg0.j_groups[0];
        assert!((jg0.j - 2.0).abs() < 1e-6, "First J-group must have J=2.0");
        assert_eq!(
            jg0.energies.len(),
            21,
            "J=2.0 group must have 21 energy points"
        );
        assert!(
            (jg0.energies[0] - 600.0).abs() < 1.0,
            "First energy must be ~600 eV, got {}",
            jg0.energies[0]
        );
        assert!(
            jg0.energies[20] > 20_000.0,
            "Last energy must be >20 000 eV, got {}",
            jg0.energies[20]
        );

        // Verify DOF is parsed (AMUN ≥ 1 for neutrons).
        assert!(jg0.amun >= 1.0, "AMUN must be ≥ 1, got {}", jg0.amun);

        // Verify widths are positive.
        assert!(jg0.d[0] > 0.0, "Level spacing D must be positive");
        assert!(jg0.gn[0] > 0.0, "Neutron width GN must be positive");
        assert!(jg0.gg[0] > 0.0, "Gamma width GG must be positive");

        println!(
            "U-233 URR parsed: lrf={} spi={} e_low={} e_high={} l_groups={}",
            urr.lrf,
            urr.spi,
            urr.e_low,
            urr.e_high,
            urr.l_groups.len()
        );
        println!(
            "  L=0, J=2.0: NE={} energies[0]={:.0} eV, D[0]={:.4} eV, GN[0]={:.4e} eV, GG[0]={:.4e} eV",
            jg0.energies.len(),
            jg0.energies[0],
            jg0.d[0],
            jg0.gn[0],
            jg0.gg[0]
        );
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
        const ENDF: &str = concat!(
            // ── HEAD: ZA=92233, AWR=231.038, NIS=1 ─────────────────────────
            " 9.223300+4 2.310380+2          0          0          1          09222 2151    1\n",
            // ── Isotope CONT: ZAI=92233, ABN=1.0, LFW=0, NER=2 ─────────────
            " 9.223300+4 1.000000+0          0          0          2          09222 2151    2\n",
            // ── Range 0: EL=1e-5, EH=600, LRU=1, LRF=3 (resolved RM) ───────
            " 1.000000-5 6.000000+2          1          3          0          09222 2151    3\n",
            // RM CONT: SPI=2.5, AP=0.96931, NLS=1
            " 2.500000+0 9.693100-1          0          0          1          09222 2151    4\n",
            // L CONT: AWRI=231.038, APL=0, L=0, NRS=1
            " 2.310380+2 0.000000+0          0          0          6          19222 2151    5\n",
            // One resonance: ER=10 eV, AJ=2.0, GN=1e-3, GG=3.5e-2, GFA=0, GFB=0
            " 1.000000+1 2.000000+0 1.000000-3 3.500000-2 0.000000+0 0.000000+09222 2151    6\n",
            // ── Range 1: EL=600, EH=3e4, LRU=2, LRF=1 (URR) ────────────────
            " 6.000000+2 3.000000+4          2          1          0          09222 2151    7\n",
            // URR CONT: SPI=2.5, AP=0.96931, NLS=1
            " 2.500000+0 9.693100-1          0          0          1          09222 2151    8\n",
            // L=0 CONT: AWRI=231.038, L=0, N1=12(=6*NJS), N2=2(=NJS)
            " 2.310380+2 0.000000+0          0          0         12          29222 2151    9\n",
            // J=2.0: D=0.5,  AJ=2.0, AMUN=1.0, GNO=3e-4,  GG=3.5e-2, GF=0
            " 5.000000-1 2.000000+0 1.000000+0 3.000000-4 3.500000-2 0.000000+09222 2151   10\n",
            // J=3.0: D=0.4,  AJ=3.0, AMUN=1.0, GNO=2e-4,  GG=3.0e-2, GF=1e-3
            " 4.000000-1 3.000000+0 1.000000+0 2.000000-4 3.000000-2 1.000000-39222 2151   11\n",
        );

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
}
