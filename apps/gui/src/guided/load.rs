//! Step 1: Data loading — multi-format TIFF + spectrum file input.
//!
//! Prototype: `.content-area` → cards with drop zones, auto-load,
//! format hints, and a Continue button with data guard.

use std::sync::Arc;

use crate::file_dialog::{DialogIntent, DialogOptions};
use crate::state::{AppState, InputMode, ProvenanceEventKind};
use crate::theme::ThemeColors;
use crate::widgets::design;
use ndarray::Axis;

const INPUT_MODE_LABELS: [&str; 4] = [
    "TIFF Pair + Spectrum",
    "HDF5 Histogram",
    "HDF5 Event",
    "Timepix-1 Raw",
];

const INPUT_MODES: [InputMode; 4] = [
    InputMode::TiffPair,
    InputMode::Hdf5Histogram,
    InputMode::Hdf5Event,
    InputMode::Tpx1Raw,
];

/// Draw the Load step content.
pub fn load_step(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(ui, "Load Data", "Select input format and load files");

    // Input mode tabs — invalidate results when switching modes
    let mut tab_idx = INPUT_MODES
        .iter()
        .position(|&m| m == state.input_mode)
        .unwrap_or(0);
    if design::underline_tabs(ui, &INPUT_MODE_LABELS, &mut tab_idx) {
        state.set_input_mode(INPUT_MODES[tab_idx]);
        if state.input_mode == InputMode::Tpx1Raw {
            state.hdf5_path = None;
            // Raw runs are histogrammed already, so there is no Bin step.
            state.data_type = Some(crate::state::DataType::PreNormalized);
            state.rebuild_pipeline();
        }
        state.hdf5_ob_path = None;
        state.load_error = false;
    }

    ui.add_space(8.0);

    match state.input_mode {
        InputMode::TiffPair => tiff_pair_tab(ui, state),
        InputMode::Hdf5Histogram => hdf5_histogram_tab(ui, state),
        InputMode::Hdf5Event => hdf5_event_tab(ui, state),
        InputMode::Tpx1Raw => tpx1_raw_tab(ui, state),
    }

    // ── Navigation ─────────────────────────────────────────────
    ui.add_space(12.0);
    let can_continue = has_required_data(state);
    let nav_hint = if state.load_error {
        "Loading failed \u{2014} fix files or retry"
    } else {
        "Select files to continue"
    };
    match design::nav_buttons(
        ui,
        Some("\u{2190} Back"),
        "Continue \u{2192}",
        can_continue,
        nav_hint,
    ) {
        design::NavAction::Back => state.nav_prev(),
        design::NavAction::Continue => state.nav_next(),
        design::NavAction::None => {}
    }
}

/// Check whether the minimum data for this input mode is loaded.
fn has_required_data(state: &AppState) -> bool {
    match state.input_mode {
        InputMode::TiffPair => {
            state.sample_data.is_some()
                && state.open_beam_data.is_some()
                && state.spectrum_values.is_some()
        }
        InputMode::Hdf5Histogram => state.sample_data.is_some() && state.spectrum_values.is_some(),
        InputMode::Hdf5Event => {
            // For events, we only need the file selected here;
            // histogramming happens in the Bin step.
            state.hdf5_path.is_some() && state.nexus_metadata.as_ref().is_some_and(|m| m.has_events)
        }
        InputMode::Tpx1Raw => tpx1_loaded(state),
    }
}

/// Both raw Timepix-1 runs, their edges and their live fractions are loaded.
pub(crate) fn tpx1_loaded(state: &AppState) -> bool {
    state.spectrum_values.is_some() && state.loaded_live().is_some()
}

// ── Timepix-1 raw tab ──────────────────────────────────────────

/// Raw Timepix-1 tab: the two run folders and, optionally, each run's NeXus file.
fn tpx1_raw_tab(ui: &mut egui::Ui, state: &mut AppState) {
    design::card(ui, |ui| {
        ui.label(
            egui::RichText::new(
                "Load the raw FITS folders of a single-shutter sample run and open-beam run. \
                 A run's NeXus file gives the pulses that carried beam, so a run paused with \
                 the beam off is corrected right.",
            )
            .size(10.0)
            .color(ThemeColors::from_ctx(ui.ctx()).fg3),
        );
        ui.add_space(8.0);
        for (intent, label, path) in [
            (
                DialogIntent::TiffSample,
                "Sample run folder",
                &state.sample_path,
            ),
            (
                DialogIntent::TiffOpenBeam,
                "Open-beam run folder",
                &state.open_beam_path,
            ),
            (
                DialogIntent::Tpx1SampleNexus,
                "Sample NeXus (optional)",
                &state.hdf5_path,
            ),
            (
                DialogIntent::Tpx1OpenNexus,
                "Open-beam NeXus (optional)",
                &state.hdf5_ob_path,
            ),
        ] {
            let display = path
                .as_ref()
                .map_or(format!("Click to select {label}..."), |p| {
                    p.file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string()
                });
            let folder = matches!(
                intent,
                DialogIntent::TiffSample | DialogIntent::TiffOpenBeam
            );
            let hint = if folder {
                "folder of FITS frames"
            } else {
                "NeXus .nxs.h5 file"
            };
            if design::drop_zone(ui, path.is_some(), &display, hint).clicked() {
                if folder {
                    state.file_dialogs.pick_folder(intent, Default::default());
                } else {
                    state.file_dialogs.pick_file(
                        intent,
                        DialogOptions {
                            filters: vec![("NeXus", &["h5", "hdf5", "nxs"])],
                            ..Default::default()
                        },
                    );
                }
            }
            ui.add_space(6.0);
        }
        let ready = state.sample_path.is_some() && state.open_beam_path.is_some();
        if ui
            .add_enabled(ready && !tpx1_loaded(state), egui::Button::new("Load runs"))
            .clicked()
        {
            load_tpx1_runs(state);
        }
    });
    load_status_ui(ui, state);
    show_loaded_info(ui, state);
}

/// Pulses with proton charge, and their total charge.
fn beam_pulses(path: &std::path::Path) -> Result<(f64, f64), String> {
    let log = nereids_io::runlog::read_run_log(path, "proton_charge").map_err(|e| e.to_string())?;
    if let Some(q) = log.values.iter().find(|q| !q.is_finite()) {
        return Err(format!(
            "{}: proton charge {q} is not finite",
            path.display()
        ));
    }
    let charged: Vec<f64> = log.values.iter().copied().filter(|&q| q > 0.0).collect();
    let total: f64 = charged.iter().sum();
    if !total.is_finite() {
        return Err(format!("{}: the proton charges overflow", path.display()));
    }
    Ok((charged.len() as f64, total))
}

/// One raw run: its counts, edges and live fractions, and its total proton charge.
struct RawRun {
    counts: ndarray::Array3<f64>,
    edges: Vec<f64>,
    live: ndarray::Array3<f64>,
    charge: Option<f64>,
}

/// One run, its triggers limited to the beam pulses of its NeXus file, and its total charge.
fn tpx1_run(
    folder: &std::path::Path,
    nexus: Option<&std::path::Path>,
) -> Result<(nereids_io::fits_stack::FitsRun, Option<f64>), String> {
    let mut run = nereids_io::fits_stack::load_fits_run(folder, None).map_err(|e| e.to_string())?;
    let charge = match nexus {
        Some(path) => {
            let (pulses, charge) = beam_pulses(path)?;
            run.limit_to_beam_pulses(pulses)
                .map_err(|e| e.to_string())?;
            Some(charge)
        }
        None => None,
    };
    Ok((run, charge))
}

/// Live fractions of a run whose excluded pixels' counts do not enter; the counts are kept.
fn live_without(
    run: &mut nereids_io::fits_stack::FitsRun,
    excluded: &ndarray::Array2<bool>,
) -> Result<ndarray::Array3<f64>, String> {
    let pixels: Vec<(usize, usize)> = excluded
        .indexed_iter()
        .filter(|(_, e)| **e)
        .map(|(p, _)| p)
        .collect();
    let kept: Vec<ndarray::Array1<f64>> = pixels
        .iter()
        .map(|&(y, x)| {
            let spectrum = run.counts.slice(ndarray::s![.., y, x]).to_owned();
            run.counts.slice_mut(ndarray::s![.., y, x]).fill(0.0);
            spectrum
        })
        .collect();
    let live = nereids_io::live::live_fractions(run.counts.view(), &run.edges_us, &run.shutters);
    for (&(y, x), spectrum) in pixels.iter().zip(&kept) {
        run.counts.slice_mut(ndarray::s![.., y, x]).assign(spectrum);
    }
    live.map_err(|e| e.to_string())
}

/// Both runs, with live fractions that leave out the pixels the Normalize step excludes.
fn tpx1_pair(
    sample: &std::path::Path,
    open: &std::path::Path,
    nexus: [Option<&std::path::Path>; 2],
) -> Result<[RawRun; 2], String> {
    let (mut s, sample_charge) = tpx1_run(sample, nexus[0])?;
    let (mut o, open_charge) = tpx1_run(open, nexus[1])?;
    let width = s
        .edges_us
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::INFINITY, f64::min);
    let same = s.edges_us.len() == o.edges_us.len()
        && s.edges_us
            .iter()
            .zip(&o.edges_us)
            .all(|(a, b)| (a - b).abs() <= 1e-3 * width);
    if !same || s.counts.dim() != o.counts.dim() {
        return Err("the sample and open-beam runs have different bins or images".into());
    }
    let excluded = nereids_io::pixels::detect_bad_pixels(
        &s.counts,
        Some(&o.counts),
        Some(nereids_io::pixels::HOT_PIXEL_K_MAD),
    )
    .map_err(|e| e.to_string())?;
    let (sample_live, open_live) = (
        live_without(&mut s, &excluded)?,
        live_without(&mut o, &excluded)?,
    );
    Ok([
        RawRun {
            counts: s.counts,
            edges: s.edges_us,
            live: sample_live,
            charge: sample_charge,
        },
        RawRun {
            counts: o.counts,
            edges: o.edges_us,
            live: open_live,
            charge: open_charge,
        },
    ])
}

/// Load both raw Timepix-1 runs and compute their live fractions.
fn load_tpx1_runs(state: &mut AppState) {
    state.invalidate_results();
    let (Some(sample_path), Some(open_path)) =
        (state.sample_path.clone(), state.open_beam_path.clone())
    else {
        return;
    };
    state.sample_data = None;
    state.open_beam_data = None;
    state.tpx1_live = None;
    state.pixel_fit = None;
    state.map_fit = None;
    let loaded = tpx1_pair(
        &sample_path,
        &open_path,
        [state.hdf5_path.as_deref(), state.hdf5_ob_path.as_deref()],
    );
    match loaded {
        Ok(
            [
                RawRun {
                    counts: sample,
                    edges,
                    live: sample_live,
                    charge: sample_charge,
                },
                RawRun {
                    counts: open,
                    live: open_live,
                    charge: open_charge,
                    ..
                },
            ],
        ) => {
            state.preview_image = Some(sample.sum_axis(Axis(0)));
            state.log_provenance(
                ProvenanceEventKind::DataLoaded,
                format!(
                    "Loaded raw Timepix-1 runs: {} bins from {} and {}; beam pulses from {}",
                    sample.shape()[0],
                    sample_path.display(),
                    open_path.display(),
                    if state.hdf5_path.is_some() && state.hdf5_ob_path.is_some() {
                        "both NeXus files"
                    } else {
                        "the triggers where no NeXus file is given"
                    },
                ),
            );
            state.status_message = match (sample_charge, open_charge) {
                (Some(s), Some(o)) if o > 0.0 => {
                    state.proton_charge_sample = s;
                    state.proton_charge_ob = o;
                    state.kl_c_ratio = s / o;
                    format!(
                        "Raw runs and live fractions loaded; charge ratio {:.4}",
                        s / o
                    )
                }
                // The fitted normalization absorbs the ratio, so the one in Analyze is kept.
                _ => {
                    state.proton_charge_sample = 0.0;
                    state.proton_charge_ob = 0.0;
                    "Raw runs and live fractions loaded; the charge ratio is the one in Analyze"
                        .into()
                }
            };
            state.spectrum_kind = nereids_io::spectrum::SpectrumValueKind::BinEdges;
            state.spectrum_unit = nereids_io::spectrum::SpectrumUnit::TofMicroseconds;
            state.spectrum_values = Some(Arc::new(edges));
            let (sample, open) = (Arc::new(sample), Arc::new(open));
            state.tpx1_live = Some(crate::state::Tpx1Live {
                sample_counts: Arc::clone(&sample),
                open_counts: Arc::clone(&open),
                sample: Arc::new(sample_live),
                open: Arc::new(open_live),
            });
            state.sample_data = Some(sample);
            state.open_beam_data = Some(open);
            state.load_error = false;
        }
        Err(e) => {
            state.status_message = format!("Failed to load the raw runs: {e}");
            state.load_error = true;
        }
    }
}

/// Sample-run NeXus file picked (raw Timepix-1 tab).
pub(crate) fn on_tpx1_nexus_picked(state: &mut AppState, path: std::path::PathBuf, open: bool) {
    if state.input_mode != InputMode::Tpx1Raw {
        return;
    }
    if open {
        state.hdf5_ob_path = Some(path);
    } else {
        state.hdf5_path = Some(path);
    }
    state.invalidate_results();
    state.pixel_fit = None;
    state.map_fit = None;
    state.sample_data = None;
    state.open_beam_data = None;
    state.tpx1_live = None;
    state.energies = None;
    state.load_error = false;
}

// ── TIFF Pair tab ──────────────────────────────────────────────

/// TIFF Pair tab: Sample + Open Beam drop zones + Spectrum.
fn tiff_pair_tab(ui: &mut egui::Ui, state: &mut AppState) {
    design::card(ui, |ui| {
        ui.label(
            egui::RichText::new("Load raw sample + open beam TIFF stacks with TOF spectrum.")
                .size(10.0)
                .color(ThemeColors::from_ctx(ui.ctx()).fg3),
        );
        ui.add_space(8.0);

        tiff_drop_zone(
            ui,
            &mut state.file_dialogs,
            DialogIntent::TiffSample,
            "Sample",
            &state.sample_path,
        );

        ui.add_space(6.0);
        tiff_drop_zone(
            ui,
            &mut state.file_dialogs,
            DialogIntent::TiffOpenBeam,
            "Open Beam",
            &state.open_beam_path,
        );

        ui.add_space(8.0);
        spectrum_section(ui, state);
    });

    // Auto-load when all files are selected
    let can_load = state.sample_path.is_some()
        && state.open_beam_path.is_some()
        && state.spectrum_path.is_some()
        && state.sample_data.is_none()
        && !state.load_error;
    if can_load {
        load_all_data(state);
    }

    load_status_ui(ui, state);
    show_loaded_info(ui, state);
}

// ── Drop zone + folder fallback ────────────────────────────────

/// TIFF drop zone: click-to-browse file, with "or browse folder" link below.
///
/// Opens a dialog tagged with `intent`; the pick (file or folder — both
/// land in the same path field) is applied by the dispatch handler.
fn tiff_drop_zone(
    ui: &mut egui::Ui,
    dialogs: &mut crate::file_dialog::FileDialogs,
    intent: DialogIntent,
    label: &str,
    path: &Option<std::path::PathBuf>,
) {
    let loaded = path.is_some();
    let display = path
        .as_ref()
        .map_or(format!("Click to select {label}..."), |p| {
            p.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string()
        });
    let hint = "TIFF file or folder of TIFFs";

    let resp = design::drop_zone(ui, loaded, &display, hint);
    if resp.clicked() {
        dialogs.pick_file(
            intent.clone(),
            DialogOptions {
                filters: vec![("TIFF", &["tif", "tiff"])],
                ..Default::default()
            },
        );
    }
    // Secondary: folder browse
    if ui.small_button("or browse folder\u{2026}").clicked() {
        dialogs.pick_folder(intent, Default::default());
    }
}

// ── Spectrum section ───────────────────────────────────────────

/// Simplified spectrum file section: drop zone, no unit/kind toggles.
fn spectrum_section(ui: &mut egui::Ui, state: &mut AppState) {
    ui.label(egui::RichText::new("Spectrum File").strong());

    let loaded = state.spectrum_path.is_some();
    let display =
        state
            .spectrum_path
            .as_ref()
            .map_or("Click to select spectrum file...".to_string(), |p| {
                p.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            });
    let resp = design::drop_zone(
        ui,
        loaded,
        &display,
        "CSV/TXT/DAT with TOF bin edges or centers",
    );
    if resp.clicked() {
        state.file_dialogs.pick_file(
            DialogIntent::SpectrumFile,
            DialogOptions {
                filters: vec![("Spectrum", &["csv", "txt", "dat"])],
                ..Default::default()
            },
        );
    }

    // Show parsed info (no unit/kind toggles — auto-detected in load_all_data)
    if let Some(ref vals) = state.spectrum_values {
        ui.label(
            egui::RichText::new(format!("Parsed: {} values", vals.len()))
                .size(10.0)
                .color(ThemeColors::from_ctx(ui.ctx()).fg3),
        );
    }
}

// ── Loaded info ────────────────────────────────────────────────

/// Display loaded data info with a Reload button.
fn show_loaded_info(ui: &mut egui::Ui, state: &mut AppState) {
    let tc = ThemeColors::from_ctx(ui.ctx());
    if state.sample_data.is_none() && state.open_beam_data.is_none() {
        return;
    }
    ui.add_space(8.0);
    if let Some(ref data) = state.sample_data {
        let shape = data.shape();
        ui.label(
            egui::RichText::new(format!(
                "\u{2713} Sample: {} frames, {}×{} px",
                shape[0], shape[1], shape[2]
            ))
            .size(11.0)
            .color(tc.fg2),
        );
    }
    if let Some(ref data) = state.open_beam_data {
        let shape = data.shape();
        ui.label(
            egui::RichText::new(format!(
                "\u{2713} Open Beam: {} frames, {}×{} px",
                shape[0], shape[1], shape[2]
            ))
            .size(11.0)
            .color(tc.fg2),
        );
    }
    // Reload button: force re-read from disk
    if ui.small_button("\u{21bb} Reload").clicked() {
        state.sample_data = None;
        state.open_beam_data = None;
        state.tpx1_live = None;
        state.load_error = false;
        state.nexus_probe_error = None;
        state.invalidate_results();
    }
}

// ── HDF5 Histogram tab ────────────────────────────────────────

/// HDF5 Histogram tab: load pre-histogrammed NeXus data.
fn hdf5_histogram_tab(ui: &mut egui::Ui, state: &mut AppState) {
    design::card(ui, |ui| {
        ui.label(
            egui::RichText::new("Load pre-histogrammed NeXus/HDF5 data.")
                .size(10.0)
                .color(ThemeColors::from_ctx(ui.ctx()).fg3),
        );
        ui.add_space(8.0);
        hdf5_drop_zone(ui, state);
        show_nexus_metadata(ui, state);
    });

    // Open beam file (optional — enables counts-domain KL fitting)
    hdf5_ob_picker(ui, state);

    show_hdf5_tree(ui, state);

    // Auto-load histogram when file is selected and has histogram data
    let can_load = state.hdf5_path.is_some()
        && state
            .nexus_metadata
            .as_ref()
            .is_some_and(|m| m.has_histogram)
        && state.sample_data.is_none()
        && !state.load_error;
    if can_load {
        load_hdf5_histogram(state);
    }

    load_status_ui(ui, state);
    show_loaded_info(ui, state);
}

// ── HDF5 Event tab ─────────────────────────────────────────────

/// HDF5 Event tab: load raw neutron events and histogram them.
fn hdf5_event_tab(ui: &mut egui::Ui, state: &mut AppState) {
    design::card(ui, |ui| {
        ui.label(
            egui::RichText::new("Load raw neutron events from NeXus/HDF5 and histogram them.")
                .size(10.0)
                .color(ThemeColors::from_ctx(ui.ctx()).fg3),
        );
        ui.add_space(8.0);
        hdf5_drop_zone(ui, state);
        show_nexus_metadata(ui, state);
    });

    // Open beam file (optional — enables counts-domain KL fitting)
    hdf5_ob_picker(ui, state);

    show_hdf5_tree(ui, state);

    show_loaded_info(ui, state);
}

/// Optional open beam NeXus file picker for HDF5 modes.
/// When provided, enables proper normalization (T = sample/OB) and
/// counts-domain KL fitting.
fn hdf5_ob_picker(ui: &mut egui::Ui, state: &mut AppState) {
    design::card(ui, |ui| {
        ui.label(
            egui::RichText::new("Open beam (optional — enables counts-domain fitting)")
                .size(10.0)
                .color(ThemeColors::from_ctx(ui.ctx()).fg3),
        );
        ui.add_space(4.0);
        let loaded = state.open_beam_data.is_some();
        let display = state
            .hdf5_ob_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "Click to select open beam...".to_string());
        let resp = design::drop_zone(ui, loaded, &display, "Open beam HDF5 (.h5, .hdf5, .nxs)");
        if resp.clicked() {
            // Capture the interpretation inputs (mode + binning) at
            // click time — see the `DialogIntent::Hdf5OpenBeam` doc
            // comment. Params are captured unconditionally: cheap, and
            // correct regardless of which mode is active at resolution.
            state.file_dialogs.pick_file(
                DialogIntent::Hdf5OpenBeam {
                    mode: state.input_mode,
                    event_params: nereids_io::nexus::EventBinningParams {
                        n_bins: state.event_n_bins,
                        tof_min_us: state.event_tof_min_us,
                        tof_max_us: state.event_tof_max_us,
                        height: state.event_height,
                        width: state.event_width,
                    },
                },
                DialogOptions {
                    filters: vec![("HDF5", &["h5", "hdf5", "nxs", "nx5"])],
                    ..Default::default()
                },
            );
        }
        if loaded && ui.small_button("Clear OB").clicked() {
            state.hdf5_ob_path = None;
            state.open_beam_data = None;
            // Same reasoning as the load path above — OB removal
            // invalidates downstream fit/normalization state only.
            state.invalidate_fit_results();
            state.status_message = "Open beam cleared".into();
        }
    });
}

// ── HDF5 shared helpers ────────────────────────────────────────

/// HDF5 file drop zone with auto-probe on selection.
fn hdf5_drop_zone(ui: &mut egui::Ui, state: &mut AppState) {
    let loaded = state.hdf5_path.is_some();
    let display =
        state
            .hdf5_path
            .as_ref()
            .map_or("Click to select NeXus/HDF5 file...".to_string(), |p| {
                p.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            });
    let resp = design::drop_zone(ui, loaded, &display, "HDF5/NeXus (.h5, .hdf5, .nxs, .nx5)");
    if resp.clicked() {
        state.file_dialogs.pick_file(
            DialogIntent::Hdf5Sample,
            DialogOptions {
                filters: vec![("NeXus/HDF5", &["h5", "hdf5", "nxs", "nx5"])],
                ..Default::default()
            },
        );
    }
}

/// Display probed NeXus metadata (or inline probe error).
fn show_nexus_metadata(ui: &mut egui::Ui, state: &AppState) {
    // Show probe error inline in red
    if let Some(ref err) = state.nexus_probe_error {
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(err)
                .size(11.0)
                .color(crate::theme::semantic::RED),
        );
        return;
    }

    if let Some(ref meta) = state.nexus_metadata {
        let tc = ThemeColors::from_ctx(ui.ctx());
        ui.add_space(4.0);
        if meta.has_histogram {
            if let Some(shape) = meta.histogram_shape {
                ui.label(
                    egui::RichText::new(format!(
                        "Histogram: {}×{}×{} (rot×y×x), {} TOF bins",
                        shape[0], shape[1], shape[2], shape[3]
                    ))
                    .size(11.0)
                    .color(tc.fg2),
                );
            }
        } else {
            ui.label(
                egui::RichText::new("No histogram data.")
                    .size(11.0)
                    .color(tc.fg3),
            );
        }

        if meta.has_events {
            if let Some(n) = meta.n_events {
                ui.label(
                    egui::RichText::new(format!("Events: {} neutrons", n))
                        .size(11.0)
                        .color(tc.fg2),
                );
            }
        } else {
            ui.label(
                egui::RichText::new("No event data.")
                    .size(11.0)
                    .color(tc.fg3),
            );
        }

        if let Some(fp) = meta.flight_path_m {
            ui.label(
                egui::RichText::new(format!("Flight path: {:.2} m", fp))
                    .size(11.0)
                    .color(tc.fg2),
            );
        }
    }
}

/// Display the HDF5 file tree structure in a collapsing header.
fn show_hdf5_tree(ui: &mut egui::Ui, state: &AppState) {
    let tree = match state.hdf5_tree {
        Some(ref t) if !t.is_empty() => t,
        _ => return,
    };

    ui.add_space(4.0);
    egui::CollapsingHeader::new("HDF5 Structure")
        .default_open(false)
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .max_height(200.0)
                .show(ui, |ui| {
                    for entry in tree {
                        let depth = entry.path.matches('/').count().saturating_sub(1);
                        let indent = "  ".repeat(depth);
                        let name = entry.path.rsplit('/').next().unwrap_or(&entry.path);
                        let label = match entry.kind {
                            nereids_io::nexus::Hdf5EntryKind::Group => {
                                format!("{indent}[G] {name}")
                            }
                            nereids_io::nexus::Hdf5EntryKind::Dataset => {
                                if let Some(ref shape) = entry.shape {
                                    format!("{indent}[D] {name} {:?}", shape)
                                } else {
                                    format!("{indent}[D] {name}")
                                }
                            }
                        };
                        ui.label(egui::RichText::new(label).monospace().small());
                    }
                });
        });
}

// ── Data loading logic ─────────────────────────────────────────

/// Show load error with retry button.
fn load_status_ui(ui: &mut egui::Ui, state: &mut AppState) {
    if state.load_error {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(&state.status_message).color(crate::theme::semantic::RED));
            if ui.button("Retry").clicked() {
                state.load_error = false;
                state.sample_data = None;
                state.open_beam_data = None;
                state.tpx1_live = None;
                state.spectrum_values = None;
            }
        });
    }
}

/// Load histogram data from HDF5 file.
fn load_hdf5_histogram(state: &mut AppState) {
    let path = match state.hdf5_path {
        Some(ref p) => p.clone(),
        None => return,
    };

    state.invalidate_results();

    // Explicit opt-in to the legacy sum-over-angles
    // behaviour on the sample-data load.  The status banner below
    // tells the user how many angles were combined.
    match nereids_io::nexus::load_nexus_histogram_with_mode(
        &path,
        nereids_io::nexus::MultiAngleMode::Sum,
    ) {
        Ok(data) => {
            let shape = data.counts.shape();
            state.preview_image = Some(data.counts.sum_axis(ndarray::Axis(0)));
            // Report the rotation angle count when angles were collapsed.
            let angle_note = if data.n_rotation_angles > 1 {
                format!(
                    " ({} rotation angles summed — multi-angle analysis not yet supported)",
                    data.n_rotation_angles
                )
            } else {
                String::new()
            };
            state.status_message = format!(
                "HDF5 histogram loaded: {} frames, {}×{} px{angle_note}",
                shape[0], shape[1], shape[2]
            );

            state.spectrum_values = Some(Arc::new(data.tof_edges_us.clone()));
            state.spectrum_unit = nereids_io::spectrum::SpectrumUnit::TofMicroseconds;

            let n_frames = data.counts.shape()[0];
            let n_tof_vals = data.tof_edges_us.len();
            state.spectrum_kind = if n_tof_vals == n_frames + 1 {
                nereids_io::spectrum::SpectrumValueKind::BinEdges
            } else {
                nereids_io::spectrum::SpectrumValueKind::BinCenters
            };

            if let Some(fp) = data.flight_path_m
                && fp.is_finite()
                && fp > 0.0
            {
                state.beamline.flight_path_m = fp;
            }

            if let Some(offset_ns) = state.nexus_metadata.as_ref().and_then(|m| m.tof_offset_ns) {
                let delay_us = offset_ns / 1000.0;
                if delay_us.is_finite() {
                    state.beamline.delay_us = delay_us;
                }
            }

            // Declared-mask provenance: the file-declared mask gets
            // its own field (assigned unconditionally — a file without a
            // mask must not inherit a previous file's), and the effective
            // mask starts as exactly that.  Every normalization recomputes
            // dead_pixels = declared ∪ detected from scratch
            // (AppState::set_detected_dead_pixels).  No detection has run
            // on the fresh data yet.
            state.file_dead_pixels = data.dead_pixels.clone();
            state.dead_pixels = data.dead_pixels;
            state.detected_dead_pixels = None;

            let shape = data.counts.shape();
            state.log_provenance(
                ProvenanceEventKind::DataLoaded,
                format!(
                    "Loaded HDF5 histogram: {} frames ({}x{})",
                    shape[0], shape[1], shape[2]
                ),
            );
            state.sample_data = Some(Arc::new(data.counts));
        }
        Err(e) => {
            state.status_message = format!("HDF5 load failed: {e}");
            state.load_error = true;
        }
    }
}

/// Format the chunk-summing / chunk-concatenation / stray-file /
/// inconsistent-chunk / pixel-clipping provenance suffix for a TIFF load, or
/// an empty string when nothing noteworthy happened.
fn tiff_load_suffix(info: &nereids_io::tiff_stack::TiffLoadInfo) -> String {
    let mut suffix = String::new();
    if info.n_chunks > 1 {
        let ids = info
            .chunk_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<String>>()
            .join(", ");
        if info.chunks_summed {
            suffix.push_str(&format!(", summed {} DAQ chunks ({ids})", info.n_chunks));
        } else {
            // Transmission mode loads with sum_chunks=false; without this
            // line a mis-picked raw chunked run folder loads as a k×
            // concatenated stack with zero provenance — the only downstream
            // symptom is a spectrum-count mismatch that never mentions
            // chunks.
            suffix.push_str(&format!(
                ", detected {} DAQ chunks ({ids}) — NOT summed (transmission \
                 mode); frames are concatenated",
                info.n_chunks,
            ));
        }
    }
    if info.n_unrecognized_files > 0 {
        suffix.push_str(&format!(
            ", {} file(s) did not match the chunk naming pattern (e.g. {}) — \
             chunk detection disabled, frames concatenated lexicographically",
            info.n_unrecognized_files,
            info.unrecognized_examples.join(", "),
        ));
    }
    if info.chunk_inconsistent {
        // The files ARE chunk-patterned but internally inconsistent (ragged
        // frame counts/sets or a duplicate (chunk, frame) pair).  Under the
        // default summing path this is a hard error; because transmission
        // mode loads with sum_chunks=false, it instead loaded verbatim as a
        // lexicographic concatenation — which the operator must see, since a
        // ragged raw run folder that would otherwise fail is now a silent
        // plain stack whose only symptom is a spectrum-count mismatch.
        suffix.push_str(
            " — WARNING: DAQ chunks are internally inconsistent (ragged frame \
             counts/sets or duplicate frames); loaded as a plain lexicographic \
             concatenation, not summed",
        );
    }
    if info.n_clipped_pixels > 0 {
        suffix.push_str(&format!(
            ", {} negative pixels clipped to 0",
            info.n_clipped_pixels
        ));
    }
    suffix
}

/// Load all data: TIFF stacks + spectrum file with validation and auto-detect.
fn load_all_data(state: &mut AppState) {
    state.invalidate_results();

    // Raw counts: reject bad pixels, sum DAQ chunks (the defaults).
    let options = nereids_io::tiff_stack::TiffFolderOptions::default();

    // Load sample TIFF (auto-detect file vs directory)
    if let Some(ref path) = state.sample_path {
        match nereids_io::tiff_stack::load_tiff_auto_with_options(path, &options) {
            Ok((data, info)) => {
                state.preview_image = Some(data.sum_axis(Axis(0)));
                let n_frames = data.shape()[0];
                state.log_provenance(
                    ProvenanceEventKind::DataLoaded,
                    format!(
                        "Loaded sample TIFF: {n_frames} frames from {}{}",
                        path.display(),
                        tiff_load_suffix(&info),
                    ),
                );
                state.sample_data = Some(Arc::new(data));
                state.status_message = "Sample loaded".into();
            }
            Err(e) => {
                state.status_message = format!("Failed to load sample: {}", e);
                state.load_error = true;
                return;
            }
        }
    }

    // Load open beam (TiffPair mode only)
    if state.input_mode == InputMode::TiffPair
        && let Some(ref path) = state.open_beam_path
    {
        match nereids_io::tiff_stack::load_tiff_auto_with_options(path, &options) {
            Ok((data, info)) => {
                let suffix = tiff_load_suffix(&info);
                if !suffix.is_empty() {
                    state.log_provenance(
                        ProvenanceEventKind::DataLoaded,
                        format!(
                            "Loaded open-beam TIFF: {} frames from {}{}",
                            data.shape()[0],
                            path.display(),
                            suffix,
                        ),
                    );
                }
                state.open_beam_data = Some(Arc::new(data));
                state.status_message = "Sample and open beam loaded".into();
            }
            Err(e) => {
                state.status_message = format!("Failed to load open beam: {}", e);
                state.sample_data = None; // Clear partial data
                state.load_error = true;
                return;
            }
        }
    }

    // Validate frame count consistency between sample and open beam
    if let (Some(sample), Some(ob)) = (&state.sample_data, &state.open_beam_data)
        && sample.shape()[0] != ob.shape()[0]
    {
        state.status_message = format!(
            "Frame count mismatch: sample has {} frames, open beam has {}",
            sample.shape()[0],
            ob.shape()[0]
        );
        state.sample_data = None;
        state.open_beam_data = None;
        state.load_error = true;
        return;
    }

    // Parse spectrum file with auto-detect bin type
    if let Some(ref path) = state.spectrum_path {
        // VENUS sidecar auto-detect: *_Spectra.txt files hold frame START
        // times in SECONDS, not verbatim-µs edges — feeding them to the
        // plain parser would be a 10^6 unit error, so the sidecar reader
        // handles them and a sidecar parse failure is surfaced, never
        // silently retried with the verbatim parser.  Detection is by
        // filename suffix only; an explicit format selector in the UI is a
        // possible follow-up, so the provenance log records which parser
        // was chosen and why.
        let is_sidecar = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.to_lowercase().ends_with("_spectra.txt"));
        if is_sidecar {
            // Cross-check the sidecar frame count against the sample stack only
            // when it is already loaded; with no stack yet, pass `None` so a
            // valid sidecar parses instead of failing the count check against a
            // spurious n_frames = 0.
            let n_frames = state.sample_data.as_ref().map(|d| d.shape()[0]);
            match nereids_io::spectrum::read_tof_sidecar(path, n_frames) {
                Ok(edges) => {
                    state.spectrum_kind = nereids_io::spectrum::SpectrumValueKind::BinEdges;
                    state.spectrum_unit = nereids_io::spectrum::SpectrumUnit::TofMicroseconds;
                    state.log_provenance(
                        ProvenanceEventKind::DataLoaded,
                        format!(
                            "Spectrum parser: TOF sidecar (filename ends in \
                             _Spectra.txt): {} frame start times in seconds \
                             converted to {} µs bin edges",
                            edges.len() - 1,
                            edges.len(),
                        ),
                    );
                    state.spectrum_values = Some(Arc::new(edges));
                    state.status_message = "All data loaded".into();
                }
                Err(e) => {
                    state.status_message = format!("Failed to parse TOF sidecar: {}", e);
                    state.load_error = true;
                }
            }
            return;
        }
        match nereids_io::spectrum::parse_spectrum_file(path) {
            Ok(values) => {
                // Validate monotonicity
                if let Err(e) = nereids_io::spectrum::validate_monotonic(&values) {
                    state.status_message = format!("Spectrum: {}", e);
                    state.load_error = true;
                    return;
                }

                // Auto-detect bin type from frame count
                let n_frames = state.sample_data.as_ref().map_or(0, |d| d.shape()[0]);
                let n_values = values.len();
                state.spectrum_kind = if n_values == n_frames + 1 {
                    nereids_io::spectrum::SpectrumValueKind::BinEdges
                } else if n_values == n_frames {
                    nereids_io::spectrum::SpectrumValueKind::BinCenters
                } else {
                    state.status_message = format!(
                        "Spectrum mismatch: {n_values} values vs {n_frames} frames \
                         (expected {f1} for BinEdges or {n_frames} for BinCenters)",
                        f1 = n_frames + 1,
                    );
                    state.load_error = true;
                    return;
                };

                // Default to TOF microseconds
                state.spectrum_unit = nereids_io::spectrum::SpectrumUnit::TofMicroseconds;

                // Validate with auto-detected kind
                if let Err(e) = nereids_io::spectrum::validate_spectrum_frame_count(
                    values.len(),
                    n_frames,
                    state.spectrum_kind,
                ) {
                    state.status_message = format!("Spectrum: {}", e);
                    state.load_error = true;
                    return;
                }

                state.log_provenance(
                    ProvenanceEventKind::DataLoaded,
                    format!(
                        "Spectrum parser: verbatim spectrum file (no \
                         _Spectra.txt suffix): {} values used as-is (µs/eV), \
                         detected as {:?}",
                        values.len(),
                        state.spectrum_kind,
                    ),
                );
                state.spectrum_values = Some(Arc::new(values));
                state.status_message = "All data loaded".into();
            }
            Err(e) => {
                state.status_message = format!("Failed to parse spectrum: {}", e);
                state.load_error = true;
            }
        }
    }
}

// ── Dialog-dispatch handlers ───────────────────────────────────
// Invoked from `crate::file_dialog::dispatch_results` when a Load-step
// picker resolves. Each mirrors the invalidation set the corresponding
// inline block used before the picks were routed through the facade.

/// Sample TIFF stack picked.
pub(crate) fn on_tiff_sample_picked(state: &mut AppState, path: std::path::PathBuf) {
    if state.input_mode == InputMode::Tpx1Raw {
        state.hdf5_path = None;
    }
    state.cancel_pending_tasks();
    state.pixel_fit = None;
    state.map_fit = None;
    state.sample_path = Some(path);
    state.sample_data = None;
    state.tpx1_live = None;
    state.dead_pixels = None;
    state.file_dead_pixels = None;
    state.detected_dead_pixels = None;
    state.energies = None;
    state.load_error = false;
}

/// Open-beam TIFF stack picked.
pub(crate) fn on_tiff_open_beam_picked(state: &mut AppState, path: std::path::PathBuf) {
    if state.input_mode == InputMode::Tpx1Raw {
        state.hdf5_ob_path = None;
    }
    state.cancel_pending_tasks();
    state.pixel_fit = None;
    state.map_fit = None;
    state.open_beam_path = Some(path);
    state.open_beam_data = None;
    state.tpx1_live = None;
    state.dead_pixels = None;
    state.file_dead_pixels = None;
    state.detected_dead_pixels = None;
    state.energies = None;
    state.load_error = false;
}

/// TOF spectrum file picked.
pub(crate) fn on_spectrum_picked(state: &mut AppState, path: std::path::PathBuf) {
    state.spectrum_path = Some(path);
    state.spectrum_values = None;
    state.sample_data = None;
    state.open_beam_data = None;
    state.energies = None;
    state.load_error = false;
}

/// Sample NeXus/HDF5 file picked: reset derived state, probe the file,
/// and build the tree view.
pub(crate) fn on_hdf5_sample_picked(state: &mut AppState, file: std::path::PathBuf) {
    state.hdf5_path = Some(file.clone());
    state.invalidate_results();
    state.sample_data = None;
    state.open_beam_data = None;
    state.hdf5_ob_path = None;
    state.load_error = false;
    state.nexus_probe_error = None;

    // Probe the file immediately
    match nereids_io::nexus::probe_nexus(&file) {
        Ok(meta) => {
            if let Some(shape) = meta.histogram_shape {
                state.event_height = shape[1];
                state.event_width = shape[2];
            }
            state.nexus_metadata = Some(meta);
            state.status_message = "NeXus file probed".into();
        }
        Err(e) => {
            state.nexus_metadata = None;
            state.nexus_probe_error = Some(format!("Probe failed: {e}"));
            state.status_message = format!("Probe failed: {e}");
        }
    }

    // Build HDF5 tree structure for browser display
    match nereids_io::nexus::list_hdf5_tree(&file, 3) {
        Ok(tree) => state.hdf5_tree = Some(tree),
        Err(_) => state.hdf5_tree = None,
    }
}

/// Optional open-beam NeXus/HDF5 file picked: load (event or histogram
/// per the mode captured at request time), validate against the
/// current sample, and install.
pub(crate) fn on_hdf5_ob_picked(
    state: &mut AppState,
    path: std::path::PathBuf,
    mode: InputMode,
    event_params: &nereids_io::nexus::EventBinningParams,
) {
    state.hdf5_ob_path = Some(path.clone());
    // Event-vs-histogram dispatch and binning use ONLY the values
    // captured when the dialog was requested (`mode`, `event_params`) —
    // state edits made while the dialog is pending must not change how
    // the picked file is loaded. The shape/TOF validation below reads
    // CURRENT state on purpose: the OB must match whatever sample is
    // loaded now.
    let ob_result = if mode == InputMode::Hdf5Event {
        nereids_io::nexus::load_nexus_events(&path, event_params)
            .map(|d| (d.counts, d.tof_edges_us, d.n_rotation_angles))
    } else {
        // The loader refuses multi-angle files by
        // default to prevent silent sum-over-angles.  The GUI
        // makes the explicit opt-in to preserve existing
        // single-volume analysis behaviour on OB input.
        //
        // Surface the rotation-angle count to the
        // user via the status banner, mirroring the sample-load
        // path.  `data.n_rotation_angles` is carried through the
        // tuple so the success arm can build the same `angle_note`
        // suffix.
        nereids_io::nexus::load_nexus_histogram_with_mode(
            &path,
            nereids_io::nexus::MultiAngleMode::Sum,
        )
        .map(|d| (d.counts, d.tof_edges_us, d.n_rotation_angles))
    };
    match ob_result {
        Ok((ob_counts, ob_tof_edges, n_rotation_angles)) => {
            // Validate that the shape matches the sample
            if let Some(ref sample) = state.sample_data
                && sample.shape() != ob_counts.shape()
            {
                state.status_message = format!(
                    "OB shape {:?} != sample {:?}",
                    ob_counts.shape(),
                    sample.shape()
                );
                state.hdf5_ob_path = None;
                state.open_beam_data = None;
                return;
            }
            // Validate TOF grid matches sample — reject if edges differ.
            // Positional bin pairing in the counts path is only correct
            // when sample and OB share the exact same TOF grid.
            if let Some(ref sv) = state.spectrum_values {
                if ob_tof_edges.len() != sv.len() {
                    state.status_message = format!(
                        "OB rejected: {} TOF edges vs sample {} — grids must match",
                        ob_tof_edges.len(),
                        sv.len()
                    );
                    state.hdf5_ob_path = None;
                    state.open_beam_data = None;
                    return;
                }
                let max_edge_diff: f64 = ob_tof_edges
                    .iter()
                    .zip(sv.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, f64::max);
                let edge_scale = sv.last().copied().unwrap_or(1.0).max(1.0);
                if max_edge_diff > 1e-6 * edge_scale {
                    state.status_message = format!(
                        "OB rejected: TOF edges differ (max delta = {:.3} µs) — \
                         sample and OB must use the same TOF grid",
                        max_edge_diff
                    );
                    state.hdf5_ob_path = None;
                    state.open_beam_data = None;
                    return;
                }
            }
            state.open_beam_data = Some(Arc::new(ob_counts));
            // OB swap invalidates normalization + fit results,
            // but NOT the sample's TOF spectrum, energies,
            // preview image, ROIs, or rebin state.  The broad
            // `invalidate_results()` would clear `spectrum_values`,
            // which gates the Continue button on Hdf5Histogram
            // mode (`has_required_data`) — using the narrow
            // variant keeps Continue enabled.
            state.invalidate_fit_results();
            // Mirror the sample-load rotation-angle note
            // so OB and sample banners use identical wording
            // when angles were collapsed.
            let angle_note = if n_rotation_angles > 1 {
                format!(
                    " ({} rotation angles summed — multi-angle analysis not yet supported)",
                    n_rotation_angles
                )
            } else {
                String::new()
            };
            state.status_message = format!("Open beam loaded{angle_note}");
        }
        Err(e) => {
            state.status_message = format!("Open beam load failed: {e}");
            state.hdf5_ob_path = None;
            // Clear data on failure
            state.open_beam_data = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-shutter raw run of 16-bit FITS frames with its sidecars.
    fn write_run(dir: &std::path::Path, frames: &[[i16; 2]], triggers: u32) {
        for (k, frame) in frames.iter().enumerate() {
            let cards = [
                "SIMPLE  =                    T".to_string(),
                "BITPIX  =                   16".into(),
                "NAXIS   =                    2".into(),
                "NAXIS1  =                    2".into(),
                "NAXIS2  =                    1".into(),
                "END".into(),
            ];
            let mut bytes: Vec<u8> = cards
                .iter()
                .flat_map(|c| format!("{c:<80}").into_bytes())
                .collect();
            bytes.resize(2880, b' ');
            bytes.extend(frame.iter().flat_map(|v| v.to_be_bytes()));
            bytes.resize(5760, 0);
            std::fs::write(dir.join(format!("run_0_114_{k:05}.fits")), bytes).unwrap();
        }
        let starts: String = (0..frames.len())
            .map(|k| format!("{}e-06\t0\r\n", k + 1))
            .collect();
        std::fs::write(dir.join("run_0_114_Spectra.txt"), starts).unwrap();
        let length = frames.len() + 1;
        std::fs::write(
            dir.join("run_0_114_ShutterTimes.txt"),
            format!("0\t1e-06\t{length}e-06\r\n1\t0\t0\r\n"),
        )
        .unwrap();
        std::fs::write(
            dir.join("run_0_114_ShutterCount.txt"),
            format!("0\t{triggers}\r\n1\t0\r\n"),
        )
        .unwrap();
    }

    #[test]
    fn raw_runs_load_with_their_live_fractions_and_refuse_rebinning() {
        let (sample, open) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        write_run(sample.path(), &[[2, 1], [3, 0], [1, 1], [0, 2]], 20);
        write_run(open.path(), &[[4, 2], [6, 0], [2, 2], [0, 4]], 50);
        let mut state = AppState {
            input_mode: InputMode::Tpx1Raw,
            sample_path: Some(sample.path().to_path_buf()),
            open_beam_path: Some(open.path().to_path_buf()),
            ..Default::default()
        };
        load_tpx1_runs(&mut state);
        assert!(tpx1_loaded(&state), "{}", state.status_message);
        let tpx1 = state.tpx1_live.as_ref().unwrap();
        let live = &tpx1.sample;
        assert_eq!(live.shape(), &[4, 1, 2]);
        assert_eq!(live[[2, 0, 0]], 1.0 - 5.0 / 20.0);
        assert_eq!(tpx1.open[[2, 0, 0]], 1.0 - 10.0 / 50.0);

        state.rebin_factor = 2;
        crate::guided::rebin::apply_rebin(&mut state);
        assert_eq!(state.sample_data.as_ref().unwrap().shape()[0], 4);
        state.sample_data = Some(Arc::new((**state.sample_data.as_ref().unwrap()).clone()));
        assert!(!tpx1_loaded(&state));
        load_tpx1_runs(&mut state);

        write_run(open.path(), &[[4, 2], [6, 0], [2, 2]], 50);
        std::fs::remove_file(open.path().join("run_0_114_00003.fits")).unwrap();
        state.sample_data = None;
        load_tpx1_runs(&mut state);
        assert!(state.load_error && state.status_message.contains("different bins"));

        write_run(open.path(), &[[4, 2], [6, 0], [2, 2], [0, 4]], 50);
        load_tpx1_runs(&mut state);
        assert!(
            !state.load_error && tpx1_loaded(&state),
            "{}",
            state.status_message
        );
    }
}
