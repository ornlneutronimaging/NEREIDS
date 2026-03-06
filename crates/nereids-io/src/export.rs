//! Export spatial mapping results to TIFF, HDF5, and Markdown formats.

use std::path::Path;

use ndarray::Array2;

use crate::error::IoError;

/// Escape a string for use in a Markdown table cell.
fn escape_md_cell(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

/// Export a single density map as a 32-bit float TIFF file.
///
/// Each pixel stores the density value (atoms/barn) as f32.  We use
/// 32-bit rather than 64-bit floats because most image viewers
/// (Preview, GIMP, ImageJ, Fiji) cannot open 64-bit float TIFFs.
/// The f64→f32 conversion preserves roughly 7 significant digits of
/// precision, which is sufficient for typical density values.
pub fn export_density_tiff(path: &Path, data: &Array2<f64>, label: &str) -> Result<(), IoError> {
    export_map_tiff(path, data, &format!("{label}_density"))
}

/// Export a single 2D map as a 32-bit float TIFF file.
///
/// The file is named `{name}.tiff` inside `path`.
pub fn export_map_tiff(path: &Path, data: &Array2<f64>, name: &str) -> Result<(), IoError> {
    let filename = path.join(format!("{name}.tiff"));
    let file = std::fs::File::create(&filename)
        .map_err(|e| IoError::TiffEncode(format!("cannot create {}: {e}", filename.display())))?;
    let mut encoder =
        tiff::encoder::TiffEncoder::new(file).map_err(|e| IoError::TiffEncode(e.to_string()))?;

    let (height, width) = (data.shape()[0] as u32, data.shape()[1] as u32);

    // tiff encoder expects a contiguous row-major slice; cast to f32
    // for broad viewer compatibility.
    let pixels: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    encoder
        .write_image::<tiff::encoder::colortype::Gray32Float>(width, height, &pixels)
        .map_err(|e| IoError::TiffEncode(e.to_string()))?;

    Ok(())
}

/// Export all spatial mapping results to a single HDF5 file.
///
/// Layout:
/// - `/density/{label}` — density map for each isotope
/// - `/uncertainty/{label}` — uncertainty map for each isotope
/// - `/chi_squared` — reduced chi-squared map
/// - `/converged` — boolean convergence map (stored as u8: 0/1)
/// - `/temperature` — fitted temperature map in Kelvin (when temperature fitting was enabled)
#[cfg(feature = "hdf5")]
pub fn export_results_hdf5(
    path: &Path,
    density_maps: &[Array2<f64>],
    uncertainty_maps: &[Array2<f64>],
    chi_squared_map: &Array2<f64>,
    converged_map: &Array2<bool>,
    labels: &[String],
    temperature_map: Option<&Array2<f64>>,
) -> Result<(), IoError> {
    let file = hdf5::File::create(path).map_err(|e| IoError::Hdf5Error(format!("create: {e}")))?;

    // Density maps
    let density_group = file
        .create_group("density")
        .map_err(|e| IoError::Hdf5Error(format!("create group /density: {e}")))?;
    for (i, map) in density_maps.iter().enumerate() {
        let name = labels
            .get(i)
            .map_or_else(|| format!("isotope_{i}"), |s| s.clone());
        let shape = [map.shape()[0], map.shape()[1]];
        let data: Vec<f64> = map.iter().copied().collect();
        density_group
            .new_dataset::<f64>()
            .shape(shape)
            .create(name.as_str())
            .and_then(|ds| ds.write_raw(&data))
            .map_err(|e| IoError::Hdf5Error(format!("write /density/{name}: {e}")))?;
    }

    // Uncertainty maps
    let unc_group = file
        .create_group("uncertainty")
        .map_err(|e| IoError::Hdf5Error(format!("create group /uncertainty: {e}")))?;
    for (i, map) in uncertainty_maps.iter().enumerate() {
        let name = labels
            .get(i)
            .map_or_else(|| format!("isotope_{i}"), |s| s.clone());
        let shape = [map.shape()[0], map.shape()[1]];
        let data: Vec<f64> = map.iter().copied().collect();
        unc_group
            .new_dataset::<f64>()
            .shape(shape)
            .create(name.as_str())
            .and_then(|ds| ds.write_raw(&data))
            .map_err(|e| IoError::Hdf5Error(format!("write /uncertainty/{name}: {e}")))?;
    }

    // Chi-squared map
    {
        let shape = [chi_squared_map.shape()[0], chi_squared_map.shape()[1]];
        let data: Vec<f64> = chi_squared_map.iter().copied().collect();
        file.new_dataset::<f64>()
            .shape(shape)
            .create("chi_squared")
            .and_then(|ds| ds.write_raw(&data))
            .map_err(|e| IoError::Hdf5Error(format!("write /chi_squared: {e}")))?;
    }

    // Converged map (stored as u8)
    {
        let shape = [converged_map.shape()[0], converged_map.shape()[1]];
        let data: Vec<u8> = converged_map.iter().map(|&b| u8::from(b)).collect();
        file.new_dataset::<u8>()
            .shape(shape)
            .create("converged")
            .and_then(|ds| ds.write_raw(&data))
            .map_err(|e| IoError::Hdf5Error(format!("write /converged: {e}")))?;
    }

    // Temperature map (when temperature fitting was enabled)
    if let Some(t_map) = temperature_map {
        let shape = [t_map.shape()[0], t_map.shape()[1]];
        let data: Vec<f64> = t_map.iter().copied().collect();
        file.new_dataset::<f64>()
            .shape(shape)
            .create("temperature")
            .and_then(|ds| ds.write_raw(&data))
            .map_err(|e| IoError::Hdf5Error(format!("write /temperature: {e}")))?;
    }

    Ok(())
}

/// Export a Markdown summary report of the spatial mapping results.
pub fn export_markdown_report(
    path: &Path,
    labels: &[String],
    density_maps: &[Array2<f64>],
    converged_map: &Array2<bool>,
    n_converged: usize,
    n_total: usize,
    provenance: &[(String, String)],
) -> Result<(), IoError> {
    let mut report = String::new();
    report.push_str("# NEREIDS Spatial Mapping Report\n\n");

    // Convergence summary
    let pct = if n_total > 0 {
        100.0 * n_converged as f64 / n_total as f64
    } else {
        0.0
    };
    report.push_str(&format!(
        "## Convergence\n\n- Converged: {n_converged} / {n_total} ({pct:.1}%)\n\n"
    ));

    // Per-isotope density statistics
    report.push_str("## Per-Isotope Density Statistics\n\n");
    report.push_str("| Isotope | Mean Density (atoms/barn) | Std Dev |\n");
    report.push_str("|---------|--------------------------|----------|\n");
    for (i, map) in density_maps.iter().enumerate() {
        let label = escape_md_cell(labels.get(i).map_or("unknown", |s| s.as_str()));
        let conv_vals: Vec<f64> = map
            .iter()
            .zip(converged_map.iter())
            .filter(|&(_, &conv)| conv)
            .map(|(&d, _)| d)
            .filter(|d| d.is_finite())
            .collect();
        if conv_vals.is_empty() {
            report.push_str(&format!("| {label} | N/A | N/A |\n"));
        } else {
            let mean: f64 = conv_vals.iter().sum::<f64>() / conv_vals.len() as f64;
            let variance: f64 =
                conv_vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / conv_vals.len() as f64;
            let std_dev = variance.sqrt();
            report.push_str(&format!("| {label} | {mean:.6e} | {std_dev:.6e} |\n"));
        }
    }
    report.push('\n');

    // Provenance log
    if !provenance.is_empty() {
        report.push_str("## Provenance Log\n\n");
        report.push_str("| Time | Event |\n");
        report.push_str("|------|-------|\n");
        for (timestamp, message) in provenance {
            let ts = escape_md_cell(timestamp);
            let msg = escape_md_cell(message);
            report.push_str(&format!("| {ts} | {msg} |\n"));
        }
        report.push('\n');
    }

    std::fs::write(path, report).map_err(|e| {
        IoError::WriteError(format!("cannot write report to {}: {e}", path.display()))
    })?;

    Ok(())
}
