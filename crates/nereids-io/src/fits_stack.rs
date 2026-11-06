//! Raw Timepix-1 frame-mode runs: FITS frames, frame times and shutters.
//!
//! A run folder holds `<prefix>_<chunk>_<frame>.fits`, one image of raw
//! counts per time bin, and per chunk the sidecars `_Spectra.txt` (frame
//! start times in s), `_ShutterTimes.txt` (index, gap before the shutter
//! and its length, in s) and `_ShutterCount.txt` (index, triggers).

use std::path::{Path, PathBuf};

use ndarray::{Array2, Array3};

use crate::error::IoError;
use crate::spectrum::{MICROSECONDS_PER_SECOND, read_tof_sidecar};

const BLOCK: usize = 2880;
const CARD: usize = 80;

/// One shutter of a frame-mode run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shutter {
    /// Opening time after the trigger, in µs.
    pub start_us: f64,
    /// Closing time after the trigger, in µs.
    pub end_us: f64,
    /// Times the shutter opened.
    pub triggers: f64,
}

/// One chunk of a raw frame-mode run.
#[derive(Debug, Clone)]
pub struct FitsRun {
    /// Raw counts in (time bin, row, column), rows in file order.
    pub counts: Array3<f64>,
    /// Time-bin edges in µs, one more than the bins.
    pub edges_us: Vec<f64>,
    /// The run's shutters in time order.
    pub shutters: Vec<Shutter>,
}

/// Read the primary image of a FITS file as counts in (row, column).
///
/// # Errors
/// [`IoError::FileNotFound`] if the file cannot be read;
/// [`IoError::InvalidParameter`] if the primary image is not 2-D with an
/// integer `BITPIX` of 8, 16 or 32, the file is short, or a scaled value is
/// not a whole number 0 or more.
pub fn read_fits_image(path: &Path) -> Result<Array2<f64>, IoError> {
    let name = path.to_string_lossy();
    let bytes =
        std::fs::read(path).map_err(|e| IoError::FileNotFound(name.clone().into_owned(), e))?;
    let bad = |what: String| IoError::InvalidParameter(format!("FITS file {name}: {what}"));

    let mut cards = bytes.as_chunks::<CARD>().0.iter();
    let mut keys: Vec<(String, String)> = Vec::new();
    let mut header = 0;
    loop {
        let card = cards.next().ok_or_else(|| bad("no END card".into()))?;
        header += CARD;
        let text = std::str::from_utf8(card.as_slice())
            .ok()
            .filter(|t| t.is_ascii())
            .ok_or_else(|| bad(format!("header card {} is not ASCII", header / CARD)))?;
        let key = text[..8].trim().to_string();
        if key == "END" {
            break;
        }
        if text.get(8..10) == Some("= ") {
            let value = text[10..]
                .split('/')
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            keys.push((key, value));
        }
    }
    let header = header.div_ceil(BLOCK) * BLOCK;
    let number = |key: &str| -> Result<Option<f64>, IoError> {
        keys.iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| {
                v.parse::<f64>()
                    .map_err(|_| bad(format!("{key} = {v} is not a number")))
            })
            .transpose()
    };
    let required = |key: &str| number(key)?.ok_or_else(|| bad(format!("no {key}")));

    if keys.first().map(|(k, v)| (k.as_str(), v.as_str())) != Some(("SIMPLE", "T")) {
        return Err(bad("first card is not SIMPLE = T".into()));
    }
    if required("NAXIS")? != 2.0 {
        return Err(bad("the primary image is not 2-D".into()));
    }
    let axis = |key: &str| -> Result<usize, IoError> {
        let v = required(key)?;
        if v >= 1.0 && v <= f64::from(u32::MAX) && v.fract() == 0.0 {
            Ok(v as usize)
        } else {
            Err(bad(format!("{key} = {v} is not a positive whole number")))
        }
    };
    let (width, height) = (axis("NAXIS1")?, axis("NAXIS2")?);
    let size = match required("BITPIX")? {
        8.0 => 1,
        16.0 => 2,
        32.0 => 4,
        b => return Err(bad(format!("BITPIX {b} is not raw integer counts"))),
    };
    let zero = number("BZERO")?.unwrap_or(0.0);
    let scale = number("BSCALE")?.unwrap_or(1.0);
    if !(zero.is_finite() && scale.is_finite()) {
        return Err(bad(format!(
            "BZERO {zero} and BSCALE {scale} must be finite"
        )));
    }
    let data = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(size))
        .and_then(|n| bytes.get(header..header.checked_add(n)?))
        .ok_or_else(|| bad(format!("shorter than its {width} × {height} image")))?;

    let values = data.chunks_exact(size).map(|b| {
        let raw = match size {
            1 => f64::from(b[0]),
            2 => f64::from(i16::from_be_bytes([b[0], b[1]])),
            _ => f64::from(i32::from_be_bytes([b[0], b[1], b[2], b[3]])),
        };
        let v = zero + scale * raw;
        // A whole count can carry the scaling's rounding error, a few ulp of its terms.
        let whole = v.round();
        if (v - whole).abs() <= 4.0 * f64::EPSILON * (zero.abs() + (scale * raw).abs()) {
            whole
        } else {
            v
        }
    });
    let image = Array2::from_shape_vec((height, width), values.collect())
        .map_err(|e| bad(e.to_string()))?;
    if let Some(((row, column), v)) = image
        .indexed_iter()
        .find(|(_, v)| !v.is_finite() || **v < 0.0 || v.fract() != 0.0)
    {
        return Err(bad(format!(
            "count {v} at row {row}, column {column} is not a whole number 0 or more"
        )));
    }
    Ok(image)
}

/// The `(prefix, chunk, frame)` of a file named `<prefix>_<chunk>_<frame>.fits`.
fn name_parts(path: &Path) -> Option<(String, String, usize)> {
    let stem = path.file_name()?.to_str()?.strip_suffix(".fits")?;
    let (rest, frame) = stem.rsplit_once('_')?;
    let (prefix, chunk) = rest.rsplit_once('_')?;
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !(digits(chunk) && digits(frame)) {
        return None;
    }
    Some((prefix.to_string(), chunk.to_string(), frame.parse().ok()?))
}

fn sidecar(dir: &Path, prefix: &str, chunk: &str, suffix: &str) -> Result<PathBuf, IoError> {
    let path = dir.join(format!("{prefix}_{chunk}_{suffix}"));
    if path.is_file() {
        Ok(path)
    } else {
        Err(IoError::InvalidParameter(format!(
            "no sidecar {}",
            path.display()
        )))
    }
}

fn rows(path: &Path, columns: usize) -> Result<Vec<Vec<f64>>, IoError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| IoError::FileNotFound(path.to_string_lossy().into_owned(), e))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let row: Option<Vec<f64>> = l
                .split_whitespace()
                .map(|v| v.parse().ok().filter(|v: &f64| v.is_finite()))
                .collect();
            if let Some(row) = row.filter(|r| r.len() == columns) {
                Ok(row)
            } else {
                Err(IoError::InvalidParameter(format!(
                    "{}: '{l}' is not {columns} numbers",
                    path.display()
                )))
            }
        })
        .collect()
}

/// Shutters from `_ShutterTimes.txt` and `_ShutterCount.txt`.
///
/// Rows are indexed 0 to n − 1; rows with no triggers and no length are
/// unused shutters. Shutter `i`
/// opens its gap after shutter `i − 1` closes; the first gap is from the
/// trigger.
fn read_shutters(times: &Path, count: &Path) -> Result<Vec<Shutter>, IoError> {
    let times = rows(times, 3)?;
    let count = rows(count, 2)?;
    let mut shutters = Vec::new();
    let mut closed = 0.0;
    for (i, (t, c)) in times.iter().zip(&count).enumerate() {
        let (gap, length, triggers) = (t[1], t[2], c[1]);
        if t[0] != i as f64 || c[0] != i as f64 {
            return Err(IoError::InvalidParameter(format!(
                "shutter row {i} has index {} and {}",
                t[0], c[0]
            )));
        }
        if length == 0.0 && triggers == 0.0 {
            continue;
        }
        if !(gap >= 0.0 && length > 0.0 && triggers > 0.0 && triggers.fract() == 0.0) {
            return Err(IoError::InvalidParameter(format!(
                "shutter row {i}: gap {gap} s, length {length} s and {triggers} triggers \
                 do not describe a used shutter"
            )));
        }
        let start = closed + gap;
        closed = start + length;
        let shutter = Shutter {
            start_us: start * MICROSECONDS_PER_SECOND,
            end_us: closed * MICROSECONDS_PER_SECOND,
            triggers,
        };
        if !shutter.end_us.is_finite() {
            return Err(IoError::InvalidParameter(format!(
                "shutter row {i} closes at {} µs",
                shutter.end_us
            )));
        }
        shutters.push(shutter);
    }
    if times.len() != count.len() || shutters.is_empty() {
        return Err(IoError::InvalidParameter(format!(
            "{} shutter times and {} shutter counts with {} used shutters",
            times.len(),
            count.len(),
            shutters.len()
        )));
    }
    Ok(shutters)
}

/// Load one chunk of a raw frame-mode run folder.
///
/// # Arguments
/// * `dir` — The run folder.
/// * `chunk` — The chunk to load; `None` when the folder holds one chunk.
///
/// # Errors
/// [`IoError::NoMatchingFiles`] if no frames are found;
/// [`IoError::InvalidParameter`] if `chunk` is `None` and the folder holds
/// several chunks, the chunk's frames have two prefixes or are not numbered
/// 0 to n − 1, a sidecar is missing or disagrees with the frames, or a frame
/// is unreadable;
/// [`IoError::DimensionMismatch`] if frame sizes differ.
pub fn load_fits_run(dir: &Path, chunk: Option<&str>) -> Result<FitsRun, IoError> {
    let mut frames: Vec<(String, String, usize, PathBuf)> = std::fs::read_dir(dir)
        .map_err(|e| IoError::FileNotFound(dir.to_string_lossy().into_owned(), e))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter_map(|p| name_parts(&p).map(|(prefix, c, f)| (prefix, c, f, p)))
        .collect();
    let mut chunks: Vec<&str> = frames.iter().map(|(_, c, _, _)| c.as_str()).collect();
    chunks.sort_unstable();
    chunks.dedup();
    let chunk = match (chunk, chunks.as_slice()) {
        (Some(c), _) => c.to_string(),
        (None, [one]) => one.to_string(),
        (None, []) => {
            return Err(IoError::NoMatchingFiles {
                directory: dir.to_string_lossy().into_owned(),
                pattern: "<prefix>_<chunk>_<frame>.fits".into(),
            });
        }
        (None, many) => {
            return Err(IoError::InvalidParameter(format!(
                "{} holds chunks {many:?}; name the one to load",
                dir.display()
            )));
        }
    };
    frames.retain(|(_, c, _, _)| *c == chunk);
    frames.sort_by_key(|(_, _, f, _)| *f);
    if frames.is_empty() {
        return Err(IoError::NoMatchingFiles {
            directory: dir.to_string_lossy().into_owned(),
            pattern: format!("<prefix>_{chunk}_<frame>.fits"),
        });
    }
    if let Some((i, (_, _, f, _))) = frames.iter().enumerate().find(|(i, (_, _, f, _))| f != i) {
        return Err(IoError::InvalidParameter(format!(
            "chunk {chunk} frame {f} is at position {i}; frames must be numbered 0 to n − 1"
        )));
    }
    let prefix = frames[0].0.clone();
    if let Some((other, _, f, _)) = frames.iter().find(|(p, _, _, _)| *p != prefix) {
        return Err(IoError::InvalidParameter(format!(
            "chunk {chunk} frame {f} is named {other}, not {prefix}"
        )));
    }

    let first = read_fits_image(&frames[0].3)?;
    let (height, width) = first.dim();
    let mut counts = Array3::zeros((frames.len(), height, width));
    counts.index_axis_mut(ndarray::Axis(0), 0).assign(&first);
    for (k, (_, _, _, path)) in frames.iter().enumerate().skip(1) {
        let image = read_fits_image(path)?;
        if image.dim() != (height, width) {
            return Err(IoError::DimensionMismatch {
                expected: (width as u32, height as u32),
                got: (image.ncols() as u32, image.nrows() as u32),
                frame: k,
            });
        }
        counts.index_axis_mut(ndarray::Axis(0), k).assign(&image);
    }

    let edges_us = read_tof_sidecar(
        &sidecar(dir, &prefix, &chunk, "Spectra.txt")?,
        Some(frames.len()),
    )?;
    let shutters = read_shutters(
        &sidecar(dir, &prefix, &chunk, "ShutterTimes.txt")?,
        &sidecar(dir, &prefix, &chunk, "ShutterCount.txt")?,
    )?;
    Ok(FitsRun {
        counts,
        edges_us,
        shutters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(text: &str) -> String {
        format!("{text:<80}")
    }

    /// A FITS file of 16-bit values `raw` stored with `BZERO = zero`.
    fn fits(width: usize, raw: &[i16], zero: f64) -> Vec<u8> {
        let mut header = [
            card("SIMPLE  =                    T"),
            card("BITPIX  =                   16"),
            card("NAXIS   =                    2"),
            card(&format!("NAXIS1  = {width:>20}")),
            card(&format!("NAXIS2  = {:>20}", raw.len() / width)),
            card(&format!("BZERO   = {zero:>20} / offset")),
            card("END"),
        ]
        .concat()
        .into_bytes();
        header.resize(BLOCK, b' ');
        header.extend(raw.iter().flat_map(|v| v.to_be_bytes()));
        header.resize(header.len().div_ceil(BLOCK) * BLOCK, 0);
        header
    }

    #[test]
    fn a_fits_image_reads_in_row_order_with_its_offset() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.fits");
        std::fs::write(&path, fits(3, &[0, 1, 2, -32768, 5, 7], 32768.0)).unwrap();
        let image = read_fits_image(&path).unwrap();
        assert_eq!(image.dim(), (2, 3));
        assert_eq!(image.row(0).to_vec(), vec![32768.0, 32769.0, 32770.0]);
        assert_eq!(image.row(1).to_vec(), vec![0.0, 32773.0, 32775.0]);

        std::fs::write(&path, fits(2, &[3, -1], 0.0)).unwrap();
        assert!(read_fits_image(&path).is_err());

        std::fs::write(&path, fits(1, &[1], 0.5)).unwrap();
        assert!(read_fits_image(&path).is_err());

        let mut scaled = fits(1, &[100], 0.0);
        scaled[5 * 80..6 * 80].copy_from_slice(card("BSCALE  =                 0.07").as_bytes());
        std::fs::write(&path, scaled).unwrap();
        assert_eq!(read_fits_image(&path).unwrap()[[0, 0]], 7.0);

        let mut large = fits(1, &[1], 1e9);
        large[6 * 80..8 * 80]
            .copy_from_slice((card("BSCALE  =                  0.5") + &card("END")).as_bytes());
        std::fs::write(&path, large).unwrap();
        assert!(read_fits_image(&path).is_err());

        let mut half = fits(3, &[0, 1, 2, 3, 4, 5], 0.0);
        half[3 * 80 + 27..3 * 80 + 30].copy_from_slice(b"1.5");
        std::fs::write(&path, half).unwrap();
        assert!(read_fits_image(&path).is_err());
    }

    fn write_run(dir: &Path, chunk: &str, frames: &[[i16; 2]]) {
        for (k, f) in frames.iter().enumerate() {
            let name = format!("run_0_{chunk}_{k:05}.fits");
            std::fs::write(dir.join(name), fits(2, f, 0.0)).unwrap();
        }
        std::fs::write(
            dir.join(format!("run_0_{chunk}_SummedImg.fits")),
            fits(2, &[9, 9], 0.0),
        )
        .unwrap();
        let spectra = "1e-06\t5\r\n2e-06\t6\r\n3e-06\t7\r\n4.5e-06\t1\r\n";
        std::fs::write(dir.join(format!("run_0_{chunk}_Spectra.txt")), spectra).unwrap();
        let times = "0\t1e-06\t2.5e-06\r\n1\t5e-07\t2e-06\r\n2\t0\t0\r\n";
        std::fs::write(dir.join(format!("run_0_{chunk}_ShutterTimes.txt")), times).unwrap();
        let count = "0\t100\r\n1\t90\r\n2\t0\r\n";
        std::fs::write(dir.join(format!("run_0_{chunk}_ShutterCount.txt")), count).unwrap();
    }

    #[test]
    fn a_run_loads_its_frames_edges_and_shutters() {
        let dir = tempfile::tempdir().unwrap();
        write_run(dir.path(), "114", &[[1, 2], [3, 4], [5, 6], [7, 8]]);
        let run = load_fits_run(dir.path(), None).unwrap();
        assert_eq!(run.counts.shape(), &[4, 1, 2]);
        assert_eq!(run.counts[[3, 0, 1]], 8.0);
        assert_eq!(run.edges_us.len(), 5);
        assert_eq!(run.shutters.len(), 2);
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        for (&edge, expected) in run.edges_us.iter().zip([1.0, 2.0, 3.0, 4.5, 6.0]) {
            assert!(close(edge, expected));
        }
        let [s0, s1] = [run.shutters[0], run.shutters[1]];
        assert!(close(s0.start_us, 1.0) && close(s0.end_us, 3.5) && s0.triggers == 100.0);
        assert!(close(s1.start_us, 4.0) && close(s1.end_us, 6.0) && s1.triggers == 90.0);
    }

    #[test]
    fn a_folder_of_two_chunks_needs_the_chunk_named() {
        let dir = tempfile::tempdir().unwrap();
        write_run(dir.path(), "118", &[[1, 2], [3, 4], [5, 6], [7, 8]]);
        write_run(dir.path(), "119", &[[1, 2], [3, 4], [5, 6], [7, 8]]);
        assert!(load_fits_run(dir.path(), None).is_err());
        assert_eq!(
            load_fits_run(dir.path(), Some("119"))
                .unwrap()
                .counts
                .shape()[0],
            4
        );

        std::fs::remove_file(dir.path().join("run_0_118_00002.fits")).unwrap();
        assert!(load_fits_run(dir.path(), Some("118")).is_err());
        std::fs::write(
            dir.path().join("other_0_118_00002.fits"),
            fits(2, &[1, 1], 0.0),
        )
        .unwrap();
        assert!(load_fits_run(dir.path(), Some("118")).is_err());
    }

    #[test]
    fn shutter_rows_need_every_index_and_whole_triggers() {
        let dir = tempfile::tempdir().unwrap();
        write_run(dir.path(), "114", &[[1, 2], [3, 4], [5, 6], [7, 8]]);
        let count = dir.path().join("run_0_114_ShutterCount.txt");
        let times = dir.path().join("run_0_114_ShutterTimes.txt");
        std::fs::write(&count, "0\t100.5\r\n1\t90\r\n2\t0\r\n").unwrap();
        assert!(load_fits_run(dir.path(), None).is_err());
        std::fs::write(&count, "0\t100\r\n2\t90\r\n").unwrap();
        std::fs::write(&times, "0\t1e-06\t2.5e-06\r\n2\t5e-07\t2e-06\r\n").unwrap();
        assert!(load_fits_run(dir.path(), None).is_err());
    }
}
