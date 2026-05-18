//! File-based tracing setup + log-path helpers for the desktop GUI.
//!
//! Logs land in a per-OS user data directory (issue #524):
//! - macOS:   `~/Library/Application Support/NEREIDS/logs/`
//! - Linux:   `~/.local/share/NEREIDS/logs/` (honours `$XDG_DATA_HOME`)
//! - Windows: `%APPDATA%\NEREIDS\logs\`
//!
//! Rotation: daily, retain 7 files. The rolling appender writes
//! `nereids-gui.YYYY-MM-DD.log` (date is UTC). Filter precedence:
//! `NEREIDS_LOG` beats `RUST_LOG`; default `info`. Both the rolling
//! file and stderr receive the same records (stderr stays useful when
//! the binary is launched from a terminal). Panics are captured to the
//! log with a forced backtrace before delegating to the previous panic
//! hook so default stderr output is preserved.
//!
//! Initialisation never panics — if the file appender cannot be built
//! (e.g. read-only home directory, full disk, sandboxed bundle), we
//! fall back to a stderr-only subscriber and emit a `tracing::error!`
//! describing the failure. The app stays usable.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use time::OffsetDateTime;
use time::macros::format_description;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{EnvFilter, fmt, prelude::*, registry::Registry};

const APP_DIR_NAME: &str = "NEREIDS";
const LOG_FILE_PREFIX: &str = "nereids-gui";
const LOG_FILE_SUFFIX: &str = "log";
const MAX_LOG_FILES: usize = 7;

/// Holds the rolling-appender's `WorkerGuard` so it stays alive for the
/// process lifetime. Dropped explicitly by [`shutdown`] before any of
/// the macOS `std::process::exit(0)` calls in `app.rs`, otherwise the
/// last buffered records would be lost (unwind never runs through
/// `process::exit`).
static GUARD: Mutex<Option<WorkerGuard>> = Mutex::new(None);

/// Returns the directory where rolling log files are written, creating
/// the directory tree on demand. Falls back to `.cache/NEREIDS/logs`
/// (relative to cwd) if `dirs::data_dir()` is unavailable.
///
/// Note: a failure to create the directory is intentionally swallowed
/// here. The caller ([`init`]) handles the downstream
/// `RollingFileAppender::build` failure that follows.
pub fn log_dir() -> PathBuf {
    let base = dirs::data_dir().unwrap_or_else(|| PathBuf::from(".cache"));
    compute_log_dir(&base)
}

/// Pure variant of [`log_dir`] that takes the base data directory as a
/// parameter — used by tests so they can point at a [`tempfile::TempDir`]
/// instead of mutating the dev's real user-data folder.
fn compute_log_dir(base: &Path) -> PathBuf {
    let dir = base.join(APP_DIR_NAME).join("logs");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Returns the path to today's log file as written by the rolling
/// appender — `<log_dir>/nereids-gui.<UTC-date>.log`. Matches the
/// naming scheme used internally by `tracing_appender::rolling::RollingFileAppender`
/// when configured with `Rotation::DAILY` + the same prefix/suffix.
///
/// The file may not exist yet on first launch (it's created on the
/// first write), but the directory is guaranteed to exist.
pub fn log_file_path() -> PathBuf {
    let date = current_utc_date();
    log_dir().join(format!("{LOG_FILE_PREFIX}.{date}.{LOG_FILE_SUFFIX}"))
}

/// Today's UTC date formatted as `YYYY-MM-DD`. Matches
/// `tracing_appender::rolling::Rotation::DAILY`'s internal format,
/// which uses `time::OffsetDateTime::now_utc()` + the same components.
fn current_utc_date() -> String {
    let now = OffsetDateTime::now_utc();
    let fmt = format_description!("[year]-[month]-[day]");
    // `format` only fails if the format description is invalid (it
    // isn't) or if the date is out of range (impossible for now()).
    // Fall back to a sentinel rather than panic, so logging is
    // resilient even on extreme system clock corruption.
    now.format(&fmt)
        .unwrap_or_else(|_| "unknown-date".to_string())
}

/// Initialise the global tracing subscriber.
///
/// Never panics. On rolling-appender failure (unwritable directory,
/// full disk, sandbox jail) falls back to a stderr-only subscriber
/// and emits a `tracing::error!` describing the failure. The returned
/// `WorkerGuard` (when the file appender came up) is stashed in a
/// process-wide `Mutex<Option<WorkerGuard>>` and dropped by
/// [`shutdown`] before any `std::process::exit(0)`.
pub fn init() {
    let dir = log_dir();
    let appender_result = RollingFileAppender::builder()
        .rotation(Rotation::DAILY)
        .filename_prefix(LOG_FILE_PREFIX)
        .filename_suffix(LOG_FILE_SUFFIX)
        .max_log_files(MAX_LOG_FILES)
        .build(&dir);

    // EnvFilter doesn't implement Clone; build a fresh one per layer.
    let env_filter = || {
        EnvFilter::try_from_env("NEREIDS_LOG")
            .or_else(|_| EnvFilter::try_from_default_env())
            .unwrap_or_else(|_| EnvFilter::new("info"))
    };

    // Consume the appender once — it isn't Clone. Capture any error
    // string for the post-init `tracing::error!` and turn the success
    // path into an `Option<Layer>`. `Option<L>: Layer<S>` when `L:
    // Layer<S>`, which lets both arms flow through the same Registry
    // builder chain (avoids match-arm type-unification problems).
    let (file_layer, appender_error) = match appender_result {
        Ok(file_appender) => {
            let (non_blocking_file, guard) = tracing_appender::non_blocking(file_appender);
            if let Ok(mut slot) = GUARD.lock() {
                *slot = Some(guard);
            }
            let layer = fmt::layer()
                .with_writer(non_blocking_file)
                .with_ansi(false)
                .with_target(true)
                .with_filter(env_filter());
            (Some(layer), None)
        }
        Err(err) => (None, Some(err.to_string())),
    };

    let stderr_layer = fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(true)
        .with_target(true)
        .with_filter(env_filter());

    // `try_init` returns Err if another subscriber is already installed
    // (e.g. accidental double-init in tests). Swallow it — the existing
    // subscriber stays in charge.
    let _ = Registry::default()
        .with(file_layer)
        .with(stderr_layer)
        .try_init();

    install_panic_hook();

    match appender_error {
        None => {
            tracing::info!(
                version = env!("CARGO_PKG_VERSION"),
                log_path = %log_file_path().display(),
                "nereids-gui starting"
            );
        }
        Some(err) => {
            tracing::error!(
                version = env!("CARGO_PKG_VERSION"),
                log_dir = %dir.display(),
                error = %err,
                "rolling log appender unavailable; falling back to stderr-only logging"
            );
        }
    }
}

/// Drop the stashed `WorkerGuard`, forcing the non-blocking writer to
/// flush pending records. Must be called before any
/// `std::process::exit(0)` because that bypasses normal stack
/// unwinding (and therefore Drop) on the `_log_guard` binding.
pub fn shutdown() {
    if let Ok(mut slot) = GUARD.lock() {
        // Take and drop, forcing WorkerGuard::drop to flush.
        let _ = slot.take();
    }
}

fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let backtrace = std::backtrace::Backtrace::force_capture();
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown>".to_string());
        let payload = info
            .payload()
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| info.payload().downcast_ref::<String>().map(String::as_str))
            .unwrap_or("<non-string panic payload>");
        tracing::error!(
            location = %location,
            backtrace = %backtrace,
            "panic: {payload}"
        );
        previous(info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn compute_log_dir_tail_is_nereids_logs() {
        let tmp = tempdir().expect("tempdir");
        let dir = compute_log_dir(tmp.path());
        let tail: Vec<String> = dir
            .components()
            .rev()
            .take(2)
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        assert_eq!(tail, vec!["logs".to_string(), "NEREIDS".to_string()]);
    }

    #[test]
    fn compute_log_dir_creates_the_directory() {
        let tmp = tempdir().expect("tempdir");
        let dir = compute_log_dir(tmp.path());
        assert!(
            dir.exists(),
            "compute_log_dir should create the directory: {}",
            dir.display()
        );
        assert!(dir.is_dir(), "expected a directory");
    }

    #[test]
    fn log_file_path_uses_dated_filename() {
        // Match the appender's `<prefix>.<UTC-date>.<suffix>` shape.
        let p = log_file_path();
        let name = p
            .file_name()
            .and_then(|s| s.to_str())
            .expect("file name str");
        assert!(
            name.starts_with("nereids-gui."),
            "expected dated filename, got {name:?}"
        );
        assert!(name.ends_with(".log"), "expected .log suffix, got {name:?}");
        // Middle segment is YYYY-MM-DD (10 chars).
        let middle = &name["nereids-gui.".len()..name.len() - ".log".len()];
        assert_eq!(
            middle.len(),
            10,
            "expected YYYY-MM-DD between prefix and suffix, got {middle:?}"
        );
        assert!(
            middle
                .chars()
                .enumerate()
                .all(|(i, c)| matches!((i, c), (4, '-') | (7, '-') | (_, '0'..='9'))),
            "expected YYYY-MM-DD digits + dashes, got {middle:?}"
        );
    }

    #[test]
    fn current_utc_date_is_yyyy_mm_dd() {
        let s = current_utc_date();
        assert_eq!(s.len(), 10, "got {s:?}");
        assert_eq!(s.as_bytes()[4], b'-');
        assert_eq!(s.as_bytes()[7], b'-');
    }
}
