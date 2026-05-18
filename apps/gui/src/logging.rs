//! File-based tracing setup + log-path helpers for the desktop GUI.
//!
//! Logs land in a per-OS user data directory (issue #524):
//! - macOS:   `~/Library/Application Support/NEREIDS/logs/`
//! - Linux:   `~/.local/share/NEREIDS/logs/` (honours `$XDG_DATA_HOME`)
//! - Windows: `%APPDATA%\NEREIDS\logs\`
//!
//! Rotation: daily, retain 7 files. Filter precedence: `NEREIDS_LOG`
//! beats `RUST_LOG`; default `info`. Both the rolling file and stderr
//! receive the same records (stderr stays useful when the binary is
//! launched from a terminal). Panics are captured to the log with a
//! forced backtrace before delegating to the previous panic hook so
//! default stderr output is preserved.

use std::path::PathBuf;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{EnvFilter, fmt, prelude::*, registry::Registry};

const APP_DIR_NAME: &str = "NEREIDS";
const LOG_FILE_PREFIX: &str = "nereids-gui";
const LOG_FILE_SUFFIX: &str = "log";
const LOG_FILE_NAME: &str = "nereids-gui.log";
const MAX_LOG_FILES: usize = 7;

/// Returns the directory where rolling log files are written, creating
/// the directory tree on demand. Falls back to `.cache/NEREIDS/logs`
/// (relative to cwd) if `dirs::data_dir()` is unavailable.
pub fn log_dir() -> PathBuf {
    let dir = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join(APP_DIR_NAME)
        .join("logs");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Returns the absolute path to the active (un-rotated) log file —
/// `<log_dir>/nereids-gui.log`. The file may not exist until the first
/// log line is emitted, but `log_dir()` is guaranteed to exist.
pub fn log_file_path() -> PathBuf {
    log_dir().join(LOG_FILE_NAME)
}

/// Initialise the global tracing subscriber.
///
/// The returned [`WorkerGuard`] MUST be bound to a local in `main()`
/// (e.g. `let _guard = logging::init();`) — when it drops, pending log
/// records are flushed.
#[must_use = "WorkerGuard must outlive main() to flush buffered log records on shutdown"]
pub fn init() -> WorkerGuard {
    let dir = log_dir();

    let file_appender = RollingFileAppender::builder()
        .rotation(Rotation::DAILY)
        .filename_prefix(LOG_FILE_PREFIX)
        .filename_suffix(LOG_FILE_SUFFIX)
        .max_log_files(MAX_LOG_FILES)
        .build(&dir)
        .expect("failed to initialise rolling log appender");

    let (non_blocking_file, guard) = tracing_appender::non_blocking(file_appender);

    // EnvFilter doesn't implement Clone; build a fresh one per layer.
    let env_filter = || {
        EnvFilter::try_from_env("NEREIDS_LOG")
            .or_else(|_| EnvFilter::try_from_default_env())
            .unwrap_or_else(|_| EnvFilter::new("info"))
    };

    let file_layer = fmt::layer()
        .with_writer(non_blocking_file)
        .with_ansi(false)
        .with_target(true)
        .with_filter(env_filter());

    let stderr_layer = fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(true)
        .with_target(true)
        .with_filter(env_filter());

    // try_init so an accidental double-init in tests is a swallowed
    // Err, not a panic.
    let _ = Registry::default()
        .with(file_layer)
        .with(stderr_layer)
        .try_init();

    install_panic_hook();

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        log_path = %log_file_path().display(),
        "nereids-gui starting"
    );

    guard
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

    #[test]
    fn log_dir_tail_is_nereids_logs() {
        let dir = log_dir();
        let tail: Vec<String> = dir
            .components()
            .rev()
            .take(2)
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        assert_eq!(tail, vec!["logs".to_string(), "NEREIDS".to_string()]);
    }

    #[test]
    fn log_dir_is_created() {
        let dir = log_dir();
        assert!(
            dir.exists(),
            "log_dir() should create the directory: {}",
            dir.display()
        );
        assert!(dir.is_dir(), "log_dir() should resolve to a directory");
    }

    #[test]
    fn log_file_path_under_log_dir() {
        let p = log_file_path();
        assert!(p.starts_with(log_dir()));
        assert_eq!(
            p.file_name().and_then(|s| s.to_str()),
            Some("nereids-gui.log")
        );
    }
}
