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

use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once};
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

/// Most recent rfd error record, latched by the log bridge.
///
/// rfd reports user-cancel as a bare `None` with no diagnostic, but it
/// emits `log`-crate error records for backend trouble — including
/// NON-final legs: on Linux the portal leg log-errors ("Failed to
/// connect to session bus: ...", "OpenFile failed: ...") and then
/// falls back to zenity, which may still serve the user. A latched
/// record therefore means "some leg failed", not "the dialog failed".
/// The latch stays deliberately broad (any rfd error); interpretation
/// lives in the dialog facade, which combines the latch with the
/// request's outcome to tell a broken chain from a fallback that
/// worked (see `FileDialogs::apply_native_outcome` in
/// `file_dialog.rs`).
static DIALOG_BACKEND_FAILURE: Mutex<Option<String>> = Mutex::new(None);

/// Take the latched native-dialog backend failure, if any (consume-once).
pub fn take_dialog_backend_failure() -> Option<String> {
    DIALOG_BACKEND_FAILURE
        .lock()
        .ok()
        .and_then(|mut slot| slot.take())
}

/// Bridge `log`-crate records (rfd, opener, other C-adjacent deps emit
/// these; without a bridge they are silently discarded) into `tracing`,
/// and latch rfd error records for the dialog facade.
struct LogBridge;

impl log::Log for LogBridge {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        // Coarse gate mirroring the tracing filter's max level (set in
        // `init_inner`), so `log_enabled!`-guarded expensive formatting
        // in dependencies is skipped instead of forwarded and then
        // dropped. This global bound is the ONLY level filtering
        // bridged records get: they re-enter tracing under this
        // module's target (the macros below), so per-target EnvFilter
        // directives (e.g. `NEREIDS_LOG=rfd=debug`) never match them —
        // the original target survives only as the `log_target` field.
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &log::Record<'_>) {
        // The original target is carried as a field because tracing
        // macro targets must be compile-time constants.
        match record.level() {
            log::Level::Error => {
                tracing::error!(log_target = record.target(), "{}", record.args());
            }
            log::Level::Warn => {
                tracing::warn!(log_target = record.target(), "{}", record.args());
            }
            log::Level::Info => {
                tracing::info!(log_target = record.target(), "{}", record.args());
            }
            log::Level::Debug => {
                tracing::debug!(log_target = record.target(), "{}", record.args());
            }
            log::Level::Trace => {
                tracing::trace!(log_target = record.target(), "{}", record.args());
            }
        }

        if record.level() == log::Level::Error
            && record.target().starts_with("rfd")
            && let Ok(mut slot) = DIALOG_BACKEND_FAILURE.lock()
        {
            *slot = Some(record.args().to_string());
        }
    }

    fn flush(&self) {}
}

static LOG_BRIDGE: LogBridge = LogBridge;

/// Map the tracing filter's static max-level hint onto the `log`
/// crate's global level, so the bridge's `enabled()` can gate
/// `log_enabled!`-guarded work in dependencies at the same bound the
/// EnvFilter would apply. `None` (no static bound, e.g. dynamic
/// reloading) stays permissive.
fn log_level_from_hint(hint: Option<tracing::level_filters::LevelFilter>) -> log::LevelFilter {
    use tracing::level_filters::LevelFilter;
    let Some(hint) = hint else {
        return log::LevelFilter::Trace;
    };
    if hint == LevelFilter::OFF {
        log::LevelFilter::Off
    } else if hint == LevelFilter::ERROR {
        log::LevelFilter::Error
    } else if hint == LevelFilter::WARN {
        log::LevelFilter::Warn
    } else if hint == LevelFilter::INFO {
        log::LevelFilter::Info
    } else if hint == LevelFilter::DEBUG {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Trace
    }
}

/// Returns the directory where rolling log files are written, creating
/// the directory tree on demand. Falls back to `.cache/NEREIDS/logs`
/// (relative to cwd) if `dirs::data_dir()` is unavailable.
///
/// Directory-creation failure is reported through [`init`]'s diagnostic
/// log line; callers of `log_dir` (e.g. the toolbar Help menu) get the
/// would-be path either way so the UI can still display it. For the
/// fallible variant, see [`compute_log_dir`].
pub fn log_dir() -> PathBuf {
    let base = dirs::data_dir().unwrap_or_else(|| PathBuf::from(".cache"));
    compute_log_dir(&base).unwrap_or_else(|(path, _err)| path)
}

/// Pure variant of [`log_dir`] that takes the base data directory as a
/// parameter — used by tests so they can point at a `tempfile::TempDir`
/// instead of mutating the dev's real user-data folder, and by [`init`]
/// so a creation failure can be surfaced through the post-init log.
///
/// On failure, returns `Err((would_be_path, io::Error))` so callers
/// can still report or display the intended path.
fn compute_log_dir(base: &Path) -> Result<PathBuf, (PathBuf, io::Error)> {
    let dir = base.join(APP_DIR_NAME).join("logs");
    match std::fs::create_dir_all(&dir) {
        Ok(()) => Ok(dir),
        Err(err) => Err((dir, err)),
    }
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
/// Idempotent: subsequent calls are no-ops. Gating with [`Once`]
/// prevents the panic-hook chain from stacking on accidental double-init
/// (each call would otherwise wrap the previous hook, producing
/// duplicated panic-log records).
///
/// Never panics. On rolling-appender failure (unwritable directory,
/// full disk, sandbox jail) falls back to a stderr-only subscriber and
/// emits a `tracing::error!` describing the failure. The `WorkerGuard`
/// (when the file appender came up) is stashed in a process-wide
/// `Mutex<Option<WorkerGuard>>` and dropped by [`shutdown`] before any
/// `std::process::exit(0)`.
pub fn init() {
    static INITIALISED: Once = Once::new();
    INITIALISED.call_once(init_inner);
}

fn init_inner() {
    let base = dirs::data_dir().unwrap_or_else(|| PathBuf::from(".cache"));
    let dir_result = compute_log_dir(&base);
    // Either the canonical dir (success) or the would-be path (failure)
    // — both are useful: the appender build below will fail-fast on
    // the latter, and the diagnostic log will name the path the user
    // can investigate.
    let dir = match &dir_result {
        Ok(p) => p.clone(),
        Err((p, _)) => p.clone(),
    };

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
    // (e.g. accidental double-init in tests, though our `Once` gate
    // makes that path unreachable from our own callers). Swallow it —
    // the existing subscriber stays in charge.
    let _ = Registry::default()
        .with(file_layer)
        .with(stderr_layer)
        .try_init();

    install_panic_hook();

    // Forward `log`-crate records into tracing. Bridged records adopt
    // this module's tracing target, so the EnvFilter applies only its
    // global level bound to them — per-original-target directives do
    // not match, and the source target is carried as the `log_target`
    // field. The log-crate global level mirrors that coarsest bound so
    // dependencies' `log_enabled!` guards short-circuit at the same
    // threshold.
    // `set_logger` only fails if a logger is already installed — then
    // that one stays in charge.
    if log::set_logger(&LOG_BRIDGE).is_ok() {
        let hint = <EnvFilter as tracing_subscriber::layer::Filter<Registry>>::max_level_hint(
            &env_filter(),
        );
        log::set_max_level(log_level_from_hint(hint));
    }

    // Surface the dir-creation failure (if any) now that the subscriber
    // is installed. Recorded as a warn so it stands out without being
    // alarming on filesystems where this is genuinely transient.
    if let Err((path, ref err)) = dir_result {
        tracing::warn!(
            dir = %path.display(),
            error = %err,
            "failed to create log directory; logging may be degraded"
        );
    }

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
    use log::Log;
    use tempfile::tempdir;

    #[test]
    fn compute_log_dir_tail_is_nereids_logs() {
        let tmp = tempdir().expect("tempdir");
        let dir = compute_log_dir(tmp.path()).expect("create log dir");
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
        let dir = compute_log_dir(tmp.path()).expect("create log dir");
        assert!(
            dir.exists(),
            "compute_log_dir should create the directory: {}",
            dir.display()
        );
        assert!(dir.is_dir(), "expected a directory");
    }

    #[test]
    fn compute_log_dir_surfaces_creation_error() {
        // Point at a path that cannot become a directory: a regular
        // file already exists at the would-be parent path. mkdir -p
        // refuses to create children under a non-directory.
        let tmp = tempdir().expect("tempdir");
        let blocker = tmp.path().join("blocker");
        std::fs::write(&blocker, b"not a directory").expect("write blocker file");
        let result = compute_log_dir(&blocker);
        let Err((path, _err)) = result else {
            panic!("expected Err, got {result:?}");
        };
        // The reported path is still the would-be log dir, useful for
        // post-init diagnostics.
        assert!(path.ends_with("NEREIDS/logs"));
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

    #[test]
    fn log_level_hint_mapping_covers_every_level() {
        use tracing::level_filters::LevelFilter as Hint;
        assert_eq!(log_level_from_hint(None), log::LevelFilter::Trace);
        assert_eq!(log_level_from_hint(Some(Hint::OFF)), log::LevelFilter::Off);
        assert_eq!(
            log_level_from_hint(Some(Hint::ERROR)),
            log::LevelFilter::Error
        );
        assert_eq!(
            log_level_from_hint(Some(Hint::WARN)),
            log::LevelFilter::Warn
        );
        assert_eq!(
            log_level_from_hint(Some(Hint::INFO)),
            log::LevelFilter::Info
        );
        assert_eq!(
            log_level_from_hint(Some(Hint::DEBUG)),
            log::LevelFilter::Debug
        );
        assert_eq!(
            log_level_from_hint(Some(Hint::TRACE)),
            log::LevelFilter::Trace
        );
    }

    /// Feed records to the bridge directly (no global logger needed,
    /// and `enabled()` is deliberately bypassed — `log()` itself never
    /// re-checks the gate): rfd *error* records reach the latch from
    /// EVERY leg of the chain — warns and other crates' errors don't —
    /// and the latch is consume-once. Targets and messages mirror rfd
    /// 0.17.2's real emission sites (`module_path!` targets:
    /// src/backend/xdg_desktop_portal/portal/libdbus.rs for the portal
    /// leg, src/backend/xdg_desktop_portal.rs for the zenity leg).
    #[test]
    fn log_bridge_latches_only_rfd_errors() {
        let bridge = LogBridge;

        // Drain anything a concurrent test may have latched.
        let _ = take_dialog_backend_failure();

        // The zenity-fallback notice is a warn, not an error; foreign
        // crates' errors are not rfd's. Neither latches.
        bridge.log(
            &log::Record::builder()
                .level(log::Level::Warn)
                .target("rfd::backend::xdg_desktop_portal")
                .args(format_args!("Using zenity fallback"))
                .build(),
        );
        bridge.log(
            &log::Record::builder()
                .level(log::Level::Error)
                .target("some_other_crate")
                .args(format_args!("unrelated error"))
                .build(),
        );
        assert_eq!(take_dialog_backend_failure(), None);

        // Portal-leg error (emitted before rfd tries zenity): latched.
        bridge.log(
            &log::Record::builder()
                .level(log::Level::Error)
                .target("rfd::backend::xdg_desktop_portal::portal::libdbus")
                .args(format_args!(
                    "Failed to connect to session bus: org.freedesktop.DBus.Error.NoServer"
                ))
                .build(),
        );
        let latched = take_dialog_backend_failure();
        assert!(
            latched
                .as_deref()
                .is_some_and(|m| m.contains("session bus")),
            "expected portal-leg failure latched, got {latched:?}"
        );

        // Final zenity-leg error: latched, and consume-once.
        bridge.log(
            &log::Record::builder()
                .level(log::Level::Error)
                .target("rfd::backend::xdg_desktop_portal")
                .args(format_args!("Failed to pick file with zenity: not found"))
                .build(),
        );
        let latched = take_dialog_backend_failure();
        assert!(
            latched.as_deref().is_some_and(|m| m.contains("zenity")),
            "expected zenity failure latched, got {latched:?}"
        );
        assert_eq!(take_dialog_backend_failure(), None);
    }
}
