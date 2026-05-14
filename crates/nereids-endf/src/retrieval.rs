//! ENDF file download and local caching.
//!
//! Downloads ENDF files from official NNDC/IAEA sources and caches them locally
//! for offline use. Follows the IAEA URL patterns established by PLEIADES.
//!
//! ## PLEIADES Reference
//! - `pleiades/nuclear/manager.py` — URL construction, cache directory layout
//! - `pleiades/nuclear/models.py` — library enum, filename patterns

use nereids_core::elements;
use nereids_core::types::Isotope;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

const IAEA_BASE_URL: &str = "https://www-nds.iaea.org/public/download-endf";
const NNDC_ENDF_BASE_URL: &str = "https://www.nndc.bnl.gov/endf-data/ENDF";
// Polite, identifiable UA: many nuclear-data servers either require a non-default
// UA or treat default `reqwest` traffic as a bot (issue #523, IAEA returning
// HTTP 403 to v0.1.8 batch fetches). Version is derived from `CARGO_PKG_VERSION`
// so it never drifts from `Cargo.toml`.
const ENDF_USER_AGENT: &str = concat!(
    "NEREIDS/",
    env!("CARGO_PKG_VERSION"),
    " (https://github.com/ornlneutronimaging/NEREIDS; contact: zhangc@ornl.gov)",
);
const IAEA_MIN_REQUEST_INTERVAL: Duration = Duration::from_secs(3);

static LAST_IAEA_REQUEST: OnceLock<Mutex<Option<Instant>>> = OnceLock::new();

/// ENDF evaluated nuclear data libraries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndfLibrary {
    /// ENDF/B-VIII.0 (default, well-tested).
    EndfB8_0,
    /// ENDF/B-VIII.1 (latest release, Aug 2024).
    EndfB8_1,
    /// JEFF-3.3 (European library).
    Jeff3_3,
    /// JENDL-5 (Japanese library).
    Jendl5,
    /// TENDL-2023 (TALYS-based, 2,300 ground-state isotopes including activation
    /// products and transuranics not covered by the major evaluated libraries).
    Tendl2023,
    /// CENDL-3.2 (Chinese library, 258 ground-state isotopes plus free neutron;
    /// Z=1–98 with no Br evaluations — no MAT entry for Br-79 / Br-81, so
    /// `mat_number(.., EndfLibrary::Cendl3_2)` returns `None` for Br before any
    /// retrieval call).
    Cendl3_2,
}

impl EndfLibrary {
    /// URL path component for this library.
    fn url_path(&self) -> &'static str {
        match self {
            Self::EndfB8_0 => "ENDF-B-VIII.0/n",
            Self::EndfB8_1 => "ENDF-B-VIII.1/n",
            Self::Jeff3_3 => "JEFF-3.3/n",
            Self::Jendl5 => "JENDL-5/n",
            Self::Tendl2023 => "TENDL-2023/n",
            Self::Cendl3_2 => "CENDL-3.2/n",
        }
    }

    /// Cache directory name.
    fn cache_dir_name(&self) -> &'static str {
        match self {
            Self::EndfB8_0 => "ENDF-B-VIII.0",
            Self::EndfB8_1 => "ENDF-B-VIII.1",
            Self::Jeff3_3 => "JEFF-3.3",
            Self::Jendl5 => "JENDL-5",
            Self::Tendl2023 => "TENDL-2023",
            Self::Cendl3_2 => "CENDL-3.2",
        }
    }

    /// Construct the ZIP filename for a given isotope.
    ///
    /// IAEA uses two naming conventions (MAT always 4-digit zero-padded):
    /// - VIII.0, JEFF-3.3: MAT-first `n_{mat:04}_{z}-{Sym}-{a}.zip` (Z unpadded)
    /// - VIII.1, JENDL-5, TENDL-2023, CENDL-3.2: Z-first
    ///   `n_{z:03}-{Sym}-{a}_{mat:04}.zip` (Z 3-digit; free neutron uses `nn`)
    fn zip_filename(&self, isotope: &Isotope, mat: u32) -> String {
        let sym = elements::element_symbol(isotope.z()).unwrap_or("X");
        let z = isotope.z();
        let a = isotope.a();
        match self {
            Self::EndfB8_0 | Self::Jeff3_3 => {
                format!("n_{mat:04}_{z}-{sym}-{a}.zip")
            }
            Self::EndfB8_1 | Self::Jendl5 | Self::Tendl2023 | Self::Cendl3_2 => {
                let zip_sym = if z == 0 && a == 1 { "nn" } else { sym };
                format!("n_{z:03}-{zip_sym}-{a}_{mat:04}.zip")
            }
        }
    }
}

/// Compute the default on-disk cache directory for a given library without
/// constructing an [`EndfRetriever`].
///
/// The retriever's constructor builds a `reqwest` blocking client + TLS
/// configuration, which is wasted work when all the caller needs is the cache
/// path for a UI hint or manual drop instruction. Mirrors the path layout
/// that [`EndfRetriever::new`] would resolve to.
pub fn default_cache_dir(library: EndfLibrary) -> PathBuf {
    default_cache_root().join(library.cache_dir_name())
}

/// Compute the default cache file path for an isotope without constructing
/// an [`EndfRetriever`]. Same layout as [`EndfRetriever::cache_file_path`].
pub fn default_cache_file_path(isotope: &Isotope, library: EndfLibrary) -> PathBuf {
    let sym = elements::element_symbol(isotope.z()).unwrap_or("X");
    default_cache_dir(library).join(format!("{}-{}.endf", sym, isotope.a()))
}

fn default_cache_root() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("nereids")
        .join("endf")
}

/// ENDF file retrieval manager with local caching.
pub struct EndfRetriever {
    /// Root cache directory.
    cache_root: PathBuf,
    /// Base URL for IAEA downloads used by libraries that are not mirrored by
    /// NNDC as raw ENDF-6 files.
    base_url: String,
    /// Shared HTTP client with explicit connect/total timeouts so a transport
    /// stall surfaces as a clear error instead of hanging the GUI worker.
    client: reqwest::blocking::Client,
}

impl EndfRetriever {
    /// Create a new retriever with default cache location (~/.cache/nereids/endf/).
    pub fn new() -> Self {
        Self {
            cache_root: default_cache_root(),
            base_url: IAEA_BASE_URL.to_string(),
            client: build_http_client(),
        }
    }

    /// Create a retriever with a custom cache directory.
    pub fn with_cache_dir(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache_root: cache_dir.into(),
            base_url: IAEA_BASE_URL.to_string(),
            client: build_http_client(),
        }
    }

    /// Get the cache directory for a specific library.
    ///
    /// Public so the GUI can show users exactly where to drop a manually-
    /// downloaded ENDF file when a fetch fails (issue #523).
    pub fn cache_dir(&self, library: EndfLibrary) -> PathBuf {
        self.cache_root.join(library.cache_dir_name())
    }

    /// Get the cached ENDF file path for an isotope.
    ///
    /// Public so callers can present the exact target path for manual file
    /// drops; see [`Self::install_local_endf`] for the programmatic equivalent.
    pub fn cache_file_path(&self, isotope: &Isotope, library: EndfLibrary) -> PathBuf {
        let sym = elements::element_symbol(isotope.z()).unwrap_or("X");
        let filename = format!("{}-{}.endf", sym, isotope.a());
        self.cache_dir(library).join(filename)
    }

    /// Retrieve the ENDF file for an isotope, using cache if available.
    ///
    /// Returns the path to the cached ENDF file and its contents as a string.
    ///
    /// # Arguments
    /// * `isotope` — The isotope to retrieve data for.
    /// * `library` — The ENDF library to use.
    /// * `mat` — The ENDF MAT (material) number.
    pub fn get_endf_file(
        &self,
        isotope: &Isotope,
        library: EndfLibrary,
        mat: u32,
    ) -> Result<(PathBuf, String), EndfRetrievalError> {
        let cache_path = self.cache_file_path(isotope, library);

        // Check cache first.
        if cache_path.exists() {
            let contents = fs::read_to_string(&cache_path)?;
            return Ok((cache_path, contents));
        }

        // Download from the remote source.
        let contents = self.download_endf(isotope, library, mat)?;

        // Cache the file.
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&cache_path, &contents)?;

        Ok((cache_path, contents))
    }

    /// Download an ENDF file from NNDC raw files or IAEA ZIP archives.
    fn download_endf(
        &self,
        isotope: &Isotope,
        library: EndfLibrary,
        mat: u32,
    ) -> Result<String, EndfRetrievalError> {
        let nndc_url = nndc_endf_url(isotope, library);
        let nndc_already_tried = if let Some(primary_url) = &nndc_url {
            if let Ok(text) = self.fetch_text(primary_url, false) {
                return Ok(text);
            }
            true
        } else {
            false
        };

        let zip_filename = library.zip_filename(isotope, mat);
        let url = format!("{}/{}/{}", self.base_url, library.url_path(), zip_filename);
        let iaea_result = self.fetch_bytes(&url, true);
        match iaea_result {
            Ok(bytes) => extract_endf_from_zip(&bytes),
            Err(err) if should_try_nndc_fallback(&err) => {
                if !nndc_already_tried
                    && let Some(fallback_url) = &nndc_url
                    && let Ok(text) = self.fetch_text(fallback_url, false)
                {
                    return Ok(text);
                }
                Err(err.into_retrieval_error(isotope, library))
            }
            Err(err) => Err(err.into_retrieval_error(isotope, library)),
        }
    }

    fn fetch_bytes(&self, url: &str, pace_iaea: bool) -> Result<Vec<u8>, DownloadError> {
        if pace_iaea {
            wait_for_iaea_slot();
        }
        let response = self
            .client
            .get(url)
            .send()
            .map_err(|e| DownloadError::Transport {
                url: url.to_string(),
                message: format_error_chain(&e),
            })?;

        let status = response.status();
        if !status.is_success() {
            return Err(DownloadError::Http {
                url: url.to_string(),
                status,
                cloudflare_challenge: has_cloudflare_challenge(&response),
            });
        }

        response
            .bytes()
            .map(|bytes| bytes.to_vec())
            .map_err(|e| DownloadError::Transport {
                url: url.to_string(),
                message: format!("Failed to read response body: {}", format_error_chain(&e)),
            })
    }

    fn fetch_text(&self, url: &str, pace_iaea: bool) -> Result<String, EndfRetrievalError> {
        let bytes = self
            .fetch_bytes(url, pace_iaea)
            .map_err(|err| err.into_retrieval_error_for_url())?;
        String::from_utf8(bytes)
            .map_err(|e| EndfRetrievalError::Parse(format!("Invalid UTF-8 ENDF response: {e}")))
    }

    /// Load an ENDF file from a local path (no download).
    pub fn load_local(path: &Path) -> Result<String, EndfRetrievalError> {
        fs::read_to_string(path).map_err(EndfRetrievalError::from)
    }

    /// Peek a user-supplied ENDF source: decode the body and parse the HEAD
    /// record so the caller can route the upload to the correct isotope entry.
    ///
    /// Accepts the same input forms as [`Self::install_local_endf`] — a raw
    /// ENDF text file or the IAEA ZIP archive distribution — and returns the
    /// isotope declared by the file's MF=2 MT=151 HEAD record alongside the
    /// decoded text. The GUI uses this to dispatch a manual upload to the
    /// matching `IsotopeEntry` without re-reading or re-extracting the file
    /// during install (issue #523, P2: avoid N-pass zip extraction).
    pub fn peek_local_endf(source: &Path) -> Result<(Isotope, String), EndfRetrievalError> {
        let raw = fs::read(source)?;
        let endf_text = if looks_like_zip(source, &raw) {
            extract_endf_from_zip(&raw)?
        } else {
            String::from_utf8(raw).map_err(|e| {
                EndfRetrievalError::Parse(format!("ENDF file is not valid UTF-8: {e}"))
            })?
        };
        let parsed = crate::parser::parse_endf_file2(&endf_text)
            .map_err(|e| EndfRetrievalError::Parse(format!("Failed to parse ENDF file: {e}")))?;
        Ok((parsed.isotope, endf_text))
    }

    /// Write already-validated ENDF text to the canonical cache slot for
    /// `isotope`/`library`. Returns the cache file path.
    ///
    /// Use this when the caller has already obtained `text` via
    /// [`Self::peek_local_endf`] and confirmed the isotope. For one-shot
    /// install-from-path, use [`Self::install_local_endf`] instead.
    pub fn install_endf_text(
        &self,
        isotope: &Isotope,
        library: EndfLibrary,
        text: &str,
    ) -> Result<PathBuf, EndfRetrievalError> {
        let cache_path = self.cache_file_path(isotope, library);
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&cache_path, text)?;
        Ok(cache_path)
    }

    /// Install a user-supplied ENDF file into the cache for `isotope`/`library`.
    ///
    /// Accepts either a raw ENDF text file (`.endf`, `.dat`, `.txt`, or
    /// extensionless) or the IAEA ZIP archive distribution (`n_…_….zip`).
    /// The file is parsed to verify that its declared isotope matches
    /// `isotope`; on success the text is written to the canonical cache slot
    /// returned by [`Self::cache_file_path`] and `(cache_path, text)` is
    /// returned. Subsequent calls to [`Self::get_endf_file`] will then hit the
    /// cache without any network access.
    ///
    /// This is the GUI's manual-upload escape hatch for users on networks
    /// where IAEA/NNDC is blocked (issue #523).
    pub fn install_local_endf(
        &self,
        isotope: &Isotope,
        library: EndfLibrary,
        source: &Path,
    ) -> Result<(PathBuf, String), EndfRetrievalError> {
        let (found, endf_text) = Self::peek_local_endf(source)?;
        if found != *isotope {
            return Err(EndfRetrievalError::IsotopeMismatch {
                expected: isotope_label(isotope),
                found: isotope_label(&found),
            });
        }
        let cache_path = self.install_endf_text(isotope, library, &endf_text)?;
        Ok((cache_path, endf_text))
    }

    /// Clear the cache for a specific library, or all if `None`.
    pub fn clear_cache(&self, library: Option<EndfLibrary>) -> Result<(), EndfRetrievalError> {
        match library {
            Some(lib) => {
                let dir = self.cache_dir(lib);
                if dir.exists() {
                    fs::remove_dir_all(&dir)?;
                }
            }
            None => {
                if self.cache_root.exists() {
                    fs::remove_dir_all(&self.cache_root)?;
                }
            }
        }
        Ok(())
    }
}

impl Default for EndfRetriever {
    fn default() -> Self {
        Self::new()
    }
}

/// Look up the ENDF MAT number for a ground-state isotope, library-aware.
///
/// Dispatches to the underlying `endf-mat` table for the requested library:
/// - `Tendl2023`: ~2,300 ground-state isotopes from the TENDL-2023 neutrons sublibrary.
/// - `Cendl3_2`: 258 isotopes plus free neutron from the CENDL-3.2 neutrons sublibrary (no Br entries).
/// - All other variants: 535 isotopes from the ENDF/B-VIII.0 neutrons sublibrary
///   (the MAT numbers in ENDF/B-VIII.1, JEFF-3.3, and JENDL-5 are identical to
///   ENDF/B-VIII.0 for the isotopes they share).
///
/// MAT numbers are *almost* universal across libraries; the one documented exception
/// is Es-255, which is MAT 9916 in ENDF/B-VIII.0 and MAT 9915 in TENDL-2023. CENDL-3.2
/// has no MAT divergences from ENDF/B-VIII.0 for shared isotopes. The library-aware
/// lookup ensures the correct MAT is used to construct retrieval URLs.
pub fn mat_number(isotope: &Isotope, library: EndfLibrary) -> Option<u32> {
    match library {
        EndfLibrary::Tendl2023 => endf_mat::mat_number_tendl(isotope.z(), isotope.a()),
        EndfLibrary::Cendl3_2 => endf_mat::mat_number_cendl(isotope.z(), isotope.a()),
        _ => endf_mat::mat_number(isotope.z(), isotope.a()),
    }
}

/// All mass numbers with an evaluation for element Z in the given library.
///
/// Library-aware counterpart to [`endf_mat::known_isotopes`] (which is
/// ENDF/B-VIII.0-only) — must be used wherever the GUI surfaces the set of
/// selectable isotopes for the *currently selected* library, otherwise
/// TENDL-2023-only isotopes (e.g. Fm-247) will be silently hidden, and Br
/// will be incorrectly shown as available under CENDL-3.2.
pub fn known_isotopes_for(z: u32, library: EndfLibrary) -> Vec<u32> {
    match library {
        EndfLibrary::Tendl2023 => endf_mat::known_isotopes_tendl(z),
        EndfLibrary::Cendl3_2 => endf_mat::known_isotopes_cendl(z),
        _ => endf_mat::known_isotopes(z),
    }
}

/// Whether the given library has an evaluation for `(Z, A)`.
///
/// Library-aware counterpart to [`endf_mat::has_endf_evaluation`] — must be
/// used by GUI availability indicators that depend on the *currently
/// selected* library.
pub fn has_endf_evaluation_for(z: u32, a: u32, library: EndfLibrary) -> bool {
    match library {
        EndfLibrary::Tendl2023 => endf_mat::has_endf_evaluation_tendl(z, a),
        EndfLibrary::Cendl3_2 => endf_mat::has_endf_evaluation_cendl(z, a),
        _ => endf_mat::has_endf_evaluation(z, a),
    }
}

/// Errors from ENDF retrieval operations.
#[derive(Debug, thiserror::Error)]
pub enum EndfRetrievalError {
    /// Transport-level failure (connection refused, DNS error, non-404 HTTP error, etc.).
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Upstream server actively blocked automated retrieval.
    #[error("Remote access blocked: HTTP {status} for {url}. {message}")]
    RemoteAccessBlocked {
        status: u16,
        url: String,
        message: String,
    },

    /// The isotope exists in ENDF/B-VIII.0 but is not available in the requested library.
    #[error("{isotope} is not available in the {library} library")]
    NotInLibrary { isotope: String, library: String },

    /// A user-supplied ENDF file did not describe the isotope it was being
    /// installed against (issue #523, manual upload path).
    #[error("ENDF file is for {found}, but expected {expected}")]
    IsotopeMismatch { expected: String, found: String },

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Isotope not found in MAT database: {0}")]
    UnknownIsotope(String),
}

impl EndfRetrievalError {
    /// Whether this error means the upstream server is denying automated access.
    pub fn is_remote_access_blocked(&self) -> bool {
        matches!(self, Self::RemoteAccessBlocked { .. })
    }
}

#[derive(Debug)]
enum DownloadError {
    Http {
        url: String,
        status: reqwest::StatusCode,
        cloudflare_challenge: bool,
    },
    Transport {
        url: String,
        message: String,
    },
}

impl DownloadError {
    fn into_retrieval_error(self, isotope: &Isotope, library: EndfLibrary) -> EndfRetrievalError {
        match self {
            Self::Http { status, .. } if status == reqwest::StatusCode::NOT_FOUND => {
                EndfRetrievalError::NotInLibrary {
                    isotope: isotope_label(isotope),
                    library: library.cache_dir_name().to_string(),
                }
            }
            other => other.into_retrieval_error_for_url(),
        }
    }

    fn into_retrieval_error_for_url(self) -> EndfRetrievalError {
        match self {
            Self::Http {
                url,
                status,
                cloudflare_challenge,
            } if is_access_block_status(status) => EndfRetrievalError::RemoteAccessBlocked {
                status: status.as_u16(),
                url,
                message: if cloudflare_challenge {
                    "The server returned a Cloudflare managed challenge; stop batch fetches and retry later from a normal browser/network."
                        .to_string()
                } else {
                    "The upstream server denied automated access; stop batch fetches and retry later."
                        .to_string()
                },
            },
            Self::Http { url, status, .. } => {
                EndfRetrievalError::NetworkError(format!("HTTP {status} for {url}"))
            }
            Self::Transport { url, message } => {
                EndfRetrievalError::NetworkError(format!("Failed to fetch {url}: {message}"))
            }
        }
    }
}

/// Extract the ENDF data file body from a ZIP archive.
///
/// Prefers archive entries ending in `.endf`, `.dat`, or `.txt`; falls back to
/// the first entry if none match. IAEA distributes one ENDF per zip, so the
/// fallback effectively only fires on hand-rolled archives.
fn extract_endf_from_zip(zip_bytes: &[u8]) -> Result<String, EndfRetrievalError> {
    let cursor = std::io::Cursor::new(zip_bytes);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| EndfRetrievalError::Parse(format!("Invalid ZIP archive: {}", e)))?;

    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| EndfRetrievalError::Parse(format!("Failed to read ZIP entry: {}", e)))?;
        let name = file.name().to_lowercase();
        if name.ends_with(".endf") || name.ends_with(".dat") || name.ends_with(".txt") {
            let mut contents = String::new();
            file.read_to_string(&mut contents).map_err(|e| {
                EndfRetrievalError::Parse(format!("Failed to read ENDF content: {}", e))
            })?;
            return Ok(contents);
        }
    }

    if !archive.is_empty() {
        let mut file = archive
            .by_index(0)
            .map_err(|e| EndfRetrievalError::Parse(format!("Failed to read ZIP entry: {}", e)))?;
        let mut contents = String::new();
        file.read_to_string(&mut contents).map_err(|e| {
            EndfRetrievalError::Parse(format!("Failed to read ENDF content: {}", e))
        })?;
        return Ok(contents);
    }

    Err(EndfRetrievalError::Parse(
        "No ENDF data file found in ZIP archive".to_string(),
    ))
}

/// Detect a ZIP archive by extension or PK magic bytes.
///
/// IAEA-distributed `n_…_….zip` files always start with `PK\x03\x04`. The
/// magic-byte check covers the case where a user has stripped or renamed the
/// extension before uploading.
fn looks_like_zip(source: &Path, raw: &[u8]) -> bool {
    let by_ext = source
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("zip"));
    let by_magic = raw.len() >= 4 && &raw[..4] == b"PK\x03\x04";
    by_ext || by_magic
}

/// Build the shared HTTP client used for ENDF downloads.
///
/// Connect timeout is short so DNS/TLS failures surface fast; total timeout
/// is generous because some library zips are several hundred KB over slow
/// links. ENDF zip files top out around ~1 MB.
fn build_http_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .user_agent(ENDF_USER_AGENT)
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(60))
        .build()
        .expect("failed to build reqwest blocking client")
}

fn wait_for_iaea_slot() {
    let mut last_request = LAST_IAEA_REQUEST
        .get_or_init(|| Mutex::new(None))
        .lock()
        .expect("IAEA request throttle mutex poisoned");
    if let Some(last) = *last_request {
        let elapsed = last.elapsed();
        if elapsed < IAEA_MIN_REQUEST_INTERVAL {
            std::thread::sleep(IAEA_MIN_REQUEST_INTERVAL - elapsed);
        }
    }
    *last_request = Some(Instant::now());
}

fn nndc_endf_url(isotope: &Isotope, library: EndfLibrary) -> Option<String> {
    let version = match library {
        EndfLibrary::EndfB8_0 => "ENDF-B-VIII.0",
        EndfLibrary::EndfB8_1 => "ENDF-B-VIII.1",
        _ => return None,
    };
    let sym = elements::element_symbol(isotope.z())?;
    Some(format!(
        "{NNDC_ENDF_BASE_URL}/{version}/n-{z:03}_{sym}_{a}.endf",
        z = isotope.z(),
        a = isotope.a()
    ))
}

fn should_try_nndc_fallback(err: &DownloadError) -> bool {
    match err {
        DownloadError::Http { status, .. } => {
            *status == reqwest::StatusCode::NOT_FOUND || is_access_block_status(*status)
        }
        DownloadError::Transport { .. } => true,
    }
}

fn is_access_block_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::FORBIDDEN
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
}

fn has_cloudflare_challenge(response: &reqwest::blocking::Response) -> bool {
    response
        .headers()
        .get("cf-mitigated")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.eq_ignore_ascii_case("challenge"))
}

fn isotope_label(isotope: &Isotope) -> String {
    format!(
        "{}-{}",
        nereids_core::elements::element_symbol(isotope.z()).unwrap_or("?"),
        isotope.a()
    )
}

/// Render an error and its full `source()` chain on one line. reqwest's outer
/// `Display` is uninformative ("error sending request for url ...") — the
/// real cause (TLS, DNS, refused, timeout) lives in the source chain.
fn format_error_chain(err: &dyn std::error::Error) -> String {
    let mut out = err.to_string();
    let mut cur = err.source();
    while let Some(s) = cur {
        out.push_str(": ");
        out.push_str(&s.to_string());
        cur = s.source();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cendl_neutron_uses_upstream_nn_filename() {
        let neutron = Isotope::new(0, 1).unwrap();
        assert_eq!(
            EndfLibrary::Cendl3_2.zip_filename(&neutron, 25),
            "n_000-nn-1_0025.zip"
        );
    }

    #[test]
    fn cendl_neutron_has_library_aware_mat_lookup() {
        let neutron = Isotope::new(0, 1).unwrap();
        assert_eq!(mat_number(&neutron, EndfLibrary::Cendl3_2), Some(25));
    }

    #[test]
    fn nndc_fallback_url_uses_raw_endf_naming() {
        let ba138 = Isotope::new(56, 138).unwrap();
        assert_eq!(
            nndc_endf_url(&ba138, EndfLibrary::EndfB8_1).as_deref(),
            Some("https://www.nndc.bnl.gov/endf-data/ENDF/ENDF-B-VIII.1/n-056_Ba_138.endf")
        );
        assert!(nndc_endf_url(&ba138, EndfLibrary::Cendl3_2).is_none());
    }

    #[test]
    fn remote_access_blocked_is_identifiable() {
        let err = EndfRetrievalError::RemoteAccessBlocked {
            status: 403,
            url: "https://example.test/file.zip".into(),
            message: "blocked".into(),
        };
        assert!(err.is_remote_access_blocked());
    }

    /// Issue #523: the polite User-Agent must carry the live package version
    /// and the contact metadata the IAEA acceptance criterion calls for.
    #[test]
    fn endf_user_agent_contains_version_and_contact() {
        assert!(
            ENDF_USER_AGENT.starts_with("NEREIDS/"),
            "UA must start with NEREIDS/, got {ENDF_USER_AGENT:?}"
        );
        assert!(
            ENDF_USER_AGENT.contains(env!("CARGO_PKG_VERSION")),
            "UA must carry CARGO_PKG_VERSION, got {ENDF_USER_AGENT:?}"
        );
        assert!(
            ENDF_USER_AGENT.contains("github.com/ornlneutronimaging/NEREIDS"),
            "UA must include the project URL, got {ENDF_USER_AGENT:?}"
        );
        assert!(
            ENDF_USER_AGENT.contains("contact: zhangc@ornl.gov"),
            "UA must include a contact mailbox, got {ENDF_USER_AGENT:?}"
        );
    }

    // Minimal valid ENDF MF=2/MT=151 fixture for W-184 — same shape as the
    // krm3 fixture in parser.rs tests, kept inline so retrieval tests stay
    // self-contained.
    const W184_FIXTURE: &str = concat!(
        " 7.418400+4 1.820000+2          0          0          1          07437 2151    1\n",
        " 7.418400+4 1.000000+0          0          0          1          07437 2151    2\n",
        " 1.000000-5 1.000000+3          1          7          0          07437 2151    3\n",
        " 0.000000+0 7.000000-1          0          3          1          07437 2151    4\n",
        " 0.000000+0 0.000000+0          1          0         12          17437 2151    5\n",
        " 1.000000+0 1.820000+2 0.000000+0 0.000000+0 5.000000-1 0.000000+07437 2151    6\n",
        " 0.000000+0 1.000000+0 0.000000+0 2.000000+0 1.000000+0 1.000000+07437 2151    7\n",
        " 5.000000-1 0.000000+0          0          0         12          27437 2151    8\n",
        " 0.000000+0 0.000000+0 0.000000+0 0.000000+0 0.000000+0 0.000000+07437 2151    9\n",
        " 1.000000+0 0.000000+0 5.000000-1 0.000000+0 7.000000-1 7.000000-17437 2151   10\n",
        " 0.000000+0 0.000000+0          0          0         12          27437 2151   11\n",
        " 1.000000+1 2.500000-2 1.000000-3 0.000000+0 0.000000+0 0.000000+07437 2151   12\n",
        " 2.000000+1 3.000000-2 2.000000-3 0.000000+0 0.000000+0 0.000000+07437 2151   13\n",
    );

    fn write_zip_with_endf(zip_path: &Path, inner_name: &str, body: &str) -> std::io::Result<()> {
        let file = fs::File::create(zip_path)?;
        let mut zw = zip::ZipWriter::new(file);
        zw.start_file::<_, ()>(inner_name, zip::write::SimpleFileOptions::default())
            .map_err(|e| std::io::Error::other(format!("zip start_file: {e}")))?;
        std::io::Write::write_all(&mut zw, body.as_bytes())?;
        zw.finish()
            .map_err(|e| std::io::Error::other(format!("zip finish: {e}")))?;
        Ok(())
    }

    #[test]
    fn install_local_endf_accepts_raw_endf_with_matching_isotope() {
        let cache = tempfile::tempdir().expect("tempdir");
        let src = tempfile::NamedTempFile::new().expect("src file");
        std::fs::write(src.path(), W184_FIXTURE).unwrap();

        let retriever = EndfRetriever::with_cache_dir(cache.path());
        let w184 = Isotope::new(74, 184).unwrap();
        let (cache_path, text) = retriever
            .install_local_endf(&w184, EndfLibrary::EndfB8_0, src.path())
            .expect("install must succeed");

        assert!(cache_path.exists(), "cache file must be written");
        assert!(text.contains("7.418400+4"), "returned text matches input");
        // Subsequent get_endf_file calls must read from cache (no network).
        let (cached_path, cached_text) = retriever
            .get_endf_file(&w184, EndfLibrary::EndfB8_0, 7437)
            .expect("cache hit must succeed offline");
        assert_eq!(cached_path, cache_path);
        assert_eq!(cached_text, text);
    }

    #[test]
    fn install_local_endf_accepts_zip_archive() {
        let cache = tempfile::tempdir().expect("tempdir");
        let zip_path = cache.path().join("upload.zip");
        write_zip_with_endf(&zip_path, "n_7437_74-W-184.endf", W184_FIXTURE)
            .expect("write zip fixture");

        let retriever = EndfRetriever::with_cache_dir(cache.path());
        let w184 = Isotope::new(74, 184).unwrap();
        let (cache_path, _) = retriever
            .install_local_endf(&w184, EndfLibrary::EndfB8_0, &zip_path)
            .expect("zip install must succeed");
        assert!(cache_path.exists());
        assert_eq!(
            fs::read_to_string(&cache_path).unwrap(),
            W184_FIXTURE,
            "cache must hold the extracted body, not the zip"
        );
    }

    /// `default_cache_dir` / `default_cache_file_path` must agree with the
    /// retriever's instance methods, otherwise UI hints would point users to
    /// the wrong location after a fetch failure.
    #[test]
    fn default_cache_paths_agree_with_retriever_instance() {
        let retriever = EndfRetriever::new();
        let w184 = Isotope::new(74, 184).unwrap();
        for lib in [
            EndfLibrary::EndfB8_0,
            EndfLibrary::EndfB8_1,
            EndfLibrary::Jeff3_3,
            EndfLibrary::Jendl5,
            EndfLibrary::Tendl2023,
            EndfLibrary::Cendl3_2,
        ] {
            assert_eq!(default_cache_dir(lib), retriever.cache_dir(lib));
            assert_eq!(
                default_cache_file_path(&w184, lib),
                retriever.cache_file_path(&w184, lib)
            );
        }
    }

    #[test]
    fn peek_local_endf_extracts_isotope_from_zip_and_raw() {
        let dir = tempfile::tempdir().expect("tempdir");

        // Raw .endf path
        let raw_path = dir.path().join("W-184.endf");
        std::fs::write(&raw_path, W184_FIXTURE).unwrap();
        let (iso, text) = EndfRetriever::peek_local_endf(&raw_path).expect("raw peek");
        assert_eq!(iso, Isotope::new(74, 184).unwrap());
        assert_eq!(text, W184_FIXTURE);

        // Zip path
        let zip_path = dir.path().join("upload.zip");
        write_zip_with_endf(&zip_path, "inner.endf", W184_FIXTURE).expect("zip");
        let (iso_z, text_z) = EndfRetriever::peek_local_endf(&zip_path).expect("zip peek");
        assert_eq!(iso_z, Isotope::new(74, 184).unwrap());
        assert_eq!(text_z, W184_FIXTURE);
    }

    /// Regression for the `looks_like_zip` magic-byte branch: an upload whose
    /// path has no `.zip` extension must still be detected via `PK\x03\x04`
    /// and routed through the zip extractor. Without this test, the magic-byte
    /// branch could silently regress to extension-only detection.
    #[test]
    fn install_local_endf_accepts_extensionless_zip_via_magic_bytes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let zip_path = dir.path().join("upload-no-extension"); // no .zip suffix
        write_zip_with_endf(&zip_path, "inner.endf", W184_FIXTURE).expect("write zip");

        // Sanity-check we really did create a zip without the .zip extension.
        assert!(zip_path.extension().is_none());
        let head = std::fs::read(&zip_path).unwrap();
        assert_eq!(&head[..4], b"PK\x03\x04");

        let retriever = EndfRetriever::with_cache_dir(dir.path().join("cache"));
        let w184 = Isotope::new(74, 184).unwrap();
        let (cache_path, text) = retriever
            .install_local_endf(&w184, EndfLibrary::EndfB8_0, &zip_path)
            .expect("extensionless zip must still install");
        assert!(cache_path.exists());
        assert_eq!(text, W184_FIXTURE);

        // peek_local_endf must also handle the extensionless zip.
        let (iso, peeked_text) =
            EndfRetriever::peek_local_endf(&zip_path).expect("peek must also work");
        assert_eq!(iso, w184);
        assert_eq!(peeked_text, W184_FIXTURE);
    }

    #[test]
    fn install_local_endf_rejects_isotope_mismatch() {
        let cache = tempfile::tempdir().expect("tempdir");
        let src = tempfile::NamedTempFile::new().expect("src file");
        std::fs::write(src.path(), W184_FIXTURE).unwrap();

        let retriever = EndfRetriever::with_cache_dir(cache.path());
        let hf180 = Isotope::new(72, 180).unwrap();
        let err = retriever
            .install_local_endf(&hf180, EndfLibrary::EndfB8_0, src.path())
            .expect_err("must reject ZA mismatch");

        match err {
            EndfRetrievalError::IsotopeMismatch { expected, found } => {
                assert!(expected.contains("Hf"), "expected label, got {expected}");
                assert!(found.contains("W"), "found label, got {found}");
            }
            other => panic!("expected IsotopeMismatch, got {other:?}"),
        }
        assert!(
            !retriever
                .cache_file_path(&hf180, EndfLibrary::EndfB8_0)
                .exists(),
            "no cache file must be written on mismatch"
        );
    }
}
