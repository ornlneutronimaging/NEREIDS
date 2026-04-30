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
const ENDF_USER_AGENT: &str =
    "NEREIDS/0.1.8 (https://github.com/ornlneutronimaging/NEREIDS; ENDF cache)";
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
        let cache_root = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("nereids")
            .join("endf");
        Self {
            cache_root,
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
    fn cache_dir(&self, library: EndfLibrary) -> PathBuf {
        self.cache_root.join(library.cache_dir_name())
    }

    /// Get the cached ENDF file path for an isotope.
    fn cache_file_path(&self, isotope: &Isotope, library: EndfLibrary) -> PathBuf {
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
        if let Some(primary_url) = nndc_endf_url(isotope, library) {
            if let Ok(text) = self.fetch_text(&primary_url, false) {
                return Ok(text);
            }
        }

        let zip_filename = library.zip_filename(isotope, mat);
        let url = format!("{}/{}/{}", self.base_url, library.url_path(), zip_filename);
        let iaea_result = self.fetch_bytes(&url, true);
        match iaea_result {
            Ok(bytes) => return self.extract_endf_from_zip(&bytes),
            Err(err) if should_try_nndc_fallback(&err) => {
                if let Some(fallback_url) = nndc_endf_url(isotope, library) {
                    let text = self.fetch_text(&fallback_url, false)?;
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

    /// Extract the ENDF data file from a ZIP archive.
    ///
    /// Looks for files ending in .endf, .dat, or .txt within the archive.
    fn extract_endf_from_zip(&self, zip_bytes: &[u8]) -> Result<String, EndfRetrievalError> {
        let cursor = std::io::Cursor::new(zip_bytes);
        let mut archive = zip::ZipArchive::new(cursor)
            .map_err(|e| EndfRetrievalError::Parse(format!("Invalid ZIP archive: {}", e)))?;

        // Find the ENDF data file within the archive.
        for i in 0..archive.len() {
            let mut file = archive.by_index(i).map_err(|e| {
                EndfRetrievalError::Parse(format!("Failed to read ZIP entry: {}", e))
            })?;

            let name = file.name().to_lowercase();
            if name.ends_with(".endf") || name.ends_with(".dat") || name.ends_with(".txt") {
                let mut contents = String::new();
                file.read_to_string(&mut contents).map_err(|e| {
                    EndfRetrievalError::Parse(format!("Failed to read ENDF content: {}", e))
                })?;
                return Ok(contents);
            }
        }

        // If no obvious extension, try the first file.
        if !archive.is_empty() {
            let mut file = archive.by_index(0).map_err(|e| {
                EndfRetrievalError::Parse(format!("Failed to read ZIP entry: {}", e))
            })?;
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

    /// Load an ENDF file from a local path (no download).
    pub fn load_local(path: &Path) -> Result<String, EndfRetrievalError> {
        fs::read_to_string(path).map_err(EndfRetrievalError::from)
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
}
