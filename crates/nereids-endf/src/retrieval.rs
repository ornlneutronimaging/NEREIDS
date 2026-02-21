//! ENDF file download and local caching.
//!
//! Downloads ENDF files from the IAEA Nuclear Data Services and caches them
//! locally for offline use. Follows the URL patterns established by PLEIADES.
//!
//! ## PLEIADES Reference
//! - `pleiades/nuclear/manager.py` — URL construction, cache directory layout
//! - `pleiades/nuclear/models.py` — library enum, filename patterns

use nereids_core::elements;
use nereids_core::types::Isotope;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

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
}

impl EndfLibrary {
    /// URL path component for this library.
    fn url_path(&self) -> &'static str {
        match self {
            Self::EndfB8_0 => "ENDF-B-VIII.0/n",
            Self::EndfB8_1 => "ENDF-B-VIII.1/n",
            Self::Jeff3_3 => "JEFF-3.3/n",
            Self::Jendl5 => "JENDL-5/n",
        }
    }

    /// Cache directory name.
    fn cache_dir_name(&self) -> &'static str {
        match self {
            Self::EndfB8_0 => "ENDF-B-VIII.0",
            Self::EndfB8_1 => "ENDF-B-VIII.1",
            Self::Jeff3_3 => "JEFF-3.3",
            Self::Jendl5 => "JENDL-5",
        }
    }

    /// Construct the ZIP filename for a given isotope.
    ///
    /// IAEA uses two naming conventions (MAT always 4-digit zero-padded):
    /// - VIII.0, JEFF-3.3: MAT-first `n_{mat:04}_{z}-{Sym}-{a}.zip` (Z unpadded)
    /// - VIII.1, JENDL-5: Z-first   `n_{z:03}-{Sym}-{a}_{mat:04}.zip` (Z 3-digit)
    fn zip_filename(&self, isotope: &Isotope, mat: u32) -> String {
        let sym = elements::element_symbol(isotope.z).unwrap_or("X");
        let z = isotope.z;
        let a = isotope.a;
        match self {
            Self::EndfB8_0 | Self::Jeff3_3 => {
                format!("n_{mat:04}_{z}-{sym}-{a}.zip")
            }
            Self::EndfB8_1 | Self::Jendl5 => {
                format!("n_{z:03}-{sym}-{a}_{mat:04}.zip")
            }
        }
    }
}

/// ENDF file retrieval manager with local caching.
pub struct EndfRetriever {
    /// Root cache directory.
    cache_root: PathBuf,
    /// Base URL for IAEA downloads.
    base_url: String,
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
            base_url: "https://www-nds.iaea.org/public/download-endf".to_string(),
        }
    }

    /// Create a retriever with a custom cache directory.
    pub fn with_cache_dir(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache_root: cache_dir.into(),
            base_url: "https://www-nds.iaea.org/public/download-endf".to_string(),
        }
    }

    /// Get the cache directory for a specific library.
    fn cache_dir(&self, library: EndfLibrary) -> PathBuf {
        self.cache_root.join(library.cache_dir_name())
    }

    /// Get the cached ENDF file path for an isotope.
    fn cache_file_path(&self, isotope: &Isotope, library: EndfLibrary) -> PathBuf {
        let sym = elements::element_symbol(isotope.z).unwrap_or("X");
        let filename = format!("{}-{}.endf", sym, isotope.a);
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

        // Download from IAEA.
        let contents = self.download_endf(isotope, library, mat)?;

        // Cache the file.
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&cache_path, &contents)?;

        Ok((cache_path, contents))
    }

    /// Download ENDF file from IAEA and extract from ZIP archive.
    fn download_endf(
        &self,
        isotope: &Isotope,
        library: EndfLibrary,
        mat: u32,
    ) -> Result<String, EndfRetrievalError> {
        let zip_filename = library.zip_filename(isotope, mat);
        let url = format!("{}/{}/{}", self.base_url, library.url_path(), zip_filename);

        let response = reqwest::blocking::get(&url).map_err(|e| {
            EndfRetrievalError::Download(format!("Failed to download {}: {}", url, e))
        })?;

        if !response.status().is_success() {
            return Err(EndfRetrievalError::Download(format!(
                "HTTP {} for {}",
                response.status(),
                url
            )));
        }

        let bytes = response.bytes().map_err(|e| {
            EndfRetrievalError::Download(format!("Failed to read response body: {}", e))
        })?;

        // Extract ENDF file from ZIP archive.
        self.extract_endf_from_zip(&bytes)
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

/// Look up the ENDF MAT number for a ground-state isotope.
///
/// Covers 535 isotopes from the ENDF/B-VIII.0 neutrons sublibrary
/// (via the `endf-mat` crate). This replaces the previous hand-coded
/// 47-entry table and fixes several incorrect MAT values (Cd-113,
/// Hf-177/178, W-182/183/184/186 were off by one isotope offset).
pub fn mat_number(isotope: &Isotope) -> Option<u32> {
    endf_mat::mat_number(isotope.z, isotope.a)
}

/// Errors from ENDF retrieval operations.
#[derive(Debug, thiserror::Error)]
pub enum EndfRetrievalError {
    #[error("Download failed: {0}")]
    Download(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Isotope not found in MAT database: {0}")]
    UnknownIsotope(String),
}
