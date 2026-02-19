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
    /// ENDF/B-VIII.0 uses MAT-first format: `n_{mat}_{z}-{Element}-{a}.zip`
    /// ENDF/B-VIII.1 and others use element-first: `n_{z}-{Element}-{a}_{mat}.zip`
    fn zip_filename(&self, isotope: &Isotope, mat: u32) -> String {
        let sym = elements::element_symbol(isotope.z).unwrap_or("X");
        let z = isotope.z;
        let a = isotope.a;
        match self {
            Self::EndfB8_0 => format!("n_{mat}_{z}-{sym}-{a}.zip"),
            _ => format!("n_{z}-{sym}-{a}_{mat}.zip"),
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

/// Well-known ENDF MAT numbers for commonly used isotopes.
///
/// Reference: ENDF-6 Formats Manual, Appendix B.
pub fn mat_number(isotope: &Isotope) -> Option<u32> {
    // MAT numbers follow: MAT = Z*100 + iso_index
    // These are the standard values from the ENDF database.
    let za = isotope.z * 1000 + isotope.a;
    match za {
        1001 => Some(125),   // H-1
        1002 => Some(128),   // H-2
        3006 => Some(325),   // Li-6
        3007 => Some(328),   // Li-7
        5010 => Some(525),   // B-10
        5011 => Some(528),   // B-11
        6012 => Some(625),   // C-12
        13027 => Some(1325), // Al-27
        26054 => Some(2625), // Fe-54
        26056 => Some(2631), // Fe-56
        26057 => Some(2634), // Fe-57
        26058 => Some(2637), // Fe-58
        28058 => Some(2825), // Ni-58
        28060 => Some(2831), // Ni-60
        29063 => Some(2925), // Cu-63
        29065 => Some(2931), // Cu-65
        40090 => Some(4025), // Zr-90
        40091 => Some(4028), // Zr-91
        40092 => Some(4031), // Zr-92
        40094 => Some(4037), // Zr-94
        40096 => Some(4043), // Zr-96
        41093 => Some(4125), // Nb-93
        47107 => Some(4725), // Ag-107
        47109 => Some(4731), // Ag-109
        48113 => Some(4849), // Cd-113
        49115 => Some(4931), // In-115
        72177 => Some(7231), // Hf-177
        72178 => Some(7234), // Hf-178
        73181 => Some(7328), // Ta-181
        74182 => Some(7425), // W-182
        74183 => Some(7428), // W-183
        74184 => Some(7431), // W-184
        74186 => Some(7437), // W-186
        79197 => Some(7925), // Au-197
        82206 => Some(8231), // Pb-206
        82207 => Some(8234), // Pb-207
        82208 => Some(8237), // Pb-208
        90232 => Some(9040), // Th-232
        92234 => Some(9225), // U-234
        92235 => Some(9228), // U-235
        92238 => Some(9237), // U-238
        94239 => Some(9437), // Pu-239
        94240 => Some(9440), // Pu-240
        94241 => Some(9443), // Pu-241
        _ => None,
    }
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
