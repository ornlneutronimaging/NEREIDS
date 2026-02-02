# SAMMY teacher map for a modern resonance-imaging library (Rust)

Date: 2026-02-02
Scope: Document code locations in SAMMY that can serve as the "teacher" for a focused, modern rewrite (resonance-imaging first), based on a scan of the local SAMMY repo at `/Users/chenzhang/code.ornl.gov/SAMMY/SAMMY`.

This is **not** a rewrite plan. It is a **reference map**: where the physics and logic live in SAMMY so a student implementation can be built module-by-module and validated against SAMMY outputs.

---

## 1) Minimal physics core (must-have for resonance imaging)

### 1.1 R-matrix (Reich-Moore) cross sections
These files contain the core math needed to compute cross sections from resonance parameters.

- `sammy/src/clm/dopush1.f90`
  - **Compact Reich-Moore implementation**: `SUBROUTINE Csrmat` computes R-matrix cross sections at a single energy.
  - Good "teacher" to port first because the logic is localized and explicit.

- `sammy/src/cro/mcro2.f90`
  - Core R-matrix algebra: construction/inversion of R-matrix, channel handling, external R-matrix terms.

- `sammy/src/cro/CroCrossCalcImpl_M.f90`
  - Bridges the calculator interface to the R-matrix cross section math.

### 1.2 Shift/penetration factors & hard-sphere phase shifts
Required for channel penetrabilities and phase terms.

- `sammy/src/xxx/mxxxb.f90` -- `Facts_NML`: shift factor `S` and penetration `P` (L=0..4).
- `sammy/src/xxx/mxxxa.f90` -- `Facphi_NML`: hard-sphere phase shift (cos/sin of 2phi).

### 1.3 Transmission transform (Beer-Lambert)
SAMMY's transmission conversion logic (including derivatives) lives here.

- `sammy/src/cro/mnrm2.f90` -- `Transm_sum`, `Transm_iso`, and non-uniform thickness variants.

### 1.4 Normalization & background functions
Used heavily in imaging fits.

- `sammy/src/cro/mnrm1.f90` -- `Norm` and `Bgfrpi`.
  - Handles constant background, 1/sqrtE, sqrtE, exponentials, and parameter derivatives.

---

## 2) Broadening and resolution

### 2.1 Doppler broadening
Free-gas broadening and a common interface are here.

- `sammy/src/clm/mclm3.f90` -- Doppler broadening "core" for cross sections.
- `sammy/src/convolution/interface/fortran/DopplerAndResolutionBroadener_M.f90`
  - Common interface for broadening; exposes grid and convolution hooks.

### 2.2 Resolution functions
There are multiple resolution models in SAMMY. Which one VENUS uses is unknown here (classified file not available). Two key families:

- RSL model (delta-L / delta-E / delta-G)
  - `sammy/src/rsl/RslResolutionFunction_M.f90`
  - `sammy/src/rsl/SetupRslRes_M.f90`

- RPI/GEEL/nTOF models
  - `sammy/src/rpi/RpiResolutionFunction_M.f90`
  - `sammy/src/rpi/mrpi*.f90`

Because VENUS uses a user-defined resolution file (classified), the safe approach is to treat the **resolution interface** as a plug-in system and map the model later.

---

## 3) Data model & parameter organization (modernizable C++ layer)
These are useful templates for the Rust data model even if the code is not reused.

- `sammy/src/endf/SammyRMatrixParameters.h`
  - Central model: isotopes -> spin groups -> channels -> resonances.

- `sammy/src/endf/SammySpinGroupInfo.h`
- `sammy/src/endf/SammyResonanceInfo.h`
  - Fit flags and resonance indexing.

- `sammy/src/endf/ResonanceParameterIO.h`
  - Parsing of SAMMY par files; good reference for file-format compatibility if needed.

---

## 4) Cross-section pipeline (0 K -> corrections -> transmission)
The high-level flow for data reconstruction is centralized here.

- `sammy/src/the/CrossSectionCalcDriver_M.f90`
  - Chooses formalism (RM/SLBW/MLBW) and drives cross-section reconstruction.

- `sammy/src/the/ZeroKCrossCorrections_M.f90`
  - 0 K reconstruction, normalization/background, conversion to transmission.

- `sammy/src/the/SumIsoAndConvertToTrans_M.f90`
  - Summation over isotopes and conversion to transmission after broadening.

---

## 5) Self-indication & cylindrical transmission corrections
This is relevant to your current Python implementation and a physics-based replacement.

- `sammy/src/sesh/*`
  - Cylindrical sample transmission (self-indication) corrections.
  - `sesh2.f90` contains transmission averaging and variance logic.
  - `sesh_init.f90` wires up sample thickness parameters from input.

This is worth extracting as a **standalone module** in the new library.

---

## 6) Unresolved resonance handling (UDR)
This matters for "unknown isotope" scenarios.

- `sammy/src/udr/*`
  - Implements unresolved resonance region calculations.
  - `mudr4.f90` includes TOF / flight-path formulations.

---

## 7) ENDF ingestion & resonance parameter mapping
Useful if you want to generate starting parameters from ENDF without SAMMY.

- `sammy/src/mas/LoadEndfData.cpp`
  - Maps ENDF resonance data to SAMMY's internal model and determines formalism.

- `sammy/src/endf/FillSammyRmatrixFromRMat.*`
  - Converts ENDF resonance info into R-matrix parameters.

---

## 8) HDF5 I/O (existing precedent)
SAMMY already contains HDF5 writing utilities. If you want to align with their structure:

- `sammy/src/io/Hdf5IO.h` / `sammy/src/io/Hdf5IO.cpp`
  - Writes ODF -> HDF5 and resonance covariance matrices.

---

## 9) API hooks (legacy, but useful for orchestration)
SAMMY's "API" interface exists and could be referenced for how to structure a new engine API.

- `sammy/src/sam/SammyApi.h`
  - Shows how SAMMY exposes its internal data objects (resonances, grid, parameters).

---

## 10) Key constants and numerical conventions
- `sammy/src/blk/Constn_common.f90` -- constants (neutron mass, Twomhb, Sm2, etc.).
- `sammy/src/blk/*` -- SAMMY global constants and shared state.

---

## 11) Open questions / unknowns that must be resolved

1) **VENUS resolution model**
   - User-defined file is classified; we need to identify which SAMMY resolution function is being used (RSL vs RPI/GEEL/nTOF or custom).
   - Recommendation: define a *pluggable resolution interface* in Rust and defer concrete model until it can be mapped on the server.

2) **Self-indication**
   - SAMMY's cylindrical correction (`sesh/*`) should be audited for assumptions (geometry, beam size, thickness usage).

3) **Unresolved resonance (UDR)**
   - Must determine exactly which UDR features are required for VENUS (e.g., probability tables, energy range handling).

4) **Parameter file compatibility**
   - Decide whether to keep SAMMY par file compatibility or define a new clean format.

5) **Cross section outputs**
   - Decide which outputs are critical: transmission only vs capture/total, and whether derivatives are required.

---

## 12) Suggested "teacher -> student" porting order

1) **Data model** (Rust structs mirroring `SammyRMatrixParameters` / `SammySpinGroupInfo` / `SammyResonanceInfo`).
2) **0 K Reich-Moore cross sections** (use `clm/dopush1.f90` as the primary reference).
3) **Transmission transform** (`cro/mnrm2.f90`).
4) **Normalization/background** (`cro/mnrm1.f90`).
5) **Doppler broadening** (`clm/mclm3.f90`).
6) **Resolution broadening** (RSL or RPI model, once VENUS model is known).
7) **Self-indication** (`sesh/*`).
8) **Unresolved resonance** (`udr/*`).

Each step can be validated against SAMMY outputs using known parameter sets.

---

## 13) Notes on classified resolution function
Because the VENUS resolution function file cannot leave the secure server, **do not attempt to capture it here**. Instead:
- Implement a resolution interface that can accept user-provided parameters from a secure source.
- Provide a temporary stub resolution model so the rest of the pipeline is testable without the classified file.

---

## 14) VENUS resonance-imaging scope -> development priorities

This section maps the **VENUS user classes** to required physics modules and suggested delivery phases.

### 14.1 Regular users (known isotopes, unknown abundances)
Goal: spatially-resolved abundance maps (2D radiograph first, 3D CT later).

**Required physics modules**
- Reich-Moore forward model (Section 1.1).
- Transmission transform (Section 1.3).
- Doppler broadening (Section 2.1).
- Resolution broadening via pluggable interface (Section 2.2).
- Normalization/background (Section 1.4).
- Self-indication / cylindrical corrections (Section 5).

**Recommended delivery**
- Phase A: 2D abundance mapping (per-pixel fits).
- Phase B: CT reconstruction using same forward model (voxel-wise or projection-wise inversion).

### 14.2 Advanced users (known isotopes + unknown components)
Goal: characterize known isotopes and identify unknown isotope contributions.

**Required physics modules**
- Everything in 14.1.
- Unresolved resonance handling (Section 6).
- Residual analysis and isotope-screening logic (new, outside SAMMY).

**Recommended delivery**
- Phase C: "Known + Unknown" workflow:
  - fit known isotopes,
  - analyze residuals across energy,
  - score candidate isotopes from a reference library,
  - include top candidates in a second-pass fit.

### 14.3 Nuclear physics users (cross-section measurement)
Goal: resonance parameter estimation rather than relying on ENDF.

**Required physics modules**
- Full resonance parameter fitting (beyond abundance-only).
- Robust uncertainty handling (covariance, priors, diagnostics).
- Optional ENDF ingestion (Section 7) to build starting points.

**Recommended delivery**
- Phase D: dedicated "cross-section measurement" mode with uncertainty reporting.

### 14.4 Implications for architecture
To support all three classes without rewriting, the Rust library should be split into:

1) **Forward model engine**
   - Contains RM cross sections, broadening, transmission, normalization, self-indication.
   - Shared by all modes.

2) **Inference layer**
   - Imaging inversion (regular users).
   - Unknown-isotope screening + UDR (advanced users).
   - Resonance parameter fitting (physics users).

---

End of document.
