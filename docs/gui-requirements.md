# NEREIDS GUI Requirements

**Epic**: #168 — GUI Application Redesign for VENUS Beamline
**Prototype**: `.prototypes/D_hybrid_v3.html` (approved hybrid design, v3 iteration)
**Framework**: egui (eframe 0.29+, egui_plot)
**Reference impl**: rustpix (`../rustpix/rustpix-gui`) — viewer, ROI, TOF slicer patterns

---

## 1. Design Principles

| Principle | Description |
|-----------|-------------|
| **Scientific look** | Professional appearance suitable for publication screenshots |
| **Apple design language** | SF Pro typography, rounded corners, subtle borders, backdrop blur |
| **Auto theme** | Follow OS dark/light preference; manual override via toolbar toggle |
| **Dual-mode UI** | Guided (wizard) for regular users, Studio (power-user) via toggle |
| **Progressive disclosure** | Simple defaults, advanced options revealed on demand |

---

## 2. Architecture

### 2.1 Dual-Mode Layout

The GUI has two modes, toggled via a segmented control in the toolbar:

**Guided Mode** (default):
- Left sidebar: numbered workflow steps (Load → Configure → Normalize → Analyze → Results) + Tools section (Forward Model, Detectability)
- Content area: step-specific card-based forms and visualizations
- Bottom: processing history (compact)

**Studio Mode**:
- Left sidebar: project browser (Data, Isotopes, Results, Provenance sections)
- Content area: document tabs (Analysis, Forward Model, Detectability) with split panes
- Right: mini inspector panel (beamline params, pixel details, ROI stats)
- Bottom dock: tabbed panels (Isotopes, Residuals, Provenance, Export)

Both modes share: toolbar, progress indicator, status bar, and application state.

### 2.2 State Architecture

Single `AppState` struct owns all data. Both modes read from / write to the same state.
Switching modes is purely a UI transformation — no data is lost or reset.

### 2.3 Module Structure (proposed)

```
apps/gui/src/
├── main.rs                 # eframe entry point, window config
├── app.rs                  # NereidsApp: mode toggle, state ownership, update()
├── state.rs                # AppState: all persistent data
├── theme.rs                # ThemeColors, dark/light palettes, OS detection
├── widgets/
│   ├── mod.rs
│   ├── toolbar.rs          # Toolbar: logo, mode toggle, tools, progress, actions
│   ├── statusbar.rs        # Status bar: mode, dimensions, isotope count, version
│   ├── viewer.rs           # Image viewer: texture display, zoom, pan, colormap
│   ├── tof_slicer.rs       # Vertical TOF bin slider
│   ├── roi.rs              # ROI selection: rectangle drawing, hit-testing, overlay
│   ├── spectrum.rs         # Spectrum plot: measured + fit lines, TOF marker, legend
│   ├── periodic_table.rs   # Modal periodic table element picker → isotope selector
│   ├── drop_zone.rs        # File drop target with loaded/empty states
│   └── provenance.rs       # Processing history log widget
├── guided/
│   ├── mod.rs              # Guided mode container
│   ├── sidebar.rs          # Step navigator + tools + history
│   ├── load.rs             # Step 1: data loading (tabbed input types)
│   ├── configure.rs        # Step 2: beamline params + isotope selection
│   ├── normalize.rs        # Step 3: transmission computation + preview
│   ├── analyze.rs          # Step 4: solver + split map/spectrum view
│   ├── results.rs          # Step 5: maps, export, provenance
│   ├── forward_model.rs    # Tool: density sliders + live spectrum
│   └── detectability.rs    # Tool: trace element assessment
├── studio/
│   ├── mod.rs              # Studio mode container
│   ├── sidebar.rs          # Project browser
│   ├── doc_tabs.rs         # Analysis / Forward Model / Detectability tabs
│   ├── split_view.rs       # Map + Spectrum + Inspector split pane
│   ├── inspector.rs        # Mini inspector (right panel)
│   └── dock.rs             # Bottom dock: isotopes, residuals, provenance, export
└── io/
    ├── mod.rs
    ├── loader.rs           # Async file loading (TIFF, HDF5, spectrum)
    └── exporter.rs         # Export to TIFF / HDF5 / Markdown
```

---

## 3. Toolbar

| Element | Description | Mode |
|---------|-------------|------|
| App logo | `nereids-logo.svg` inline as egui image, 22x22 px | Both |
| App name | "NEREIDS" bold label | Both |
| Mode toggle | Segmented: "Guided" / "Studio" | Both |
| Tool buttons | Select, ROI, Probe, Zoom | Studio |
| Progress | Mini progress bar + "72% — 47k/65k" | Both |
| Spatial Map | Primary action button (green when ready) | Both |
| Theme toggle | Light/Dark/Auto | Both |

---

## 4. Data Input (Step 1: Load)

### 4.1 Input Types

Four input modes, presented as tabs in the Load step:

| Tab | Files | Description |
|-----|-------|-------------|
| **TIFF Pair + Spectrum** | Sample TIFF stack + Open beam TIFF stack + Spectrum file (.csv/.txt) | Memory-efficient frame-by-frame loading. Rebinning limited to spectrum file bins. Most common VENUS workflow. |
| **Transmission TIFF + Spectrum** | Transmission TIFF stack + Spectrum file (.csv/.txt) | Pre-computed T(E) data. Spectrum file required to map frames to energy/TOF. Skips normalization step. |
| **HDF5 Event Data** | .nxs or .h5 event file | Raw event data with flexible TOF rebinning. User configures: bin count, bin mode (equal-width/equal-count/log), TOF range. Energy axis derived from binning + flight path. |
| **HDF5 Histogram** | .h5 or .nxs histogram file | Pre-binned 3D data with embedded energy axis. User selects dataset path from HDF5 tree. |

**TIFF input flexibility**: Each TIFF drop zone accepts either a **folder of individual TIFFs** or a **single multipage TIFF file**. Auto-detect on drop/browse.

### 4.2 File Loading

- Use `rfd` for native file dialog
- Support drag-and-drop onto drop zones
- Async loading via channels (non-blocking UI)
- Show progress: frame count, dimensions, file size
- Validate on load: check shape consistency, TOF monotonicity

### 4.3 Data Model

```
Input → (sample_stack, ob_stack, spectrum) | (transmission_stack, spectrum) | event_data | histogram_data
     → normalize (if needed)
     → Hyperspectral3D { data: [tof × height × width], energies: Vec<f64> }
```

All input modes ultimately produce a `Hyperspectral3D` with an associated energy axis.
The spectrum file (required for TIFF-based inputs) provides the TOF/energy bin edges.

---

## 5. Image Viewer

### 5.1 Core Features (reference: rustpix `viewer/`)

| Feature | Description |
|---------|-------------|
| **Texture display** | Render 2D slice as `ColorImage` via `PlotImage` in egui_plot |
| **Colormaps** | Viridis (default), Inferno, Grayscale, Hot. Selection via dropdown. |
| **Zoom** | Mouse wheel zoom centered on cursor. Box-zoom via Zoom tool. |
| **Pan** | Click-drag when in Select mode |
| **Reset** | Double-click resets to auto-fit bounds |
| **Pixel readout** | Hover shows (x, y) coordinates + value in status bar |
| **Grid overlay** | Optional toggle |

### 5.2 TOF Dimension Slicer

Vertical slider alongside the image viewer, enabling stepping through TOF/energy bins:

- Vertical range slider (0 to N_bins)
- Current bin index displayed above slider
- Changing the slider updates the displayed 2D slice in real time
- Corresponding vertical marker line on the spectrum plot shows current TOF position
- Label: "TOF" (rotated vertically)

Implementation: `tof_slicer.rs` widget, integrated into viewer container.
Reference: rustpix `UiState::current_tof_bin` + vertical slider.

### 5.3 ROI Selection

Rectangle region-of-interest selection on the image viewer:

| Feature | Description |
|---------|-------------|
| **Drawing** | Shift+drag to draw rectangle (matches rustpix convention) |
| **Display** | Semi-transparent fill + colored border |
| **Resize** | Drag corner/edge handles |
| **Move** | Drag interior to reposition |
| **Delete** | Select + Delete key |
| **Statistics** | ROI dimensions and averaged spectrum shown in inspector |
| **Spectrum** | ROI-averaged spectrum overlaid on the spectrum plot |

Reference: rustpix `viewer/roi.rs` (534 lines, production-grade).
Initially support rectangle only; polygon ROI is a future enhancement.

---

## 6. Spectrum Viewer

### 6.1 Core Features

| Feature | Description |
|---------|-------------|
| **Measured line** | Solid line, accent color |
| **Fit line** | Dashed line, red |
| **ROI spectrum** | Additional line when ROI is active |
| **TOF marker** | Vertical dashed line at current TOF slicer position |
| **Axes** | X: Energy (eV) default, toggle to TOF (us). Y: Transmission (0–1) |
| **Axis toggle** | Segmented control: "Energy (eV)" / "TOF (us)". Default: Energy for resonance imaging. |
| **Resonance markers** | Optional overlay showing known resonance dip positions for selected isotopes (element symbol + energy label at each dip). Toggle via checkbox. |
| **Legend** | Color-coded, toggleable visibility |
| **Zoom** | Independent zoom from image viewer |
| **Chi-squared** | Badge in pane header |

### 6.2 Interaction

- Click on spectrum selects corresponding TOF bin (updates slicer + image)
- Zoom via mouse wheel / box-zoom
- Pan via click-drag
- Double-click resets zoom

### 6.3 Normalize Preview

In the Normalize step (Step 3), the spectrum viewer serves as a **quality-check tool**:
- Shows the computed transmission spectrum (full image average or ROI average)
- Data source selector: "Full image (all pixels)" / "ROI average"
- Resonance dip markers help beamline scientists verify normalization quality
- This is the primary tool for checking "does the spectrum make sense?"

---

## 7. Isotope Selection

### 7.1 Direct Entry

Text input field accepting isotope names like "U-235", "Pu-241", "Fe-56".
Autocomplete from known ENDF isotope list.

### 7.2 Periodic Table Element Picker

Modal dialog with compact periodic table grid:

1. Click element cell to select it
2. Element's available isotopes shown below the table as selectable chips
3. User selects one or more isotopes
4. User sets initial density (atoms/barn) and ENDF library
5. "Add Selected" button adds isotopes to the session

Layout: 18-column periodic table grid (rows 1–7 + lanthanide/actinide rows).
Color-coding: metals, non-metals, actinides.

### 7.3 Isotope Table (Studio dock)

| Column | Type | Description |
|--------|------|-------------|
| Checkbox | bool | Enable/disable for fitting |
| Name | string | e.g. "U-235" |
| Z | int | Atomic number |
| A | int | Mass number |
| Initial Density | float input | Starting guess (atoms/barn) |
| ENDF | status icon | Fetched / pending |
| Fitted | float | Result density |
| Uncertainty | float | 1-sigma error |
| Status | icon | Converged / failed |

---

## 8. Output & Export (Step 5: Results)

### 8.1 Output Quantities

| Quantity | Type | Description |
|----------|------|-------------|
| Density maps | 2D per isotope | Areal density in atoms/barn |
| Uncertainty maps | 2D per isotope | 1-sigma uncertainty |
| Fitted temperature | scalar or 2D | Temperature from Doppler fitting (K) |
| Convergence map | 2D | Boolean or iteration-count map |
| Enrichment map | 2D | Derived ratio (e.g. U-235/U-total) |

### 8.2 Export Formats

| Format | Description |
|--------|-------------|
| **TIFF Stack** | One TIFF file per output quantity. Universal compatibility. |
| **HDF5/NeXus (New File)** | All maps + metadata in a single structured HDF5. |
| **Append to Existing HDF5** | Add NEREIDS results to the original experiment file. Preserves experiment context. |
| **Provenance Report (Markdown)** | Processing history as human-readable + AI-ingestible markdown. |

### 8.3 Export Contents Selection

Checkboxes for: density, uncertainty, temperature, convergence, enrichment, provenance metadata.
User selects format + contents, then clicks "Export".

---

## 9. Provenance Tracking

Every user action that modifies the analysis state is logged as a provenance entry:

### 9.1 Logged Events

| Event | Data Captured |
|-------|---------------|
| File load | File path, type, dimensions, timestamp |
| Beamline config | Flight path, delta_t, delta_l, temperature |
| Isotope addition/removal | Isotope name, Z, A, initial density, ENDF library |
| Normalization | TOF range, bin count, pixel count |
| Fit execution | Solver type, max_iter, convergence %, median chi-squared, wall time |
| Export | Format, output path, contents selection |

### 9.2 Display

- **Guided mode**: compact history in sidebar bottom section; full log in Results step
- **Studio mode**: "Provenance" tab in bottom dock

### 9.3 Export

- **Markdown**: Timestamped log with parameters, formatted for human reading and AI ingestion
- **HDF5 metadata**: Structured attributes in `/nereids/provenance/` group
- **Includes**: NEREIDS version, git commit hash, timestamp, all parameter values

---

## 10. Beamline Configuration

### 10.1 Parameters

| Parameter | Unit | Default | Validation |
|-----------|------|---------|------------|
| Flight path | m | 25.0 | > 0, finite |
| Delta_t (1-sigma) | us | 1.0 | >= 0, finite |
| Delta_l (1-sigma) | m | 0.01 | >= 0, finite |
| Temperature | K | 296 | >= 0, finite |

### 10.2 ENDF Library

Dropdown: ENDF/B-VIII.1 (default), JEFF-3.3, JENDL-5.
Fetch button triggers async ENDF download with progress indicator.

---

## 11. Normalize & Preview (Step 3)

### 11.0 Analysis Mode Selection

After normalization, users choose one of three analysis modes:

| Mode | Description |
|------|-------------|
| **Full Spatial Map** (default) | Fit every pixel independently. Produces full-resolution density and temperature maps. |
| **ROI → Single Spectrum** | Accumulate counts from selected ROI into one high-statistics spectrum. Fit a single result. Useful for nuclear physicists with special samples. |
| **Spatial Binning → Map** | Bin NxN pixels together (2x2, 4x4, 8x8) to improve statistics, then fit the binned map. User decision — not strictly necessary with KL fitting. |

The Normalize step includes:
- Transmission computation controls
- Side-by-side image viewer (with ROI and TOF slicer) + transmission spectrum preview
- Spectrum data source selector: "Full image" / "ROI average"
- Resonance dip markers (optional toggle) for isotope identification
- Energy/TOF axis toggle (default: Energy)
- Analysis mode selection cards

---

## 12. Analysis (Step 4)

### 12.1 Solver Configuration

**Primary controls** (always visible):

| Parameter | Options | Default | Note |
|-----------|---------|---------|------|
| Method | Levenberg-Marquardt, Poisson KL | LM | Dropdown |
| Max iterations | int | 200 | Labeled input field |

**Advanced controls** (behind gear icon toggle):

| Parameter | Options | Default |
|-----------|---------|---------|
| Fit temperature | Yes/No | No |
| Compute covariance | Yes/No | Yes |
| Convergence tolerance | float | 1e-6 |
| Lambda init (LM) | float | 0.001 |

### 12.2 Execution

- "Spatial Map" button in toolbar: fit all pixels (rayon parallel)
- Progress bar: pixel count / total, percentage, ETA
- Cancel button during fitting

### 12.3 Results Display

- Split view: density map (left) + spectrum at selected pixel (right)
- Click pixel on map → update spectrum + inspector
- ROI → averaged spectrum shown
- Pixel results table: per-isotope density, uncertainty, status badge
- Energy/TOF axis toggle on spectrum

---

## 13. Forward Model (Tool)

The Forward Model operates **independently** from the main Configuration step.

### 13.1 Independent Isotope Controls

The Forward Model has its own isotope table with:
- Checkbox to enable/disable each isotope (quick toggle without removing)
- Density slider + numeric input per isotope
- Add/remove isotope buttons (+ Periodic Table access)
- **"Copy from Config"** button: reset FM isotopes to match the Configuration step
- **"Push to Config"** button: copy FM isotope settings to the Configuration step

This allows users to experiment with different isotope combinations without navigating back to Configure.

### 13.2 Spectrum Display

- Forward model transmission spectrum with per-isotope contribution lines
- Energy/TOF axis toggle
- Real-time update as density sliders change

---

## 14. Detectability (Tool)

### 14.1 Expandable Composition Lists

Both matrix and trace isotope lists are **expandable** (not limited to one each):

| List | Per-row fields | Actions |
|------|----------------|---------|
| **Matrix isotopes** | Isotope name + density (at/barn) | Add row, remove row |
| **Trace isotopes** | Isotope name + concentration (ppm) | Add row, remove row |

### 14.2 Advanced Configuration (behind gear/toggle)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Detection threshold | 3.0 sigma | Minimum SNR to declare "detectable" |
| I_0 (counts/bin) | 10,000 | Incident beam intensity |
| Energy range | 1–100 eV | Search range for resonance peaks |
| Background model | Flat | Flat or linear background assumption |

### 14.3 Results Table

Multi-row verdict table showing each trace isotope's best peak, SNR, minimum detectable ppm, and pass/fail badge.

---

## 15. Results & Export (Step 5)

### 15.1 Image Toolbelt

Each density map tile has a per-tile toolbar (inspired by plotly):

| Tool | Action |
|------|--------|
| **Colormap selector** | Dropdown: Viridis, Inferno, Plasma, Grayscale, RdBu |
| **Toggle colorbar** | Show/hide colorbar overlay with min/max labels |
| **Save image** | Export the individual map as PNG/TIFF |
| **Zoom to fit** | Reset view bounds |

The Results step content area must be **scrollable** — it contains stat summary, result grid (4+ tiles), export panel, and provenance log.

---

## 16. Step Navigation

### 16.1 Teleport Buttons

Bi-directional jump buttons allow quick navigation between related steps:

| From | Teleport to |
|------|-------------|
| Configure | Forward Model |
| Forward Model | Configure, Analyze |
| Analyze | Forward Model |
| Detectability | Configure |

Teleport buttons appear as small accent-colored pills alongside the standard Back/Continue flow.

### 16.2 Rationale

Users frequently iterate between Configure ↔ Forward Model (to test isotope hypotheses) and Forward Model ↔ Analyze (to compare predictions vs. data). Direct jumps eliminate repetitive sidebar clicking.

---

## 17. Theme System

### 17.1 Color Palette

```
Light:                          Dark:
  bg:     #f5f5f7                 bg:     #1c1c1e
  bg2:    #ffffff                 bg2:    #2c2c2e
  bg3:    #e8e8ed                 bg3:    #3a3a3c
  fg:     #1d1d1f                 fg:     #f5f5f7
  fg2:    #6e6e73                 fg2:    #98989d
  fg3:    #86868b                 fg3:    #636366
  accent: #0071e3                 accent: #0a84ff
  border: #d2d2d7                 border: #48484a
```

### 17.2 Semantic Colors (fixed across themes)

```
green:  #34c759  (success, converged)
red:    #ff3b30  (error, failed)
orange: #ff9500  (warning, pending)
yellow: #ffcc00  (highlight)
```

### 17.3 Detection

Use `ui.visuals().dark_mode` to detect current theme.
Reference: rustpix `ui/theme.rs`, bm3dornl `theme.rs`.

---

## 18. Status Bar

Always visible at bottom. Contents:

| Item | Example |
|------|---------|
| Status dot | Green (ready), orange (processing) |
| Mode + step | "Guided — Step 4: Analyze" or "Studio Mode" |
| Data dimensions | "256 x 256 x 500" |
| Isotope count | "3 isotopes" |
| Beamline | "VENUS 25 m" |
| Version | "NEREIDS v0.1.0" |

---

## 19. Implementation Priorities

### Phase 1: Core Framework
- [ ] Theme system (dark/light/auto)
- [ ] Toolbar + status bar
- [ ] Mode toggle (Guided/Studio skeleton)
- [ ] State architecture refactor
- [ ] Step navigation (sidebar, teleport buttons)

### Phase 2: Data Pipeline
- [ ] TIFF stack loading (async, drop zone — folder or multipage)
- [ ] Spectrum file loading (energy/TOF bin mapping)
- [ ] Transmission computation
- [ ] Image viewer with colormap + per-tile toolbelt
- [ ] Normalize preview (transmission spectrum, data source selector)

### Phase 3: Analysis
- [ ] TOF slicer widget
- [ ] Spectrum plot (egui_plot) with energy/TOF axis toggle
- [ ] Resonance dip markers overlay
- [ ] Isotope selection (direct entry)
- [ ] Analysis mode selection (Full Map / ROI→Spectrum / Spatial Binning)
- [ ] Solver config — primary controls + advanced gear toggle
- [ ] Single-pixel fitting + results display
- [ ] Spatial map fitting with progress

### Phase 4: Advanced Features
- [ ] ROI selection (rectangle)
- [ ] Periodic table isotope picker
- [ ] Forward Model tool (independent isotope controls, Copy/Push buttons)
- [ ] Detectability tool (expandable lists, advanced config toggle)
- [ ] HDF5/NeXus input support
- [ ] Export panel (TIFF, HDF5, Markdown)
- [ ] Per-tile image toolbelt (colormap, colorbar, save, zoom)
- [ ] Provenance tracking

### Phase 5: Polish
- [ ] Keyboard shortcuts
- [ ] Window state persistence
- [ ] Error handling + user feedback
- [ ] Studio mode full integration

---

## 20. Dependencies

| Crate | Purpose | Current |
|-------|---------|---------|
| `eframe` | Window + app framework | 0.33 |
| `egui` | Immediate-mode GUI | 0.33 |
| `egui_plot` | Spectrum plotting | 0.34 |
| `rfd` | Native file dialogs | 0.17 |
| `image` | Texture generation | existing |
| `hdf5` | HDF5/NeXus I/O | TBD |
| `tifffile` (via nereids-io) | TIFF stack loading | existing |
| `rayon` | Parallel fitting | existing |

---

## 21. Key References

| Reference | Path | Relevance |
|-----------|------|-----------|
| rustpix GUI | `../rustpix/rustpix-gui/` | Viewer, ROI, TOF slicer, colormaps, theme |
| bm3dornl GUI | `../bm3dornl/src/rust_core/crates/bm3d_gui_egui/` | Widget architecture, ProcessingManager, HDF5 tree |
| Current NEREIDS GUI | `apps/gui/src/` | Existing code to refactor |
| Prototype D v3 | `.prototypes/D_hybrid_v3.html` | Approved visual design (current iteration) |
