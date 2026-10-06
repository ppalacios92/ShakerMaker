# ShakerMaker `examples/` Reference

This document catalogs every script and notebook under `examples/`, one folder at a time:
what it does, what it needs as input, and what it produces as output.

**Method**: every `.py` and `.ipynb` file in scope was read in full (code cells, markdown
cells, and — where present — saved cell outputs). Claimed output files were cross-checked
against what actually exists on disk in this repo checkout. **No script was executed** as
part of this review, so any output described as "produced" reflects what the code says it
writes (and, where a matching file exists on disk, that this has in fact happened at least
once before), not a freshly reproduced run.

Anything that could not be confirmed from the code, notebook output, or repo contents alone
is marked **⚠️ Open question** inline, and repeated in the [consolidated appendix](#open-questions-appendix)
at the end. Nothing in this document is inferred or invented to fill those gaps.

## How examples are run as a suite

`examples/run_all_smoke.py` walks every `.py` file under `examples/` (skipping `legacy_examples/`,
any `notebooks/` directory, and any generated `_*` output directory) and runs each as a
subprocess, classifying it PASS / SKIP (if `"SKIP"` appears in stdout/stderr — the pattern
every gracefully-degrading script in this repo uses) / FAIL. `12_validation/` is skipped
unless `--full` is passed, because it runs a full (non-OP) FK synthesis and is slow.
Notebooks are **not** part of this smoke suite — they are the narrated/visual counterparts,
run manually.

## Folder index

| Folder | Topic | Needs MPI | Needs external tool/data | Runs the FK engine |
|---|---|---|---|---|
| [01_crustmodel](#01-crust-models) | Velocity models, CRUST 1.0 | No | No (CRUST1.0 ships with the package) | No |
| [02_sources](#02-sources) | Point sources, finite-fault subfault grids | No | No | No |
| [03_stf](#03-source-time-functions) | The 5 STF classes | No | No | No |
| [04_receivers](#04-receivers) | Station/DRMBox/SurfaceGrid/PointCloud layouts | No | 1 fixture file (included) | No |
| [05_engine_direct](#05-engine-direct) | Raw Fortran kernel → legacy `.run()` | No | No | Yes (legacy path) |
| [06_nearest_method](#06-nearest-method) | OP pipeline (Stage 0/1/2) | No | Optional legacy DB (not included) | Yes (OP path) |
| [07_writers](#07-writers) | HDF5 / DRM / native `.npz` persistence | No | No | Yes (legacy path) |
| [08_drm](#08-drm) | Domain Reduction Method end-to-end | Yes (`drm_loh1.py`) | No | Yes |
| [09_sw4_export](#09-sw4-export) | Export to SW4, round-trip back to `.h5drm` | No | **Yes — SW4 itself, run externally** | No (export only) |
| [10_ffsp](#10-ffsp) | Stochastic finite-fault source (FFSP) | No | No | Optional (one notebook) |
| [11_plotting](#11-plotting) | Generic plotting helpers | No | No | Partial (one notebook) |
| [12_validation](#12-validation) | SCEC LOH.1 + LOH.3 physics benchmarks | No | Reference solutions (included) | Yes (legacy path) |
| [13_shakermaker_sw4](#13-shakermaker-sw4) | FK vs. real SW4 run, cross-validation | Recommended | SW4 output + ObsPy (included as data) | Yes |
| [14_SFSI](#14-sfsi) | Real-site (Samoa Beach) end-to-end case study | Yes (HPC scripts) | No | Yes |
| [legacy_examples](#legacy-examples) | Original upstream regression suite | No | No | Yes (legacy path) |

---

## 01. Crust Models

`examples/01_crustmodel/`

Covers two independent workflows: building/editing a `CrustModel` by hand (layers,
splitting, depth queries, plotting), and querying the bundled CRUST 1.0 global dataset at
an arbitrary lat/lon to get a ready-to-use crustal profile plus a paste-ready `CrustModel`
snippet. No external data, no MPI, no HPC — pure local Python, no FK engine run.

#### `crustmodel_build.py`
- **Purpose**: Builds a 2-layer `CrustModel` (slow layer over half-space), modifies a
  layer in place, samples Vp/Vs/rho/Qp/Qs at given depths, and instantiates two library
  presets (`SCEC_LOH_1`, `SCEC_LOH_3`).
- **Required inputs**: None external. All parameters hardcoded.
- **Outputs**: stdout only (`print(crust)`, depth arrays). No files. Ends in
  `assert` + `print("PASS")` — a smoke test.
- **Dependencies**: numpy, `shakermaker.crustmodel`, `shakermaker.cm_library.LOH`.

#### `crust1_sites.py`
- **Purpose**: Queries the bundled CRUST 1.0 dataset at two lat/lon sites (Santiago, San
  Francisco), prints per-site summaries and a ready-to-paste `CrustModel` snippet via
  `Crust1.print_shakermaker`. The second half exercises the full `Crust1` API (cell index,
  midpoint, profile dict, geological type, table print, 6 plotting methods).
- **Required inputs**: None — CRUST 1.0 grid data ships inside the `shakermaker.crust1`
  package, no path argument needed.
- **Outputs**: stdout + interactive matplotlib figures (`plot_profile`,
  `plot_global_topo`, `plot_regional_topo`, `plot_regional_geological`,
  `plot_global_velocity`, `plot_stacked_columns`) — **none saved to disk** (no `savefig`).
- **Dependencies**: `shakermaker.crust1` only.
- **Note**: the full-API section (lines 21-55) intentionally runs *outside* the
  `for lat, lon in sites:` loop, reusing the loop's last values (San Francisco) — a
  deliberate two-part structure: a quick multi-site pass first, then a single deep-dive
  through the full `Crust1` API on one site.

#### `notebooks/crustmodel.ipynb`
- **Purpose**: Part 1 — builds an LOH.1-style 2-layer `CrustModel`, plots the layered
  column and velocity profile, queries properties at depth, demonstrates `split_at_depth`
  + `modify_layer` to carve a shallow weak zone, re-plots. Part 2 — repeats the full
  `Crust1` API walkthrough at one site (lat=40.764, lon=-123.104).
- **Required inputs**: None external.
- **Outputs** (confirmed present in repo): `crust_layers.png`, `crust_velocity_profile.png`,
  `crust_split_modified_profile.png`. The 6 `Crust1` plots have no `savefig` call and
  correspondingly no matching PNGs exist — display-only, consistent with `crust1_sites.py`.
- **Dependencies**: numpy, matplotlib, `shakermaker.crustmodel`, `shakermaker.crust1`.

---

## 02. Sources

`examples/02_sources/`

Demonstrates the two source-building patterns: a single `PointSource` with a smooth
`Gaussian` STF, and a finite-fault-like grid of subfaults with `SRF2` slip-rate STFs and
randomized rake/slip. Only the SRF2/multi-subfault path is visualized. No MPI/HPC, no
external data files. Plotting requires a SciPy compatibility shim (see below).

#### `pointsource.py`
- **Purpose**: Builds a single `PointSource` (depth 2 km, strike/dip/rake = [0,90,0])
  driven by a `Gaussian` STF, wrapped in a `FaultSource`.
- **Required inputs**: None, all hardcoded (`sigma=0.06`, `M0=1e18/5e14/2`).
- **Outputs**: stdout only (`assert fault.nsources == 1`, `PASS`). No files/plots.
- **Dependencies**: `shakermaker.cm_library.LOH`, `shakermaker.pointsource`,
  `shakermaker.faultsource`, `shakermaker.stf_extensions.gaussian`.
- ⚠️ **Open question**: unlike `faultsource_srf2.py`, this script has no notebook
  counterpart showing it plotted or run through the FK engine — it only builds the object
  graph.

#### `faultsource_srf2.py`
- **Purpose**: Builds a 2×5 grid (10 subfaults) of `PointSource`s around a hypocenter at
  (0,0,3) km, each with randomized rake (±10° around 0°) and slip (0.5–1.5 m, fixed RNG
  seed `np.random.default_rng(0)`), each driven by an `SRF2` slip-rate STF
  (`Tr=2.0, Tp=0.1, Te=1.5, dt=0.01, a=1.0, b=1.0`). Wrapped into one `FaultSource`.
- **Required inputs**: None, all hardcoded; fixed RNG seed for reproducibility.
- **Outputs**: stdout only (`assert fault.nsources == 10`, `PASS`).
- **Dependencies**: numpy, `shakermaker.cm_library.LOH`, `shakermaker.pointsource`,
  `shakermaker.faultsource`, `shakermaker.stf_extensions.srf2`.

#### `notebooks/sources.ipynb`
- **Purpose**: Notebook version of `faultsource_srf2.py` (same grid, same seed), plus a
  3D visualization via `SourcePlot(fault, colorby="slip", colorbar=True)`.
- **Required inputs**: None external. The first cell installs a SciPy compatibility
  shim (`scipy.integrate.trapz = scipy.integrate.trapezoid` before importing
  `shakermaker.tools.plotting`), working around `SourcePlot` importing the SciPy
  ≥1.14-removed name `trapz`.
- **Outputs**: `sources_geometry.png` (confirmed present, dpi=150), titled "Fault
  geometry (2x5 SRF2 subfaults)".
- **Dependencies**: numpy, matplotlib, scipy.integrate (for the shim),
  `shakermaker.tools.plotting.SourcePlot`.
- **Resolved**: the shim is no longer needed. `SourcePlot` was fixed upstream in
  commit `baddb87` to import `trapezoid` natively (`shakermaker/tools/plotting.py:233`)
  — this notebook's shim is now a harmless no-op, not a required workaround. It does not
  affect `StationPlot`/`ZENTPlot`, which never used `trapz`.

---

## 03. Source Time Functions

`examples/03_stf/`

Self-contained gallery of the 5 STF classes ShakerMaker ships. No external inputs, no
engine run, no MPI. The `.py` file is a pure smoke test; the notebook is the actual visual
reference and produces exactly the 6 PNGs present in the repo — the cleanest 1:1
correspondence between code and artifacts of any folder reviewed.

#### `stf_gallery.py`
- **Purpose**: Instantiates all 5 STF types (`Dirac`, `Discrete`, `Brune`, `Gaussian`,
  `SRF2`) with representative parameters, sets `.dt = 0.001` on each (triggers internal
  data generation), asserts each produced non-empty `.t`/`.data` arrays.
- **Required inputs**: None. For `Discrete`, the script builds its own synthetic pulse
  (`np.exp(-((t_user-0.1)/0.02)**2)` on `t_user = linspace(0, 0.3, 301)`) rather than
  reading a file.
- **Outputs**: stdout only (`PASS`).
- **Dependencies**: numpy, the 5 `shakermaker.stf_extensions.*` classes.

#### `notebooks/stf_gallery.ipynb`
- **Purpose**: Builds and plots each of the 5 STFs individually, then a combined 2×3 grid
  figure with all 5 together.
- **Required inputs**: None; same synthetic parameters as the `.py` version (`DT=0.001`;
  Brune `f0=10.0, t0=0.5`; Gaussian uses the LOH.1 convention `sigma=0.06s → t0=6*sigma,
  freq=1/sigma`; SRF2 `Tr=2.0, Tp=0.1, Te=1.5, slip=12.0, a=1.0, b=1.0`).
- **Outputs** (all confirmed present): `stf_dirac.png` (stem plot), `stf_discrete.png`
  (the user-supplied pulse), `stf_brune.png` (original vs. smoothed, overlaid),
  `stf_gaussian.png` (LOH.1 parameters), `stf_srf2.png`, `stf_gallery.png` (all 5 in one
  2×3 grid).
- **Dependencies**: numpy, matplotlib (`ggplot` style).

---

## 04. Receivers

`examples/04_receivers/`

Covers the receiver/station layout classes: a single `Station`/`StationList`, `DRMBox`
(closed box for DRM), `SurfaceGrid` (plane/hollow/filled modes), and
`PointCloudDRMReceiver` (import from an external FEM mesh export). All layouts are kept
small (≤50 stations) for fast smoke-testing. One real external input file
(`_drm_nodes.txt`) is used and is present/consistent. No MPI/numba/HPC.

#### `single_station.py`
- **Purpose**: Builds one `Station` (at [6,8,0] km) wrapped in a `StationList`; checks
  `nstations==1`, coordinate round-trip, `is_internal` defaults to `False`.
- **Outputs**: stdout only.
- **Dependencies**: `shakermaker.station`, `shakermaker.stationlist`.

#### `drmbox.py`
- **Purpose**: Builds a `DRMBox` receiver layout — a 1×1×1-element box at center
  [6,8,0] km, 10 m spacing (`h=[0.010]*3`) — asserts station count between 1 and 50.
  In-code comment explicitly notes `azimuth` was **removed** from `DRMBox`'s signature.
- **Outputs**: stdout only (`nstations`, `metadata["drmbox_x0"]`).
- **Dependencies**: `shakermaker.sl_extensions.DRMBox`.

#### `pointcloud_drm.py`
- **Purpose**: Builds a `PointCloudDRMReceiver` from the tab-separated node file
  `_drm_nodes.txt` (6 FEM nodes, millimeters), applying a coordinate transform
  (`crd_scale=1/1e6` mm→km, `x0_fem` offset, `drmbox_x0` target). Asserts 7 total stations
  (6 from file + 1 QA station the class appends automatically).
- **Required inputs**: `_drm_nodes.txt` (present, tab-separated, columns
  `Node_ID X Y Z Type`, `Type ∈ {internal, external}`). Transform parameters
  (`x0_fem=[22000,15500,0]`, `drmbox_x0=[6,8,0]`) are stated in-comment to mirror "the
  user's `main_HalfSpace` block" from an external STKO/FEM workflow.
- **Outputs**: stdout (`nstations`, count of internal stations, `drmbox_x0`).
- **Dependencies**: `shakermaker.sl_extensions.PointCloudDRMReceiver`, `os`.
- **Note**: the transform values (`x0_fem`, `drmbox_x0`) are illustrative example
  numbers, not tied to a specific real project — kept as-is, intended purely as an example.

#### `surface_grid.py`
- **Purpose**: Builds three `SurfaceGrid` variants at center [6,8,0] km, 2 km spacing:
  `mode='plane'` (XY plane at z=0, 36 pts), `mode='hollow'` (box boundary only),
  `mode='filled'` (full 27-pt volume). Asserts each stays ≤50 stations.
- **Outputs**: stdout (station counts for all three).
- **Dependencies**: `shakermaker.sl_extensions.SurfaceGrid`.

#### `_drm_nodes.txt`
- Fixture data: 6-row TSV of FEM node coordinates, columns `Node_ID, X, Y, Z, Type` (mm).
  Consumed by `pointcloud_drm.py` and `notebooks/receivers_geometry.ipynb`. Content is
  internally consistent with both consumers.

#### `notebooks/receivers_geometry.ipynb`
- **Purpose**: Visual companion to `drmbox.py`, `surface_grid.py`, and
  `pointcloud_drm.py`. Defines a shared `scatter_stations()` helper (blue = exterior,
  red triangle = internal/QA), then builds/plots a `DRMBox`, a `SurfaceGrid` in
  `'plane'` mode, and a `PointCloudDRMReceiver` from `_drm_nodes.txt`.
- **Required inputs**: `_drm_nodes.txt`, one directory up (resolves correctly).
- **Outputs** (all confirmed present): `drmbox_geometry.png`, `surfacegrid_geometry.png`,
  `pointcloud_geometry.png`.
- **Dependencies**: numpy, matplotlib (3D projection), `shakermaker.sl_extensions`.
- ⚠️ **Open question**: the notebook does not visualize `SurfaceGrid`'s `'hollow'` mode
  (only `'plane'`), and `single_station.py` has no visual counterpart at all — not an
  error, just incomplete coverage relative to the `.py` smoke tests.

**Cross-cutting note for this folder set (01–04)**: none of these 16 files invoke
`mpi4py`, `numba`, or `h5py`, and none call `ShakerMaker(...).run()`/`run_nearest()` — they
are all pre-engine, object-construction-and-visualization examples. Every `.py` script
ends in `assert ...; print("PASS")`, consistent with being part of the smoke suite.

---

## 05. Engine Direct

`examples/05_engine_direct/`

Introduces the lowest-level entry points to the FK engine, from the raw Fortran kernel
(`subgreen`) up through the direct/legacy `model.run()` path (no OP/nearest optimization,
no MPI). Everything is self-contained, centered on the SCEC LOH.1 canonical benchmark
crust. The notebook is the real tutorial; the three `.py` files are minimal PASS/FAIL
smoke tests of the same three levels.

#### `check_parameters.py`
- **Purpose**: Calls `model.check_parameters(dt=..., nfft=..., dk=..., tb=..., tmax=...)`
  — a pure-arithmetic pre-run validation (no FK core call) — on SCEC_LOH_1 crust + one
  Gaussian point source + one station. Asserts the return is a dict.
- **Outputs**: stdout only (`PASS`).
- **Dependencies**: core shakermaker classes only.

#### `core_subgreen.py`
- **Purpose**: Calls the raw Fortran kernel `shakermaker.core.subgreen` directly,
  bypassing the whole object layer, with a manually specified 3-layer SCEC LOH.1-like
  velocity model and a full explicit FK parameter list (`mb, src, rcv, stype, updn, nx,
  sigma, smth, wc1, wc2, pmin, pmax, dk, kc, taper, pf, df, lf, sx, sy, rx, ry`). Asserts
  output array shapes.
- **Outputs**: stdout only (`PASS`). Returns `tdata, z, e, n, t0` in-memory, not saved or
  plotted.
- **Dependencies**: numpy, `shakermaker.core.subgreen` (compiled f2py extension).
- ⚠️ **Open question**: the low-level FK flags (`mb, src, rcv, stype, updn, sigma, smth,
  wc1, wc2, kc, taper, pf, df, lf`) are not documented in this file or in `core.pyf`
  beyond their names — their semantics were not verified in this pass.

#### `run_simple.py`
- **Purpose**: Full direct-engine smoke test — SCEC_LOH_1 crust + Gaussian point source +
  3 stations, `check_parameters` then `model.run(...)` (legacy/direct pipeline), then
  reads station 1's response via `get_response()`.
- **Outputs**: stdout only (`PASS`). Response checked in-memory only.
- **Dependencies**: core shakermaker classes only.

#### `notebooks/engine_direct.ipynb`
- **Purpose**: The actual tutorial. Same SCEC_LOH_1 + Gaussian source + 3-station model.
  Explains `check_parameters` as a cheap pre-run sanity check, previews the model (crust
  layers, velocity profile, source geometry, STF), runs `model.run(...)`, plots station
  `s1`'s response with `ZENTPlot`.
- **Required inputs**: None; must run cells in order (STF needs `.dt` set before
  `SourcePlot` samples it).
- **Outputs** (all confirmed present): `crust_layers.png`, `crust_velocity_profile.png`,
  `source_geometry.png`, `source_stf.png`, `engine_direct_s1.png` (final 3-component
  Z/E/N response of station `s1`).
- **Dependencies**: matplotlib, `shakermaker.tools.plotting` (`ZENTPlot`, `SourcePlot`).

---

## 06. Nearest Method

`examples/06_nearest_method/`

Documents the 3-stage OP/"nearest" pipeline (`gen_pairs` → `compute_gf` → `run_fast`,
i.e. Stage 0/1/2) both as one convenience call (`run_nearest(stage='all')`) and as
explicit separate calls, on identical toy models (SCEC_LOH_1 crust, 1 Gaussian source,
36-station `SurfaceGrid`). The notebook is the pedagogical piece, proving *why* the
method saves compute (geometric slot deduplication) using only the cheap Stage-0 output.
`legacy_migration.py` is a fourth, structurally different script — a format-migration
demo that is a no-op in a fresh checkout.

#### `nearest_all.py`
- **Purpose**: Smoke test of the OP pipeline in one call, `model.run_nearest(stage='all', ...)`,
  writing through `DRMHDF5StationListWriter`.
- **Outputs**: `out_nearest_all/gf_database_map.h5` + `out_nearest_all/gf_database_gf.h5`
  (Stage 0/1 database, root name passed and suffixed by the engine) and
  `out_nearest_all/surface.h5drm` (DRM writer output). Asserts only `h5drm_output` exists
  (not the two `_map.h5`/`_gf.h5` intermediates, unlike `stage_by_stage.py` below).
- **Dependencies**: `shakermaker.slw_extensions.DRMHDF5StationListWriter`,
  `shakermaker.sl_extensions.SurfaceGrid`.

#### `stage_by_stage.py`
- **Purpose**: Same model as `nearest_all.py`, but runs the pipeline manually:
  `model.gen_pairs(...)` (Stage 0, geometry only) → `model.compute_gf(...)` (Stage 1, FK
  evaluation per unique slot) → `model.run_fast(...)` (Stage 2, per-station
  convolution/assembly using precomputed GFs).
- **Outputs**: `out_stage_by_stage/gf_database_map.h5` (Stage 0, asserted),
  `out_stage_by_stage/gf_database_gf.h5` (Stage 1, asserted),
  `out_stage_by_stage/surface.h5drm` (Stage 2, asserted). This is the clearest, most
  explicit demonstration of the 3-stage file contract in the repo.

#### `legacy_migration.py`
- **Purpose**: Demonstrates `model.build_pair_to_slot_from_legacy_h5(legacy_db,
  delta_h=..., delta_v_rec=..., delta_v_src=...)`, which migrates an older-format
  ("JAA/PXP") GF database (expected datasets: `dh_of_pairs`, `zrec_of_pairs`,
  `zsrc_of_pairs`, `tdata_dict`) into the OP format by adding a `pair_to_slot` index.
- **Required inputs**: A pre-existing legacy database `legacy_gf_database.h5`, which the
  script explicitly checks for and does **not** ship. Prints `SKIP` and exits cleanly if
  absent — this is documented in-script behavior, not an oversight.
- **Outputs**: None in a fresh checkout (skip path). No sample/fixture legacy database
  exists anywhere in the repo, so this script is not runnable end-to-end as shipped.
- **Note**: per the method's docstring in `shakermaker/shakermaker.py`,
  `build_pair_to_slot_from_legacy_h5` writes its three new datasets (`pair_to_slot`,
  `nstations`, `nsources`) **into the same legacy database file, in place** — after the
  call, that same file is fully compatible with `run_fast`/`run_nearest(stage=2)`.

#### `notebooks/nearest_explained.ipynb`
- **Purpose**: Explains that in a horizontally layered crust, a Green's function depends
  only on `(d_h, z_src, z_rec)`, so geometrically-equivalent (source, receiver) pairs
  (within tolerances `delta_h`, `delta_v_src`, `delta_v_rec`) can share one computed GF
  ("slot"). Runs **only Stage 0** (`gen_pairs`) on the same 36-receiver `SurfaceGrid`,
  then visualizes which receivers got a freshly-computed slot vs. reused one.
- **Outputs** (confirmed present): the 4 standard preview plots (`crust_layers.png`,
  `crust_velocity_profile.png`, `source_geometry.png`, `source_stf.png`) plus
  `nearest_calc_vs_reuse.png` — a scatter of all 36 receivers colored orange
  ("computed fresh") vs. blue ("reused"), source epicenter marked with a red star, title
  reporting fresh/reused/total counts.
- **Documents the `<base>_map.h5` schema explicitly**: datasets `pair_to_slot
  (nstations*nsources,)`, `pairs_to_compute (n_slots,2)`, `dh_of_pairs`, `dv_of_pairs`,
  `zrec_of_pairs`, `zsrc_of_pairs`, plus scalars `delta_h`, `delta_v_rec`, `delta_v_src`,
  `nstations`, `nsources`.
- **Dependencies**: h5py, numpy, matplotlib.
- ⚠️ **Open question**: the output directories referenced by these scripts
  (`out_nearest_all/`, `out_stage_by_stage/`, `out_nearest_explained/`) are not present in
  the current repo tree. Not confirmed whether this is because they're `.gitignore`d
  runtime artifacts (most likely) or simply never committed — `.gitignore` content for
  these specific patterns was not checked.

---

## 07. Writers

`examples/07_writers/`

Covers ShakerMaker's output/persistence layer: the two `StationListWriter` classes
(`HDF5StationListWriter` for plain multi-station HDF5, `DRMHDF5StationListWriter` for
OpenSees-consumable `.h5drm`), both writer modes (legacy vs. progressive), the
geometry-only export path (`export_drm_geometry`, no FK run needed), generic HDF5
introspection, and the native `.npz` single-station save/load path.

#### `drm_writer.py`
- **Purpose**: Runs a small `DRMBox` (1×1×1 elements, dx=0.5 km, ≤50 stations) through
  the legacy `model.run(...)` path, writing progressively via
  `DRMHDF5StationListWriter`.
- **Required inputs**: None; guards with `try/except ImportError` on h5py, SKIPs cleanly
  if unavailable.
- **Outputs**: `drm_writer.h5drm` (confirmed present on disk — a leftover artifact from a
  prior real run, not just a fixture).
- **Dependencies**: h5py (guarded), `shakermaker.sl_extensions.DRMBox`,
  `shakermaker.slw_extensions.DRMHDF5StationListWriter`.

#### `hdf5_writer.py`
- **Purpose**: Runs a small 2-station model twice via `model.run(...)` with
  `HDF5StationListWriter`, once `writer_mode="legacy"`, once `writer_mode="progressive"`,
  to demonstrate both modes produce a valid file.
- **Outputs**: `hdf5_writer_legacy.h5` and `hdf5_writer_progressive.h5` — **neither is
  currently present on disk** (unlike `drm_writer.h5drm`), so these outputs are not
  currently sitting in the repo as committed/leftover artifacts.
- **Dependencies**: h5py (guarded), `shakermaker.slw_extensions.HDF5StationListWriter`.
- **Note**: per `shakermaker/slw_extensions/hdf5stationlistwriter.py`, both modes write
  the **same** on-disk schema (`/Data/xyz`, `/Data/internal`, `/Data/data_location`,
  `/Data/velocity` etc., `/Metadata/dt`/`tstart`/`tend`) — the only difference is *when*
  data hits disk: `"legacy"` accumulates every station in memory and flushes once at
  `close()`; `"progressive"` writes each station immediately as it finishes (lower peak
  memory, crash-resilient, visible progress on long runs).

#### `explore_h5_output.py`
- **Purpose**: Produces a `.h5drm` file **without running the FK engine at all**, via
  `model.export_drm_geometry(f_drm)` (geometry-only export) on a `DRMBox` (3×3×2
  elements, dx=0.5 km). Walks the file with `h5py.visititems` and prints every
  group/dataset name, shape, dtype.
- **Outputs**: `explore_geometry.h5drm` (confirmed present). Stdout: full tree dump (not
  captured statically — would need to run the script to see the exact printed schema).
- **Dependencies**: h5py (guarded).

#### `save_load_station.py`
- **Purpose**: Runs one station via `model.run(...)`, saves it via
  `station.save("sta.npz")` (native format, no h5py needed), reloads into a fresh
  `Station()` via `.load(...)`, asserts Z/E/N/t arrays are numerically identical
  (`np.allclose`) before/after.
- **Outputs**: `sta.npz` — not present on disk currently, consistent with being an
  ephemeral round-trip test artifact.
- **Dependencies**: numpy only (plus core classes) — no h5py needed for this format.

#### `notebooks/writers.ipynb`
- **Purpose**: The documented tutorial. Small 2-station model over SCEC_LOH_1 + 1
  Gaussian source, preview plots, `model.run(...)` with `HDF5StationListWriter`
  (`writer_mode="progressive"`), reopens the output with raw h5py, walks/prints its
  structure, plots station 0's velocity/acceleration/displacement traces read directly
  from the reopened HDF5 file (not the in-memory object) — explicitly demonstrating the
  round-trip-from-disk path.
- **Outputs** (confirmed present): the 4 standard preview PNGs plus
  `writers_station0.png` (3-panel: velocity/acceleration/displacement, E/N/Z overlaid).
  **Documented schema** (directly from the notebook): `/Data` group with datasets
  `velocity`, `acceleration`, `displacement`, each shaped `(3*nstations, nsamples)`, rows
  ordered `E, N, Z` per station, stations concatenated in order (station 0 = rows 0-2,
  station 1 = rows 3-5); `/Metadata` group with scalar dataset `dt`. The output file
  itself (`writers_demo.h5`) is **not** committed — only the plots are ("commit the plots,
  not the data" pattern seen across most notebooks in this repo).
- **Dependencies**: h5py, numpy, matplotlib.

**Cross-cutting note**: several output files/dirs referenced by scripts in `06_` and `07_`
are not present in the repo tree (`out_nearest_all/`, `out_stage_by_stage/`,
`hdf5_writer_legacy.h5`, `hdf5_writer_progressive.h5`, `sta.npz`). Most likely these are
gitignored run artifacts generated on first run, consistent with the general "commit the
plots, not the data" pattern — but `.gitignore` content was not checked to confirm this
for every specific filename.

---

## 08. DRM

`examples/08_drm/`

Demonstrates the DRM (Domain Reduction Method) receiver workflow end-to-end: building
`DRMBox`/`SurfaceGrid` receivers, exporting pure geometry with no FK run for fast
inspection, running the full OP pipeline through MPI to produce real `.h5drm` motions for
named stations, and a lightweight direct-vs-DRM sanity check.

#### `drm_loh1.py`
- **Purpose**: Runs the SCEC LOH.1 point-source model through the full OP pipeline
  (`run_nearest(stage='all')`) with `DRMBox` receivers at named station locations,
  writing `.h5drm` via `DRMHDF5StationListWriter` in `progressive` mode.
  It is the script that exposed the `/DRM_QA_Data` horizontal-component NaN at station
  `Centro` (zero epicentral distance), fixed in `subfk.f` by commit `2a83ca6`.
- **Required inputs**: MPI (`mpiexec -n N python drm_loh1.py`). No external files — crust
  (2-layer), source (Gaussian STF, `sigma=0.06`, `M0=1e18/5e14/2`), station coordinates
  (`utmx`/`utmy` for `Centro` and `s1`) are all inline. User-editable:
  `selected_stations` (currently `['Centro']`), DRM box geometry (`Lx_drm/Ly_drm/Lz_drm/
  dx_drm` in km), run parameters (`dt=0.005, nfft=4096, dk=0.1, tb=20, tmax=12`), and
  nearest-method tolerances (`delta_h, delta_v_rec, delta_v_src, npairs_max`).
- **Outputs**: per selected station, under `./drm_loh1_output/`: `drm_{name}_sta{idx}.h5drm`
  and `gf_db_{name}_sta{idx}_map.h5` + `gf_db_{name}_sta{idx}_gf.h5`. Console prints box
  geometry, node count, distance from source per station.
- **Dependencies**: `mpi4py`, numpy; the OP pipeline internally may use `numba` if
  installed.
- **Note**: `drm_loh1_output/` matches the `.gitignore` pattern
  `examples/**/drm_*_output/` — it is **not tracked by git**, and its git log is empty.
  The `drm_s1_sta1.h5drm`/`gf_db_s1_sta1_{gf,map}.h5` files sitting there locally don't
  correspond to any committed version of this script (every commit has
  `selected_stations = ['Centro']`) — they are local, untracked artifacts from a
  manually-edited local run, not part of the shipped repo.

#### `drm_vs_direct.py`
- **Purpose**: Regression/sanity check — runs a `Station` directly at a point and a
  1-node-equivalent tiny `DRMBox` at the same point (legacy `.run()` engine), asserts the
  QA station response matches the direct station response (nonzero, non-empty). Ends
  `PASS`.
- **Outputs**: None written. Stdout only.
- **Dependencies**: numpy (plus core classes). No MPI (uses legacy `.run()`).
- ⚠️ **Open question**: despite the filename "vs_direct," the assertions do **not**
  numerically compare the DRM QA response to the direct response against each other
  (`np.allclose`) — only that both are populated and nonzero. Whether this is intentional
  (e.g. a known small numerical difference) or an oversight is not stated in-code.

#### `export_drm_geometry.py`
- **Purpose**: Demonstrates `model.export_drm_geometry(...)` for two receiver types
  (`DRMBox`, `SurfaceGrid`) — geometry/coordinates only, no FK computation, near-instant.
- **Required inputs**: h5py (self-skips if unavailable).
- **Outputs**: `drm_geometry_box.h5drm`, `drm_geometry_surface.h5drm` (both confirmed
  present at `examples/08_drm/`, confirming this script has been run there before).
- **Dependencies**: h5py.

#### `notebooks/drm.ipynb`
- **Purpose**: Narrated walkthrough of `DRMBox` geometry export/visualization — builds a
  `DRMBox` (137 stations incl. QA) over SCEC_LOH_1, previews the model, calls
  `export_drm_geometry`, reopens the result with h5py, produces a 3D scatter colored by
  internal/external boundary flag.
- **Outputs** (all confirmed present): the 4 standard preview PNGs, `drm_geometry.h5drm`,
  `drm_geometry.png`. Saved cell output confirms 136 non-QA + 1 QA station, with
  `DRM_Data`, `DRM_Metadata`, `DRM_QA_Data` groups (shape `(408,2)` — the "2" is a 2-sample
  synthetic ramp, not real motion, consistent with "no FK run").
- **Dependencies**: h5py, matplotlib, numpy.

#### `drm_loh1_output/ShakerMakerResults.ipynb`
- **Purpose**: Post-processing/visualization notebook loading the `.h5drm` + GF database
  produced by `drm_loh1.py` for station `s1`, using the **separate**
  `ShakerMakerResults` package (external, see `shakermaker-results-skill` — not part of
  this repo) to plot node responses, domain geometry, GF connections, Newmark spectra,
  tensor Green's functions, and a manual STF-convolution sanity check.
- **Required inputs**: `drm_s1_sta1.h5drm`, `gf_db_s1_sta1_gf.h5`, `gf_db_s1_sta1_map.h5`
  (all present). Imports `LadrunoGraphStyle` and `ShakerMakerResults` — **neither package
  is part of this repo**; both are assumed to be on `PYTHONPATH` from elsewhere.
- **Outputs**: None to disk — all outputs are inline plots. Saved cell output confirms
  real data (8178 DRM nodes, 2400 time steps, `dt=0.005s`, domain 110×110×32.5 m, GF map
  reporting "99.76% fewer GF evaluations").
- **Dependencies**: `ShakerMakerResults`, `LadrunoGraphStyle` (both external to this
  repo), numpy, matplotlib.
- ⚠️ **Open questions**:
  1. This notebook cannot run using only this repo — it depends on two external, non-shipped packages.
  2. One cell contains just the literal text `ppp` with no output — appears to be leftover debug/scratch content; running it as-is would raise `NameError`.
  3. Two commented-out cells reference a hardcoded Windows path to an `ffmpeg.exe` binary — machine-specific, not runnable elsewhere, and inactive.

**Note**: the `/DRM_QA_Data` horizontal-component NaN reproduced by `drm_loh1.py` (station
`Centro`) was a 0/0 at zero epicentral distance in `subfk.f`, fixed by commit `2a83ca6`.

---

## 09. SW4 Export

`examples/09_sw4_export/`

Covers the ShakerMaker → SW4 → back-to-`.h5drm` round trip for OpenSees DRM consumption.
**SW4 itself is never invoked anywhere in this folder** — it is an external tool the user
must install and run separately; these scripts only write SW4 input files and, later,
read SW4's text output back in.

#### `export_sw4.py`
- **Purpose**: Exports a single-point-source model (real UTM station geometry, 4-layer
  crust, Gaussian STF) to an SW4-ready case via `model.export_sw4(...)` — writes SW4
  input files plus a compact transport `.h5` package. Does not run the FK core or SW4.
- **Required inputs**: None external — 12 named UTM stations (`Centro` = source) and
  crust layers are hardcoded. Passes `h=100.0` m, `size_domain=[40000,40000,25000]` m,
  `tmax=40.0` s to `export_sw4`.
- **Outputs**: `./_sw4_out/shakermakerexports/sw4_package.h5` (asserted to exist), plus
  SW4 `.in` input file(s) and per-source slip-rate files under `./_sw4_out/`. This output
  directory is **not currently present** in the repo — must be regenerated by running the
  script; three downstream scripts in this folder depend on it existing first.
- **Dependencies**: numpy; `shakermaker.export_sw4` internally may use h5py/pyvista.

#### `export_sw4_topo.py`
- **Purpose**: Same model as `export_sw4.py` but with real Cartesian topography via
  `model.export_sw4_topo(...)`, re-centering an absolute-UTM `.topo` file into the local
  km frame.
- **Required inputs**: A topography file (absolute UTM `x y z` rows, 500 m spacing) given
  by the environment variable `SHAKERMAKER_TOPO_FILE` (default
  `topography_h500_cartesian.topo` in the working directory). No such file ships with the
  repository: without one the script prints `SKIP` and exits.
- **Outputs** (only if the topo file is found): `./_sw4_out_topo/topography_h500_local.topo`
  and `./_sw4_out_topo/shakermakerexports/sw4_package.h5`.
- **Dependencies**: numpy.
- ⚠️ **Open question**: this example is not runnable as shipped outside the original
  author's machine, by design (self-skips). Should be documented explicitly as
  non-portable rather than as a working example.

#### `package_h5_roundtrip.py`
- **Purpose**: Exports the same model to SW4 (into `_sw4_out_roundtrip/`), then calls
  `unpack_sw4_package_h5(package, unpacked)` to unpack the compact `.h5` package back into
  a full SW4 file tree, asserts the unpacked input file exists.
- **Required inputs**: h5py (self-skips if missing). Fully self-contained otherwise.
- **Outputs**: `./_sw4_out_roundtrip/shakermakerexports/sw4_package.h5`, then
  `./_sw4_out_roundtrip/unpacked/sw4/shakermaker2sw4.in` (asserted) plus the rest of the
  unpacked SW4 tree.
- **Dependencies**: h5py.

#### `build_h5drm_from_sw4_case.py`
- **Purpose**: Library module (also a CLI via `argparse`) that reads a compact SW4
  transport package (`attrs['purpose'] == 'transport_unpack_to_sw4_files'`) plus SW4's own
  per-receiver `.txt` result files, and assembles a `.h5drm` with `DRM_Data`,
  `DRM_QA_Data`, `DRM_Metadata`, `Crust`, `SW4_Receivers` groups — velocity/displacement/
  acceleration derived from SW4's raw velocity via trapezoidal integration (displacement)
  and central differences (acceleration).
- **Required inputs**: `case_path` (an SW4 case directory with `sw4/` and
  `shakermakerexports/` subfolders); inside `sw4/<fileio_path>/`, one `.txt` file per
  receiver **produced by actually running SW4 externally**. **SW4 must be installed and
  run outside this repo** before this script does anything useful. Optional CLI flags:
  `--package-h5`, `--output-name` (default `motions.h5drm`), `--use-filter` (bandpass via
  ObsPy, lazily imported), `--freqmin/--freqmax/--corners/--no-zerophase`,
  `--move-2-shakermaker-coor`.
- **Outputs**: `<case_path>/shakermakerexports/<output_name>` (default `motions.h5drm`).
- **Dependencies**: h5py, numpy; ObsPy only if `--use-filter` is set.
- ⚠️ **Open questions**:
  1. If a per-receiver `.txt` file is missing, the code prints a `WARNING` and leaves that row as **zeros** rather than raising — worth being aware of, since it can silently produce a partially-zeroed `.h5drm` if SW4 did not finish writing all stations.
  2. The component-mapping comment (`E=SW4_Y, N=SW4_X, Z=SW4_Z(down+)`, stored as metadata `component_map`) could not be independently verified against SW4's own documentation (an external tool, out of scope here).

#### `build_h5drm_from_sw4.py`
- **Purpose**: Thin driver that calls `build_h5drm_from_sw4_case` against the case
  produced by `export_sw4.py` (`./_sw4_out/`), with explicit staged skip checks: (1) h5py
  installed, (2) `export_sw4.py` already run, (3) **SW4 already run externally** on that
  case. Prints `SKIP: ...` and exits cleanly if any prerequisite is missing — the cleanest,
  most machine-checkable statement in the repo of the "you must run SW4 yourself" boundary.
- **Outputs** (only if all prerequisites met): `./_sw4_out/shakermakerexports/motions.h5drm`.
- **Dependencies**: h5py.

#### `notebooks/sw4_export.ipynb`
- **Purpose**: Narrated version of `export_sw4.py` — same model, preview plots, geometry
  plots (map view + 3D), export to SW4, peek at the compact package's contents via h5py,
  optional re-export with `plot_geometry=True` if `pyvista` is available.
- **Outputs** (confirmed present): the 4 standard preview PNGs plus
  `sw4_geometry_map.png`, `sw4_geometry_3d.png`. Its own `./_sw4_out_nb/` export directory
  is not currently present on disk (likely cleaned up after a prior run, or never
  committed as a build artifact).
- **Dependencies**: h5py, matplotlib, numpy; pyvista optional.

---

## 10. FFSP

`examples/10_ffsp/`

Covers the FFSP stochastic finite-fault source end-to-end — running the Fortran kernel
(small/fast configs for smoke tests, a large 256×128 config for the "real" walkthrough),
the full product surface per realization (kinematics, quality/temporal/spectral metrics,
STF, spectrum, octave-averaged spectrum), and HDF5/legacy-format persistence + reload. It
also contains the only in-repo worked example of the manual FFSP → `FaultSource` bridge
(meters→km conversion, custom slip-rate synthesis, slip-threshold filtering) — this bridge
is hand-written independently in more than one notebook and is not exposed as a reusable
helper anywhere in the package.

#### `ffsp_run.py`
- **Purpose**: Smoke script — builds a small/fast FFSP stochastic finite-fault source and
  runs the Fortran kernel once (single realization). Asserts `subfaults is not None`.
- **Required inputs**: A 4-layer `CrustModel`, hardcoded inline. `FFSPSource` constructed
  with 27 explicit args, notably: `id_sf_type=8, freq_min=0.01, freq_max=24.0,
  fault_length=30.0, fault_width=16.0, x_hypc=15.0, y_hypc=8.0, depth_hypc=8.0,
  magnitude=6.0, fc_main_1=0.09, fc_main_2=3.0, rv_avg=3.0, ratio_rise=0.3, strike=358.0,
  dip=40.0, rake=113.0, pdip_max=15.0, prake_max=30.0, nsubx=16, nsuby=8,
  nb_taper_trbl=[5,5,5,5], seeds=[52,448,4446], id_ran1=1, id_ran2=1,
  angle_north_to_x=0.0, is_moment=3, output_name="FFSP_OUTPUT"`. Self-contained, no CLI
  args.
- **Outputs**: None written to disk by this script (no `write_hdf5`/`write_ffsp_format`
  call) — only the in-memory `subfaults` dict and `PASS`.
- **Dependencies**: core shakermaker only.
- **Note**: an in-code comment says "same crust as `example8_ffsp.py`" — confirmed stale.
  Git history shows the pre-reorganization flat layout had `examples/example7_ffsp.py`
  (not `example8`) as the FFSP example; this is an off-by-one leftover from the 2026-06-07
  reorganization into numbered topic folders. Kept as-is in the source; documented here
  for clarity.

#### `ffsp_io.py`
- **Purpose**: Round-trip smoke test — runs the same small FFSP source as `ffsp_run.py`,
  writes it to HDF5 and to the legacy FFSP text format, reloads via
  `FFSPSource.from_hdf5`, asserts the reloaded `best_realization` (`npts`, `slip`,
  `rupture_time`) matches the in-memory one.
- **Required inputs**: Identical crust/`FFSPSource` params as `ffsp_run.py` (same stale
  `example8_ffsp.py` comment). Requires h5py — soft `try/except ImportError` guard,
  prints `SKIP` and exits cleanly if missing.
- **Outputs**: `ffsp.h5` (via `write_hdf5`, written to CWD) and a full legacy-format
  directory `FFSP_OUTPUT/` (via `write_ffsp_format`). **Confirmed on disk**:
  `examples/10_ffsp/FFSP_OUTPUT/` exists with exactly the files `write_ffsp_format`
  produces: `ffsp.inp`, `FFSP_OUTPUT.001`, `FFSP_OUTPUT.bst`, `source_model.score`,
  `source_model.list`, `source_model.params`, `velocity.vel`, `calsvf.dat`,
  `calsvf_tim.dat`, `logsvf.dat`. Direct inspection of `FFSP_OUTPUT/ffsp.inp` confirms
  `magnitude=6.0, nsubx nsuby = 16 8` — i.e. this checked-in directory was produced by
  `ffsp_run.py`/`ffsp_io.py`'s small config, not by `example_FFSP.ipynb`'s large config.
  Also present, loose, directly under `examples/10_ffsp/`: `calsvf.dat`,
  `calsvf_tim.dat`, `logsvf.dat` — these appear to be working-directory side effects the
  Fortran kernel itself always drops in CWD, separate from the copies `write_ffsp_format`
  places inside `FFSP_OUTPUT/`.
- **Dependencies**: h5py (soft-guarded), numpy.
- ⚠️ **Open question**: `ffsp.h5` itself is not present on disk in this checkout, so the
  round-trip assertions could not be independently confirmed to currently pass without
  executing the (slow) Fortran kernel.

#### `notebooks/example_FFSP.ipynb`
- **Purpose**: The larger, non-toy, end-to-end walkthrough — builds a 3-layer crust, runs
  a **large** FFSP source (`nsubx=256, nsuby=128`; a commented-out `nsubx=32, nsuby=32`
  alternative is present but inactive), inspects it with histogram/spatial/quality/
  temporal/spectral plots, hand-builds a `FaultSource` from `get_subfaults()` using a
  **custom local `srf2()` function** (not `shakermaker.stf_extensions.SRF2`), runs a full
  `ShakerMaker.run()` (legacy path) against a single station, and post-processes
  velocity → displacement/acceleration by hand.
- **Required inputs**: 3-layer `CrustModel`. `FFSPSource` with `magnitude=6.5,
  nsubx=256, nsuby=128, id_ran1=1, id_ran2=1`, no `output_name` passed (falls back to the
  constructor default `"FFSP_OUTPUT"`). Manual bridge parameters, hardcoded and not
  exposed as function args: `pt_rt = 0.15` (fraction of rise time used as peak time
  `Tp`), `Te = 0.7 * Tr`, **`MINSLIP = 1.9217`** (a hardcoded slip threshold used to filter
  subfaults before converting to `PointSource`s — a practical runtime limiter, not a
  physically-derived value: it exists purely to avoid turning every one of the 256×128
  subfaults into a `PointSource`/legacy `run()` call, which would be far too slow for an
  example; it keeps only a small subset of the highest-slip subfaults).
  **Unit handling confirmed explicit**: the notebook itself divides FFSP's raw x/y/z by
  1000 (`xsrc = x[i] / 1e3`, etc.) to convert meters to ShakerMaker's km, consistent with
  the known FFSP-returns-meters gotcha. Run config (legacy `model.run(...)`):
  `dt=0.0025, nfft=8192*2, dk=0.2, tb=0, tmin=0., tmax=150., smth=1, sigma=2`. Single
  station at `[6.0, 8.0, 0.0]` km, `"Station_01"`.
- **Outputs**: No `.png` files are explicitly saved via `savefig` in this notebook (the
  6 `plot_*` calls render inline without a save step). `station.save("sta01.npz")` writes
  an `.npz` to whatever the kernel's CWD was — see the note below on `sta01/02/03.npz`
  provenance. Plot methods used: `plot_histogram` (fields `slip`, `rake`),
  `plot_spacial_distribution` (default, then `field='slip'` and `field='rake'` with
  `rotate=True, show_contours=True, contour_field='rupture_time',
  show_hypocenter=True`), `plot_quality_metrics`, `plot_temporal_metrics`,
  `plot_spectral_comparison`, `plot_source_time_function` — 6 of the 9 documented
  `FFSPSource.plot_*` methods (all except `plot_rupture_snapshot`). Also two hand-rolled
  (non-`plot_*`) matplotlib figures: per-subfault `s(t)`/`ṡ(t)`/`s̈(t)` and Z/E/N velocity
  and acceleration time histories built manually rather than via `ZENTPlot`.
- **Dependencies**: `scipy.integrate.cumulative_trapezoid`, numpy, matplotlib (no
  `Agg` backend set — assumes an interactive/inline backend).
- ⚠️ **Open questions**:
  1. No saved cell outputs are present, so none of the notebook's claimed numeric results (Mw, subfault counts, filtering counts) could be independently verified.
  2. The local `srf2()` function is a separate reimplementation from `shakermaker.stf_extensions.SRF2`; whether the two are mathematically equivalent was not checked.
  3. A variable named `slip_max` is actually assigned `slip.mean()` (the mean, not the max) when computing `M0`/`Mw`, which also uses a fixed `ρ=2500, Vs=3140` independent of the actual layered crust — reported as-is, not corrected.
  4. **`sta01.npz`/`sta02.npz`/`sta03.npz` provenance — now resolved** (see `12_validation` below): these three untracked files sitting in `examples/12_validation/notebooks/` are confirmed to come from `examples/12_validation/notebooks/example_LOH1_gf.ipynb`, **not** from this notebook (this notebook's `station.save("sta01.npz")` call was initially flagged as a possible source, but the LOH1-folder notebook is the actual, confirmed producer — see below).

#### `notebooks/ffsp.ipynb`
- **Purpose**: Documented, saved-output notebook — a clean walkthrough of the
  small/fast FFSP source (same 16×8 params as `ffsp_run.py`/`ffsp_io.py`) with crust
  preview, run, spatial-distribution plot of slip (contours + hypocenter), and STF plot.
  The "example" version of `ffsp_run.py`, written for documentation rather than CI.
- **Required inputs**: Same 4-layer crust and 27-arg `FFSPSource` call as `ffsp_run.py`/
  `ffsp_io.py`. Fully self-contained.
- **Outputs** (all confirmed present): `crust_layers.png`, `crust_velocity_profile.png`,
  `ffsp_slip_distribution.png` (via `plot_spacial_distribution(field="slip",
  show_contours=True, show_hypocenter=True)` then manual `savefig`),
  `ffsp_source_time_function.png` (via `plot_source_time_function()` then manual
  `savefig`). A markdown cell notes `plot_spacial_distribution` also supports writing its
  own PNG directly via `save_fig=True`/`model_name=`, but this notebook always uses a
  manual `savefig` instead, "to guarantee a .png in this folder."
- **Dependencies**: matplotlib.

#### `notebooks/ffsp_all_realization_products.ipynb`
- **Purpose**: A verification/regression notebook proving the full multi-realization
  product set — kinematics, quality/temporal/spectral metrics, full STF, synthetic
  spectrum, octave-averaged spectrum — is available for **every** realization (not just
  the best one), that any realization can be selected without recomputation, and that
  HDF5 round-trips this whole structure losslessly.
- **Required inputs**: Same 4-layer crust as the other FFSP notebooks. `FFSPSource` with
  `magnitude=6.0, nsubx=16, nsuby=8` (matching the small configs) but **`id_ran1=1,
  id_ran2=3`** (3 realizations), `output_name="FFSP_ALL_PRODUCTS"`, `verbose=False`.
- **Outputs**: No PNGs (an inline comparison plot of STF/spectrum by realization has no
  `savefig`). Writes a temporary HDF5 file inside a `tempfile.TemporaryDirectory()`
  (auto-deleted). Captured output shows this notebook was last run **on a Windows
  machine** (temp path `C:\Users\ppala\AppData\Local\Temp\...`). Asserts exact shapes:
  kinematics `(128,3)` (16×8 subfaults × 3 realizations), STF `(131072,3)`, spectrum
  `(65536,3)`, octaves `(16,3)`; asserts `best_index=1` (0-based) ↔ `best_realization_id=2`;
  full round-trip equality check of `stf_time.stf` and `spectrum.moment_rate_synth`
  across all 3 realizations between the in-memory object and the HDF5-reloaded one.
- **Dependencies**: numpy, matplotlib, `tempfile`, `pathlib.Path`. Notably path-hacks
  `sys.path.insert(0, repo_root)` via `Path.cwd().resolve().parents[2]` — i.e. it does
  **not** rely on `shakermaker` being `pip install`ed; it assumes the notebook is run
  from inside `examples/10_ffsp/notebooks/` (3 levels below repo root).
- ⚠️ **Open question**: the `parents[2]` path-hack is brittle — it will silently resolve
  to the wrong directory if this notebook is ever run from a different working directory
  or moved. Not confirmed to have caused an actual failure, just a fragility worth
  knowing about.

---

## 11. Plotting

`examples/11_plotting/`

Narrow scope: the three generic (non-FFSP) plotting helpers in
`shakermaker.tools.plotting` (`SourcePlot`, `StationPlot`, `ZENTPlot`), demonstrated first
on static geometry (no simulation needed) and then, for `ZENTPlot`, against a real
minimal FK run. Also documents a real, current SciPy compatibility issue and its
workaround.

#### `plotting_tools.py`
- **Purpose**: Smoke script — builds two `PointSource`s (no FK run, no crust needed)
  sharing one `Gaussian` STF, wraps them in a `FaultSource`, saves a `SourcePlot`.
- **Required inputs**: `Gaussian(t0=0.36, freq=16.6667, M0=1.0)`; two `PointSource`s at
  `[0,0,2.0]` and `[0,0,2.5]` km, strike/dip/rake `[0., 90., 0.]`. Uses
  `matplotlib.use("Agg")` before importing pyplot (headless-safe).
- **Outputs**: `sources_plot.png`, written next to the script (path-anchored via
  `os.path.dirname(os.path.abspath(__file__))`, not CWD-dependent). Confirmed present.
- **Dependencies**: matplotlib (Agg backend).

#### `notebooks/plotting_tools.ipynb`
- **Purpose**: Documented walkthrough of the three plotting helpers. Runs one tiny real
  FK simulation (LOH.1 crust) specifically so `ZENTPlot` has real data to show.
- **Required inputs**: The same (now unnecessary — see `02_sources/notebooks/sources.ipynb`
  above) SciPy `trapz`/`trapezoid` compatibility shim (`scipy.integrate.trapz = np.trapz`).
  Crust:
  `SCEC_LOH_1()` (packaged preset). Source: single `PointSource([0,0,2], [0.,90.,0.],
  stf=Gaussian(t0=0.36, freq=16.6667, M0=1.0))`. Station: single `Station([6,8,0],
  metadata={'name':'surface_rcv'})`. Run parameters (legacy `model.run(...)`, after
  `check_parameters(...)`): `dt=0.05, nfft=1024, dk=0.2, tb=1000, tmax=20`. Sets `dt=0.05`
  on the source's STF before plotting so `SourcePlot`'s STF-based coloring has
  discretized data.
- **Outputs** (all confirmed present): `crust_layers.png`, `crust_velocity_profile.png`,
  `source_geometry.png` (`SourcePlot(fault, show=False)`), `source_stf.png` (hand-rolled,
  not a packaged method), `sources_plot.png` (a second, differently-configured
  `SourcePlot(fault, show=False, colorbar=True)` — deliberately different from
  `source_geometry.png`), `stations_plot.png` (`StationPlot`), `zent_plot.png`
  (`ZENTPlot`, after the run cell populates the station's response).
- **Dependencies**: numpy, matplotlib, scipy.integrate (patched).
- **Note**: `examples/11_plotting/sources_plot.png` exists both at the top level of
  `11_plotting/` (written by `plotting_tools.py`) and inside `notebooks/` (written by this
  notebook) — same filename, two different producers in two different directories, not a
  conflict.

---

## 12. Validation

`examples/12_validation/`

The SCEC LOH.1 and LOH.3 benchmarks — **the repo's confirmed physics regression gates**,
comparing ShakerMaker's FK output against a semi-analytical reference solution
("Prose") via component-wise cross-correlation (pass threshold 0.9). Contains a runnable
script pair (`LOH1.py` → `LOH1_check.py`, sequentially dependent) plus three notebooks of
increasing depth. This is the "slow" directory in the smoke runner (`SLOW_DIRS`,
skipped unless `--full`) because `LOH1.py` runs a full, non-OP FK synthesis — the
correlation check itself is cheap.

#### `LOH1.py`
- **Purpose**: Runs the SCEC LOH.1 benchmark (buried double-couple source, single
  surface receiver at (6,8,0) km) via the classic FK engine (`model.run(...)`, not OP),
  saves the response.
- **Required inputs**: None — all hardcoded (`dt=0.025, nfft=4096, dk=0.1, tb=1000,
  tmax=nfft*dt=102.4`). No MPI (single station, serial). Must be run with CWD =
  `examples/12_validation/` since it writes `loh1_station.npz` as a bare relative path.
- **Outputs**: `loh1_station.npz` (via `sta.save(...)`). Console `PASS` if
  `len(t)>0` and array lengths match — a weak internal check, not the physics check
  (that's `LOH1_check.py`'s job).
- **Dependencies**: core shakermaker only.
- **Note**: `check_parameters` is called but its result is only printed, not used to gate
  execution — see the cross-cutting note below; this is by design (see the appendix).

#### `LOH1_check.py`
- **Purpose**: Loads the analytical reference (`data/LOH.1_prose3`) and the ShakerMaker
  output from `LOH1.py`, convolves the reference with the LOH.1 source ramp + Gaussian
  STF (reproducing the SCEC LOH.1 recipe), rotates ShakerMaker's (Z,E,N) into (radial,
  transverse, vertical) using the fixed source→receiver azimuth (unit vector (0.6, 0.8),
  hardcoded for the (0,0)→(6,8) geometry), interpolates onto the reference time grid,
  computes zero-mean normalized cross-correlation per component.
- **Required inputs**: `loh1_station.npz` must already exist (produced by `LOH1.py` —
  the script explicitly checks and prints `SKIP: run LOH1.py first` + exits if missing:
  **hard sequential dependency**). Reference file `data/LOH.1_prose3` (present — 2048-row
  ASCII table, 4 columns: time, vertical×(-1e5), radial×(1e5), transverse×(1e5), no
  header, no unit annotation in-file).
- **Outputs**: Console printout of correlation per component (radial/transverse/vertical)
  and `PASS`/`AssertionError`. Pass criterion: `min(corrs) > 0.9` across all three
  components. No files written.
- **Dependencies**: numpy only.
- **Provenance**: `data/LOH.1_prose3` originates from SW4's own LOH.1 validation
  materials (confirmed by the repo maintainer) — it is not a ShakerMaker-generated file.

#### `notebooks/LOH1_validation.ipynb`
- **Purpose**: Notebook version of the `LOH1.py` → `LOH1_check.py` pipeline, but
  self-contained and richer: builds the model, previews it, runs it, inline-reproduces
  the Prose-reference convolution+rotation+correlation logic, plots an overlay figure.
- **Outputs** (all confirmed present, currently modified per `git status`): the 4
  standard preview PNGs plus `LOH1_validation.png` (final 3-panel Prose-vs-ShakerMaker
  overlay with correlation values in the legend).
- **Dependencies**: matplotlib, `shakermaker.tools.plotting.SourcePlot`.
- **Note**: the captured `check_parameters` cell output reports `RESULT: 1 error(s) --
  fix before running / tmax 102.4 -> end of record is 79.1s (raise nfft or lower tmax)`,
  yet the next cell runs `model.run(...)` anyway with the same not-recommended
  parameters and completes without visible failure. This is consistent with
  `check_parameters` being advisory-only by design (see appendix) — the correlation check
  only inspects `t ∈ [0,9]` (per the plot's `xlim`), well within the valid window.

#### `notebooks/LOH1_greens_functions.ipynb`
- **Purpose**: Opens up the internals of `ShakerMaker.run()` on the same LOH.1 crust with
  **three** receivers ((8,8,0), (6,8,0), (4,4,0.5) km), using `save_gf=True` to retain
  each station's raw Green's functions. Extracts the 9-element `tdata` tensor
  (Helmberger DD/DS/SS elementary GFs), the pre-STF recombined (Z,E,N) impulse response,
  manually convolves with the Gaussian STF (mirroring what `run()` does internally),
  cross-checks the manual result against `get_response()`.
- **Required inputs**: None external; `dt=0.005, nfft=4096, dk=0.025, tb=20`. No MPI (3
  stations, serial, ~5.7s per captured log).
- **Outputs** (all confirmed present, currently modified per `git status`):
  `loh1_crust_layers.png`, `loh1_velocity_profile.png`, `loh1_stf.png`,
  `loh1_responses.png` (3×3 grid: Z/E/N × 3 stations), `loh1_gf_tensor.png` (9 elementary
  GFs for station 1), `loh1_gf_components.png` (recombined Z/E/N impulse response before
  STF), `loh1_convolution_check.png` (manual-convolution vs. `run()` overlay), plus a
  printed correlation number whose actual value is **not preserved** in the notebook JSON
  ("Outputs are too large to include" placeholder).
- **Dependencies**: numpy, matplotlib.
- **Note**: same advisory-only `check_parameters` pattern (captured output shows
  `RESULT: 2 error(s)` for `tb` and `tmax`, run proceeds regardless — by design, see
  appendix). The final numeric correlation result could not be read from the saved
  notebook (output not preserved in the JSON).

#### `notebooks/example_LOH1_gf.ipynb`
- **Purpose**: An earlier, less-documented near-duplicate of `LOH1_greens_functions.ipynb`
  — same 3 stations (`sta01`/`sta02`/`sta03`) at the same coordinates, same crust, same
  Gaussian STF, same GF-tensor and manual-convolution logic, but without markdown
  narration, and additionally **saves each station to `.npz`** (`s1.save("sta01.npz")`,
  etc.), and shows the full verbose per-pair engine debug log (which the newer notebook
  suppresses).
- **Outputs**: `sta01.npz`, `sta02.npz`, `sta03.npz` — **confirmed** (by filename and
  matching station coordinates) to be the exact three files listed as untracked in this
  session's `git status`, since `LOH1_greens_functions.ipynb` never calls `.save()`. This
  resolves the open question raised while reviewing `10_ffsp/notebooks/example_FFSP.ipynb`
  (which also writes a file named `sta01.npz` but from a different, unrelated run/folder).
- **Note**: both notebooks were added together in the same commit (`9859844`,
  2026-07-20); a same-day follow-up commit (`bbd3593`) only fixed a stray UTF-8 BOM/
  Spanish comment in this file. Git history shows no "replace" relationship — both are
  intentionally kept: `LOH1_greens_functions.ipynb` as the narrated/documented version,
  `example_LOH1_gf.ipynb` as the lower-level version with full debug output and `.npz`
  side-saves.

#### `data/LOH.1_prose3`
- Not a script — the reference data table consumed by `LOH1_check.py` and
  `LOH1_validation.ipynb`. See open question on provenance above.

---

### LOH.3 (added 2026-09-26)

LOH.3 is LOH.1 **plus anelastic attenuation**: same crust, same buried double-couple,
same receiver at (6,8,0) km, but `Q_S = Vs[m/s]/50` and `Q_P = (3/4)(Vp/Vs)² Q_S`
(⇒ layer `Qp=120, Qs=40`; half-space `Qp=155.9, Qs=69.3`), and `sigma = 0.05` s instead
of 0.06 s. Where LOH.1 gates the elastodynamics, LOH.3 gates the **attenuation model**.

**The finding this exercise exists to expose.** The FK core carries Q through the
Futterman complex velocity of Aki & Richards p.182 (`shakermaker/core/subfk.f:92`),
`c(f) = c0 [1 + ln(f)/(πQ) + i/(2Q)]`, whose logarithm is anchored at **1 Hz**. The
benchmark anchors its phase velocities at **2.5 Hz** (`attenuation phasefreq=2.5` in the
SW4 deck). Taken as specified the correlation is capped at **~0.95** by a ~16 ms
travel-time bias that *no `dt` refinement removes*; re-anchoring the dispersion at 2.5 Hz
takes it to **~0.999** with zero lag. Both scripts run both models so the difference is
visible rather than assumed.

**Two prerequisites were fixed to make this work** (see the notes at the end of this
section): the wrong Q values in `SCEC_LOH_3`, and the stale-install import hazard.

#### `LOH3.py`
- **Purpose**: Runs SCEC LOH.3 twice — as specified, and with the Q dispersion
  re-anchored at `F_REF = 2.5` Hz via the module-level helper `scec_loh_3_qref()`
  (divides each speed by `1 + ln(f_ref)/(πQ)`, leaving Q untouched).
- **Required inputs**: None — all hardcoded (`dt=0.004, nfft=8192, dk=0.1, tb=1000,
  tmax=16.0`). `dt` is set by the **frequency band**, not the output rate: `fk.f:73`
  tapers from `0.1·f_Nyq` to zero at `f_Nyq`, so the flat band is `f ≤ 0.05/dt = 12.5` Hz,
  against a 3.2 Hz Gaussian corner. Unlike `LOH1.py`, `check_parameters` reports
  **`RESULT: all hard checks passed`** for these values. No MPI, ~1.5 s per model.
- **Outputs**: `loh3_station.npz`, `loh3_station_qref.npz`, plus the package-provenance
  line and the crust table on stdout.
- **CWD**: writes bare relative paths, so run with CWD = `examples/12_validation/`. The
  import bootstrap at the top is CWD-independent (uses `__file__`).

#### `LOH3_check.py`
- **Purpose**: Same role as `LOH1_check.py` but for both LOH.3 runs, and with a wider
  metric set: correlation, peak ratio, **least-squares amplitude scale**, **normalized L2
  misfit**, and **correlation after the best integer time shift** (which is what reveals
  the 16 ms bias). The reference convolution is a line-for-line transcription of
  `loh3exact.m`, identical to the LOH.1 recipe with `sigma = 0.05`.
- **Required inputs**: both `.npz` from `LOH3.py` (prints `SKIP` per file if missing),
  and `data/LOH.3_prose_corrected`.
- **Pass criterion**: `min(corr) > 0.94` as specified **and** `min(corr) > 0.99`
  re-anchored — i.e. the test asserts that re-anchoring actually fixes the waveform,
  not merely that some number is high.
- **Measured** (2026-09-26): as specified R/T/V = 0.9535 / 0.9528 / 0.9754 with best lag
  −0.016 s; re-anchored 0.9994 / 0.9997 / 0.9996 at zero lag, L2 misfit 4.3–5.2%,
  amplitude +3.3%.

#### `notebooks/LOH3_validation.ipynb`
- **Purpose**: The LOH.3 counterpart of `LOH1_validation.ipynb`, plus the three figures
  that carry the Q-reference-frequency argument.
- **Outputs**: `loh3_crust_layers.png`, `loh3_velocity_profile.png`,
  `loh3_source_geometry.png`, `loh3_source_stf.png`, `LOH3_validation.png` (3-panel
  overlay: Prose vs both models), `loh3_qref_zoom.png` (S arrival, where the 16 ms is
  visible by eye), `loh3_qref_sweep.png` (**correlation vs `f_ref` for 1–5 Hz — peaks at
  2–2.5 Hz and nowhere else**, a cheap refutation test of the whole explanation), and
  `loh3_vs_loh1.png` (what Q did, with Q as the *only* difference: peaks drop to
  **0.57–0.64** of the elastic case and the arrival advances 20–28 ms).
- **Trap this figure was built to avoid**: running each benchmark with its own `sigma`
  (0.06 vs 0.05) moves the onset by 60 ms — `t0 = 6·sigma` — and changes peak velocity
  by ~1.2–1.4×, which *masks* the attenuation and reads as a spurious 0.74–0.84 ratio.
  The cell therefore drives both crusts with the same `sigma = 0.05` STF and compares
  against LOH.3 **as specified** (same 1 Hz anchor), and measures the residual shift
  rather than leaving it to the eye. That residual is real: causal constant-Q comes
  with dispersion, so above the 1 Hz anchor the attenuated medium is faster and the
  LOH.3 trace arrives early.
- **Runtime**: ~10 runs × ~12 s under the notebook kernel.

#### `notebooks/LOH3_greens_functions.ipynb`
- **Purpose**: The LOH.3 counterpart of `LOH1_greens_functions.ipynb` — same three
  receivers, same `save_gf=True` dissection (9-element `tdata`, pre-STF impulse response,
  manual convolution, cross-check against `get_response()`: correlation 0.9985) — plus a
  final section that **isolates the attenuation operator**: it reruns the identical model
  with `Q = 10000` and takes the LOH.1/LOH.3 spectral ratio, which must decay as
  `exp(-π f t*)`.
- **Measured**: the impulse response loses far more than the seismogram does -- peak
  ratios `0.171` (Z) and `0.080` (E/N), against ~0.6 on the seismograms -- because the
  raw Green's function carries the whole 12.5 Hz band while the `sigma = 0.05` s
  Gaussian keeps only what is below ~8 Hz. `loh3_gf_vs_loh1.png` therefore uses one
  column per model with independent vertical scales: on a shared axis the attenuated
  trace is simply invisible.
- **Measured**: effective `t* = 71 ms` from the smoothed ratio (72.5 ms unsmoothed),
  against a straight-ray bracket of 95.7 ms. The raw ratio is spiky (spectral nulls where
  the elastic GF nearly vanishes) and both the raw and smoothed curves are plotted, so the
  scatter is not hidden.
- **Outputs**: `loh3_gf_crust_layers.png`, `loh3_gf_velocity_profile.png`, `loh3_stf.png`,
  `loh3_responses.png`, `loh3_gf_tensor.png`, `loh3_gf_components.png`,
  `loh3_seismograms_manual.png`, `loh3_convolution_check.png`, `loh3_gf_vs_loh1.png`,
  `loh3_gf_tensor_elastic.png`, `loh3_tstar.png`.
- **Why the `tdata` figure looks unlike the LOH.1 one**: the plotting code is
  byte-identical (verified by diff) on the same `tdata[0, i*3+j, :]` array. Two
  things differ. (a) Q: attenuation kills the later multiples and low-passes each
  arrival, so LOH.1's picket fence of spikes becomes a few rounded pulses plus a
  smooth tail that *looks* like a velocity seismogram although nothing was
  convolved. `loh3_gf_tensor_elastic.png` is the control that isolates this: same
  code, same `dt/tb/dk`, `Q = 10000` — and the picket fence comes back. The
  spectral centroid drops from ~7 Hz to ~5 Hz (×0.65–0.70). (b) Different FK
  parameters between the two notebooks: LOH.1's uses `dt=0.005, nfft=4096, tb=20,
  dk=0.025`, this one `dt=0.004, nfft=8192, tb=250, dk=0.05`. `tb` is padding **in
  samples**, so the reduction time `t0` differs and the trace starts at ~2.0 s
  there and ~1.08 s here. `check_parameters` flags LOH.1's `tb=20` as below its
  sane range (≥250), which is why it was not copied. Impulse-response amplitudes
  are consequently comparable only *within* a notebook, not across the two.

#### `data/LOH.3_prose_corrected`
- The semi-analytical reference, copied from the SW4 matlab tools
  alongside its generating script `loh3exact.m`.
  Same format as `LOH.1_prose3`: 2048 rows, `dt = 0.008` s, columns
  `time, vertical×(-1e5), radial×(1e5), transverse×(1e5)`, no header. Same provenance
  chain as the LOH.1 table.

**Note 1 — `SCEC_LOH_3` carried wrong Q until 2026-09-26.** It had layer
`Qa=54.65, Qb=137.95` and half-space `Qa=69.3, Qb=120.` Those are the values of SW4's
*interface-averaging* block — the arithmetic means `(40+69.3)/2` and `(120+155.9)/2` used
to smear the material discontinuity over one grid cell — and on top of that with Qp and Qs
swapped. Anything run with the old values was not LOH.3. Fixed in
`shakermaker/cm_library/LOH.py` with the derivation in the docstring.

**Note 2 — the examples now pin the import to the working tree.** `LOH3.py`,
`LOH3_check.py` and both LOH.3 notebooks start by prepending the repo root to
`sys.path` and printing `shakermaker.__file__`. ShakerMaker is installed **non-editable**
in some development environments, so a plain `import shakermaker` from `examples/12_validation/`
(or from `notebooks/`) resolves to a frozen `site-packages` snapshot whose
`cm_library/LOH.py`, `shakermaker.py` and compiled `core` are all older than the working
tree. This bit during development: the first executed run of `LOH3_validation.ipynb`
silently used the **old, wrong** Q values and produced plausible-looking but wrong
numbers. The LOH.1 scripts and notebooks in this folder do **not** have this guard and
are presumably affected the same way.

**Cross-cutting note for this folder**: `check_parameters` in this repo is **advisory
only** — it prints a recommendation/error report but does not block or alter the
subsequent `run()` call. Every LOH1 notebook in this folder that captures the
`check_parameters` output shows it reporting hard errors, and every one of them proceeds
to run anyway with the un-recommended parameters. Whether this is an accepted pattern for
these particular examples (e.g., because the correlation check only inspects an early
time window unaffected by the flagged issue) or something that should be called out as a
caveat is not something this review can determine from static code alone.

---

## 13. ShakerMaker SW4

`examples/13_shakermaker_sw4/`

Cross-validates the semi-analytic FK method against a real SW4 finite-difference run on
an identical 100-source model, band-passed to a common 0.25–15 Hz band for a fair
comparison. Well-documented via its own `README.md`.

#### `shaker_vs_sw4.py`
- **Purpose**: Rebuilds a ShakerMaker model (4-layer crust, 100 point sources with
  `Discrete` slip-rate STFs, 2 stations) entirely from a stored HDF5 package
  (`data/model_summary.h5`), runs the FK engine, reads two SW4 receiver text outputs,
  band-passes them with ObsPy to match the FK usable band, overlays FK vs. SW4 per
  component per station.
- **Required inputs**: `data/model_summary.h5` (present — confirmed by direct HDF5
  inspection to contain a 4-layer crust + 100 sources + 2 stations), `data/sf00001.txt`
  and `data/sf00002.txt` (SW4 receiver outputs, ASCII, 13-header-row format, columns
  time/x/y/z — both present). No user-editable parameters — `dt=0.0025, nfft=32768,
  tb=800, tmax=60, FMIN/FMAX=0.25/15 Hz` are all hardcoded. Documented as heavy
  (`nfft=32768 × 100 sources`); both the script's own comments and the folder's README
  recommend `mpiexec -n 8 python shaker_vs_sw4.py`, though it also runs serially.
- **Outputs**: `compare_sf00001.png`, `compare_sf00002.png` — written to the script's own
  directory (`examples/13_shakermaker_sw4/`, **not** `notebooks/`). Neither file is
  currently present in this repo checkout, but the script **has been run to completion**
  (confirmed) — the PNGs simply weren't committed here, unlike the notebook's equivalent
  outputs (see below).
- **Dependencies**: h5py, numpy, matplotlib, **ObsPy** (only for `bandpass()` — the
  README explicitly calls out `pip install obspy` as a non-default dependency),
  `shakermaker.stf_extensions.Discrete`.
- **Note**: `data/model_summary.h5` is called a "compact SW4 export package" by this
  folder's README, but its internal structure matches the full `sw4_exporter` bundle
  format (`files/entries/...`, `manifest/`, `sw4_input/text`, etc. — see
  `shakermaker/sw4_exporter/README.md`, which documents the bundle simply as "a single
  HDF5 transport bundle," with no separate reduced format). "Compact" here is just
  informal phrasing for "one summary file," not a technical claim of a smaller format.

#### `notebooks/shaker_vs_sw4.ipynb`
- **Purpose**: Step-by-step notebook version of the same cross-validation (run from
  `notebooks/`, paths point to `../data`), broken into 4 documented steps: (1) rebuild
  model from package + plot crust/sources/station map, (2) run FK engine, (3) read +
  band-pass SW4 output, (4) overlay + compare. Only compares station 1
  (`sf00001`) — station 2 is only handled by the standalone script (whose own output is
  not currently present, as noted above).
- **Required inputs**: Same `data/model_summary.h5`, needs ObsPy.
- **Outputs** (all confirmed present, tracked in git): `sw4_crust_layers.png`,
  `sw4_source_stfs.png`, `sw4_stations_map.png`, `shaker_seismogram.png`,
  `sw4_velocity_raw.png`, `sw4_velocity_filtered.png`, `sw4_vs_shaker_comparison.png`.
  Captured `check_parameters` output shows `RESULT: all hard checks passed` at these
  parameters — unlike the LOH1 notebooks in `12_validation/`, this run's parameters pass
  cleanly with no override needed.
- **Dependencies**: h5py, numpy, matplotlib, ObsPy.

---

## 14. SFSI

`examples/14_SFSI/`

The most complete real-site workflow in the repo — Samoa Beach, Humboldt County, CA —
spanning source-to-spectrum: UTM geometry → live CRUST 1.0 lookup → FK run → velocity/
displacement/acceleration → Newmark response spectrum, fully narrated with committed
output artifacts. Two companion HPC/SLURM scripts reuse the same source/crust to drive
the OP pipeline into DRM-ready `.h5drm` files for OpenSees.

#### `example_documented.ipynb`
- **Purpose**: Defines 48 named candidate station locations in UTM 10N, converts to
  WGS84 for an interactive `folium` sanity-check map, selects one station ("Samoa
  Beach"), defines a point source via UTM coordinates + a Gaussian STF, builds the crust
  from a **live CRUST 1.0 lookup** at the site's (lat, lon) via
  `Crust1().profile_at(...)` (two alternative crust definitions — LOH.1 and a manual
  4-layer model — are left commented out as reference, not used), runs the classic FK
  engine with an `HDF5StationListWriter`, saves the station to `.npz` as a backup,
  converts velocity → displacement/acceleration by hand (trapezoidal integration /
  backward finite difference), computes an elastic pseudo-acceleration response spectrum
  via a hand-written Newmark-β SDOF integrator (`NewmarkSpectrumAnalyzer`,
  `@njit`-accelerated) swept over 0–5 s period.
- **Required inputs**: None external beyond the packaged CRUST 1.0 data. Needs `folium`,
  `pyproj` (for the UTM↔WGS84 map only, not the physics), `numba` (for the `@njit`
  Newmark solver), `scipy.integrate.cumulative_trapezoid`. No MPI — single station,
  `model.run(...)` (classic engine), ~14.3s per the captured log.
- **Outputs**: `ssfi_stations.h5` (HDF5StationListWriter output, 1.4 MB, **tracked in
  git**), `Samoa Beach.npz` (per-station backup, 4.95 MB, **tracked in git**). **No PNGs**
  are saved to disk — every plot in this notebook uses `plt.show()` only, unlike every
  other notebook reviewed in this document.
- **Dependencies**: folium, pyproj, numba, scipy.
- ⚠️ **Open questions**:
  1. The interactive `folium.Map` cell output cannot be statically verified (Leaflet maps have no static image capture in notebook JSON).
  2. The Newmark spectrum implementation is presented with no comparison to a reference/known solution anywhere in this notebook — unlike `12_validation/`, there is no ground-truth check here; it is a worked example, not a validation.
  3. `M0=(1e18/5e14/20)=100.0` and source depth `z=30` km are specific to this case study with no stated justification in the markdown (unlike the LOH.1 notebooks' `M0`, which is the SCEC-prescribed value) — if there is a real earthquake/scenario behind these numbers, it is not stated in-notebook.

#### `DRM/drm.py`
- **Purpose**: HPC/SLURM production script — takes the same Samoa Beach source/crust
  (copy-pasted from the notebook, including the same commented-out alternative crust
  options) but replaces the single surface station with a `PointCloudDRMReceiver` built
  from an external FEM node file (`drm_nodes.txt`, 9864 nodes, tab-separated
  `Node_ID X Y Z Type`, FEM-local meters), and runs the full OP pipeline
  (`model.run_nearest(stage='all', ...)`), producing an `.h5drm` for OpenSees via
  `DRMHDF5StationListWriter`.
- **Required inputs**: `drm_nodes.txt` in the same directory (present, tracked, 9864 data
  rows). Must be launched under MPI — the accompanying `run.sh` is a SLURM batch script
  requesting **5 nodes × 16 tasks/node = 80 MPI ranks**, activating a venv at
  a user-specific virtual environment and invoking `mpirun <venv>/bin/python -s drm.py`
  — **hardcoded to one machine, not portable as-is.**
  `HDF5_USE_FILE_LOCKING=FALSE` is set, consistent with the OP-pipeline HDF5 gotchas
  documented in the shakemaker-skill.
- **Outputs**: `ssfi_gf.h5` (+ implied `_map.h5`/`_gf.h5` pair per OP convention) and
  `ssfi_h5drm.h5drm`. Neither file is present in this repo checkout, but the script
  **has been run to completion** (confirmed) on the cluster — outputs simply weren't
  committed here.
- **Dependencies**: `mpi4py` (implicit via the OP pipeline),
  `shakermaker.sl_extensions.PointCloudDRMReceiver`,
  `shakermaker.slw_extensions.DRMHDF5StationListWriter`.
- ⚠️ **Likely bug, not yet fixed**: the constant `_m = 0.001/1e12` is used to derive the
  OP nearest-method tolerances `delta_h = delta_v_rec = delta_v_src = 2.5*_m = 2.5e-15`
  (km). Compare against the working reference example `examples/08_drm/drm_loh1.py`,
  which correctly defines `_m = 0.001` (1 metre expressed in km) and uses physically
  meaningful tolerances (`delta_h=40*_m`=40m, `delta_v_rec=5*_m`=5m,
  `delta_v_src=200*_m`=200m). This script's extra `/1e12` factor produces a tolerance
  many orders of magnitude smaller than floating-point precision at km scale — almost
  certainly an unintended typo/leftover, not a deliberate value. Flagged here for
  awareness; not corrected in the source, since that would require your sign-off.

#### `DRM/run.sh`
- SLURM launcher for `drm.py`: 5 nodes × 16 tasks, hardcoded venv/mount paths (see above).

#### `Surface/surface_SSFI.py`
- **Purpose**: Same Samoa Beach source+crust as `drm.py`, but the receiver is a
  `SurfaceGrid` (`Lx=Ly=100 m, Lz=10 m, dx=10 m` → `nx=ny=nz=10`, `mode='plane'`,
  `plane_z=0.0`) instead of an imported FEM point cloud — i.e. a synthetic free-field
  surface-motion grid rather than DRM boundary motions tied to a specific FEM mesh.
- **Required inputs**: No external file (grid generated in-code). Same SLURM launch
  pattern as `drm.py` (`Surface/run.sh`, identical 5-node/16-task/hardcoded-path
  structure). **The job name in `run.sh` is copy-pasted from the DRM script's job name**
  (`run_drm_SSFI`), not renamed for the Surface case — likely a leftover from duplicating
  the run script.
- **Outputs**: `ssfi_gf_surface.h5` and `ssfi_h5drm_surface.h5drm` — not present in this
  repo checkout, but this script **has been run to completion** (confirmed), same as
  `drm.py`.
- **Dependencies**: `shakermaker.sl_extensions.SurfaceGrid`,
  `shakermaker.slw_extensions.DRMHDF5StationListWriter`.
- ⚠️ **Likely bug, not yet fixed**: same unexplained `_m`/`delta_h` constant as `drm.py`
  (identical code block, copy-pasted) — see the note under `DRM/drm.py` above.
- **Note on the `SurfaceGrid` + `DRMHDF5StationListWriter` pairing**: intentional and
  temporary. This is not meant to be a genuine DRM boundary (which normally needs a
  closed box, not a single plane) — the `.h5drm` container format is used here only so
  the output can be read by the separate `ShakerMakerResults` tool. In principle this
  should just be a plain `.h5` file; the writer choice is expected to be revisited later.

---

## Legacy Examples

`examples/legacy_examples/`

A 6-script regression suite (no notebooks) preserving José A. Abell's original upstream
ShakerMaker workflow, deliberately kept unmodified so it can serve as a canary. Per its
own `README.md`: "if any of them stops working after a change, the original workflow was
broken." Five scripts (`example0`–`example4`) exercise the classic high-level API
(`CrustModel`/`PointSource`/`FaultSource`/`Station`/`ShakerMaker.run()`). The sixth
(`example5`) is qualitatively different — it drops to the raw Fortran `subgreen` kernel
entirely. None require MPI, numba, or any external data file beyond what a sibling script
in this same folder generates.

#### `README.md`
States these are the unmodified upstream examples, kept as a regression reference for the
classic `ShakerMaker.run()` path (not the OP pipeline).

#### `example0_readme_example.py`
- **Purpose**: Minimal end-to-end demo — 2-layer `CrustModel` by hand, one point source,
  one station, `model.run()` with class defaults (no explicit `dt`/`nfft`/`dk`/`tb`),
  plots with `ZENTPlot`.
- **Outputs**: No file written. Interactive `ZENTPlot(s, xlim=[0,60], show=True)`
  (blocking).
- **Dependencies**: base shakermaker only.

#### `example1_simple.py`
- **Purpose**: Same pattern as example0 but uses the pre-packaged `SCEC_LOH_1()` crust
  instead of a hand-built one, passes explicit `dt/nfft/dk/tb` to `model.run()`. Station
  has `filter_results=True` (10 Hz low-pass).
- **Outputs**: No file. `ZENTPlot(s, show=True, xlim=[0,3])`.
- **Note**: imports `from shakermaker import shakermaker` then
  `shakermaker.ShakerMaker(...)`, whereas `example0` and the numbered `examples/` folders
  use `from shakermaker.shakermaker import ShakerMaker` — cosmetic only, both resolve to
  the same class.

#### `example2_drm.py`
- **Purpose**: DRM workflow — single point source with a `Brune` STF, single-layer
  (halfspace-only) crust, `DRMBox` sized off `fmax` and `vs`
  (`dx = vs/fmax/15`), writes results via `DRMHDF5StationListWriter` (no plot).
  `dt` derived from `fmax` (`dt=1/(2*fmax)`, Nyquist-consistent).
- **Outputs**: `motions.h5drm`, written **to the current working directory** (relative
  path, not scoped to the example folder).
- **Dependencies**: h5py. No MPI (uses `.run()`, not `.run_nearest()`).
- **Note**: this script's `DRMBox(x0, [nx,ny,nz], [dx,dx,dx], metadata={...})` call
  matches the current (positional `pos, nelems, h`) constructor signature — no
  inconsistency found against the current codebase for this particular script.

#### `example3-save-station.py`
- **Purpose**: Builds a 2-layer crust from elastic parameters (`Vs`, `nu`, `rho` →
  derives `G`, `M`, `Vp` via Lamé/Poisson relations) with a `Brune`-STF source, runs
  `model.run()`, saves the station via `s.save("mystation.npz")`, prints `s`, plots.
- **Outputs**: `mystation.npz`, written to CWD.
- **Dependencies**: base shakermaker + `math.sqrt`.
- ⚠️ **Open questions**:
  1. `nfft=4096/8` evaluates to a Python **float** (`512.0`), not cast to `int`, before being passed to `model.run(nfft=nfft, ...)`. Whether the underlying f2py `subgreen` call tolerates a float `nfft` (silent numpy coercion vs. an error) was not traced into `shakermaker.py`/`core.pyf` to confirm.
  2. A `for z in [1.]:` loop wraps the whole script body, with a commented-out multi-value alternative (`#[0.2,0.5,1,1.5]`) right next to it — reads like leftover exploratory/parametric-sweep code never cleaned up. No functional effect (loop always runs once), but the reason for keeping it as a loop of one is not stated.

#### `example4-load-station.py`
- **Purpose**: Companion/continuation of `example3` — loads a previously saved `Station`
  from `mystation.npz` via `s.load(...)`, re-plots with `ZENTPlot`.
- **Required inputs**: **`mystation.npz` must already exist in the CWD**, produced by
  running `example3-save-station.py` first (confirmed: `mystation.npz` does not exist as
  a committed file anywhere in the repo — this is a two-step runtime dependency between
  two scripts, not a missing repo asset). This is the only example in the folder that
  isn't self-contained (must run `example3` first, same working directory).
- **Outputs**: No file written; prints `s`, shows `ZENTPlot`.
- **Dependencies**: base shakermaker only.

#### `example5-exploregreen.py`
- **Purpose**: Two things in one file: (a) a fully commented-out block mirroring an
  `example1`-style workflow using `SCEC_LOH_3()`, left in as inactive reference code; (b)
  the active code, which calls the low-level Fortran kernel directly
  (`from shakermaker.core import subgreen`) — bypassing `CrustModel`/`PointSource`/
  `ShakerMaker` entirely. Builds a 3-case perturbation study (baseline, +Δ in `sy`, +Δ in
  `ry`) and plots all 9 raw Green's function tensor components per case on a 3×3 grid.
- **Required inputs**: None external; crust/source/receiver arrays (`d, a, b, rho, qa,
  qb`, etc.) hardcoded, passed positionally into `subgreen(...)`.
- **Outputs**: No file. Only `plt.show()` at the end.
- **Dependencies**: base shakermaker + matplotlib, numpy. Imports `SOCal_LF` and
  `SCEC_LOH_3` but **never uses either** (both live only in the dead commented-out block).
- ⚠️ **Open questions**:
  1. The 29-positional-argument call to `subgreen(...)` was checked against the f2py signature in `core.pyf` and lines up correctly (the `.pyf`'s extra names — `tdata, sx, sy, rx, ry, zz, ee, nn, t0` — are `intent(out)` return values, consistent with the script unpacking `tdata, z, e, n, t0 = subgreen(...)`); this was verified by inspection only, not by executing the code.
  2. The `for i in range(9): plt.subplot(...); plt.plot(...)` plotting block appears once correctly inside the 3-case loop, then again, verbatim, **after** the loop closes — reusing whatever `t`/`tdata`/`i` were left over from the *last* case only. This looks like a leftover/duplicate block rather than intentional; not resolved, only flagged.
  3. `x = 7.0` is set at module level then immediately overwritten inside the loop before any use — dead code.
  4. The low-level Fortran flags (`mb, src, rcv, stype, updn, sigma, smth, wc1, wc2, pmin, pmax, kc, taper, pf, df, lf`) have no in-script documentation of their meaning or valid ranges.

---

## Open Questions Appendix

This section was fully re-reviewed against the actual implementation (not just the
example scripts) and against git history. Of the original 17 items, 15 are now resolved
with evidence from the repo itself or confirmed directly; 1 needs a small design
decision; 1 remains genuinely external to the repo.

### Resolved with evidence from the code/git history

- **`crust1_sites.py`** (01_crustmodel): the single-site-only second half is intentional
  — a deliberate two-part structure (quick multi-site loop, then a full-API deep-dive on
  one site).
- **`pointcloud_drm.py` / `DRM/drm.py`** (04_receivers, 14_SFSI): the `x0_fem`/
  `drmbox_x0` transform values are illustrative example numbers, kept as-is — not tied to
  a specific real project.
- **`legacy_migration.py`** (06_nearest_method): `build_pair_to_slot_from_legacy_h5`
  writes its new datasets into the same legacy database file, in place (confirmed by the
  method's docstring in `shakermaker/shakermaker.py`).
- **`hdf5_writer.py`** (07_writers): `"legacy"` and `"progressive"` write the identical
  `/Data` + `/Metadata` schema — they differ only in write timing (buffered-then-flushed
  vs. streamed immediately), confirmed by reading
  `shakermaker/slw_extensions/hdf5stationlistwriter.py`.
- **`drm_loh1.py`** (08_drm): the locally-present `s1`-named output files are confirmed
  untracked (the whole `drm_loh1_output/` directory has no git history — it matches the
  `.gitignore` rule `examples/**/drm_*_output/`), and every commit of the script has
  `selected_stations=['Centro']`. These are local artifacts from a manually-edited,
  never-committed run.
- **`data/model_summary.h5`** (13_shakermaker_sw4): not a real format mismatch — the
  actual `sw4_exporter` bundle has no separate "compact" variant; the folder's README is
  just using informal phrasing.
- **`shaker_vs_sw4.py`, `DRM/drm.py`, `Surface/surface_SSFI.py`**: confirmed run to
  completion (outside this checkout); their output files simply weren't committed here.
- **`MINSLIP = 1.9217`** (`example_FFSP.ipynb`, 10_ffsp): a practical runtime limiter, not
  a physically-derived value — it keeps only a small subset of the highest-slip subfaults
  so the manual FFSP→`FaultSource` bridge doesn't have to run all 256×128 subfaults
  through the legacy engine.
- **Stale `example8_ffsp.py` comment** (`ffsp_run.py`/`ffsp_io.py`, 10_ffsp): confirmed
  stale — git history shows the pre-reorganization flat layout had
  `examples/example7_ffsp.py` (not `example8`) as the FFSP example.
- **`data/LOH.1_prose3`** (12_validation): originates from SW4's own LOH.1 validation
  materials, not a ShakerMaker-generated file.
- **`example_LOH1_gf.ipynb` vs. `LOH1_greens_functions.ipynb`** (12_validation): both
  added together in the same commit; a same-day follow-up only fixed an encoding issue in
  one of them. Both are intentionally kept (narrated version + raw-internals version).
- **`check_parameters`-warns-then-runs-anyway pattern** (12_validation, multiple
  notebooks): confirmed by design — the method's own docstring states it separates "hard
  checks" from "recommended changes" and never blocks execution.
- **`_m = 0.001/1e12`** in `DRM/drm.py` and `Surface/surface_SSFI.py` (14_SFSI): compared
  against the correct working pattern in `examples/08_drm/drm_loh1.py` (`_m = 0.001` = 1
  metre in km, with physically meaningful tolerances like `delta_h=40*_m`=40m), the SFSI
  scripts' extra `/1e12` factor produces an effectively-zero, physically meaningless
  tolerance (`2.5e-15` km). **Flagged as a likely bug** in those two scripts — not
  corrected in the source, since fixing it would need your sign-off.
- **`Surface/surface_SSFI.py`**'s `SurfaceGrid` + `DRMHDF5StationListWriter` pairing
  (14_SFSI): intentional and temporary — the `.h5drm` container is used only so
  `ShakerMakerResults` can read the output; this isn't meant to be a real DRM boundary,
  and is expected to become a plain `.h5` file later.
- **Missing output directories** across `06_nearest_method`, `07_writers`,
  `09_sw4_export`: `.gitignore` confirms `*.h5drm` and `examples/**/drm_*_output/` are
  deliberately excluded ("regenerate by running the examples"). Plain `*.h5` files and
  `_sw4_out*/` directories are **not** covered by any gitignore rule, though — their
  absence just means those specific runs were never generated/committed in this
  checkout.

### Needs a small design decision from you

- **`drm_vs_direct.py`** (08_drm): the script computes a `Station` directly at a point
  and a tiny `DRMBox`'s QA station at the same point, but its assertions only check that
  both responses are non-empty and non-zero — they never compare `zd` vs. `zq` to each
  other (e.g. via `np.allclose` or a correlation, the way `LOH1_check.py` does). Confirmed
  via git history that this has been the case since the file was first added — it was
  never weakened from a stronger check. **Question**: should a real numeric comparison be
  added so the script actually validates DRM-vs-direct equivalence, or is the current
  "both ran and are nonzero" check sufficient for this smoke-test tier?

### Still genuinely open (external to the repo)

- None remaining that require your domain knowledge beyond what's answered above —
  everything else has either been resolved from the code/git history or confirmed
  directly.

### Would require actually executing code (not done in this review)

- Whether `example3-save-station.py`'s float `nfft=512.0` is silently coerced or errors.
- Whether `ffsp_io.py`'s HDF5 round-trip assertions currently pass on this machine.
- The exact printed group/dataset schema from running `explore_h5_output.py`.
- The final numeric correlation value in `LOH1_greens_functions.ipynb`'s last cell.
- Whether `example5-exploregreen.py`'s duplicate trailing plot block produces the
  intended figure or an artifact of leftover loop state.
