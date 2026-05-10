"""
build_h5drm_from_sw4_case
=========================
Build an .h5drm file from an SW4 case directory.

Reads SW4 .txt output files for all receivers listed in the .in file
(ShakerMaker DRM stations, topography surface, between-topo-and-z0,
and SW4 domain grid stations) and writes them into a single .h5drm
file compatible with ShakerMaker Results.

The coordinate system can be either:
  - SW4 local km (default, move_2_shakermaker_coor=False)
  - ShakerMaker/UTM km  (move_2_shakermaker_coor=True)
"""

import datetime
import shutil
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# .in file parsers
# ---------------------------------------------------------------------------

def _parse_kv(line):
    out = {}
    for token in line.split():
        if "=" in token:
            k, v = token.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _read_sw4_input(input_file):
    fileio_path   = None
    topo_rel_file = None
    rec_lines     = []
    with open(input_file, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("fileio "):
                fileio_path = _parse_kv(line).get("path")
            elif line.startswith("topography "):
                topo_rel_file = _parse_kv(line).get("file")
            elif line.startswith("rec "):
                kv = _parse_kv(line)
                if "file" in kv:
                    rec_lines.append(kv)
    return fileio_path, topo_rel_file, rec_lines


def _build_topo_lookup(topo_path):
    if topo_path is None or not Path(topo_path).exists():
        return {}
    lines = [l.strip() for l in Path(topo_path).read_text("ascii").splitlines() if l.strip()]
    lookup = {}
    for line in lines[1:]:
        cols = line.split()
        if len(cols) >= 3:
            x, y, z = float(cols[0]), float(cols[1]), float(cols[2])
            lookup[(round(x), round(y))] = z
    return lookup


def _xyz_from_rec(rec, topo_lookup):
    """
    Convert an SW4 rec line to [x_km, y_km, z_km].

    SW4 / h5drm convention:
      z= positive   -> below datum
      depth=0       -> on the surface  -> z = -topo_z (above datum)
      depth=d       -> d meters below topography -> z = -(topo_z - d)
    """
    x = float(rec.get("x", 0.0))
    y = float(rec.get("y", 0.0))
    if "depth" in rec:
        topo_z = topo_lookup.get((round(x), round(y)), 0.0)
        depth  = float(rec["depth"])
        z = -(topo_z - depth)
    elif "z" in rec:
        z = float(rec["z"])
    else:
        z = 0.0
    return [x / 1000.0, y / 1000.0, z / 1000.0]


def _kind_from_rec(rec):
    if "depth" in rec:
        return "topo_surface" if float(rec["depth"]) == 0.0 else "topo_z0"
    elif "z" in rec:
        return "domain_grid"
    return "unknown"


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def _fast_loadtxt(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    lines = [l for l in raw.split(b"\n") if l and l[0:1] != b"#"]
    return np.fromstring(b" ".join(lines), dtype=np.float64, sep=" ").reshape(-1, 4)


def _signals(data, t, dt, use_filter, freqmin, freqmax, corners, zerophase):
    """
    SW4 .txt columns: time | X=North | Y=East | Z=down+
    h5drm component order: E, N, Z
    """
    n_v = data[:, 1]
    e_v = data[:, 2]
    z_v = data[:, 3]

    if use_filter:
        from obspy import Trace, Stream
        traces = [Trace(data=a.astype(np.float32)) for a in (e_v, n_v, z_v)]
        for tr in traces:
            tr.stats.delta = dt
        st = Stream(traces)
        st.filter("bandpass", freqmin=freqmin, freqmax=freqmax,
                  corners=corners, zerophase=zerophase)
        e_v = st[0].data.astype(np.float64)
        n_v = st[1].data.astype(np.float64)
        z_v = st[2].data.astype(np.float64)

    velocity     = np.vstack((e_v, n_v, z_v)).astype(np.float64)
    displacement = np.zeros_like(velocity)
    displacement[:, 1:] = np.cumsum(
        0.5 * (velocity[:, 1:] + velocity[:, :-1]) * dt, axis=1)
    acceleration = np.empty_like(velocity)
    acceleration[:, 1:-1] = (velocity[:, 2:] - velocity[:, :-2]) / (2.0 * dt)
    acceleration[:, 0]    = (velocity[:, 1]  - velocity[:, 0])   / dt
    acceleration[:, -1]   = (velocity[:, -1] - velocity[:, -2])  / dt
    return velocity, displacement, acceleration


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def build_h5drm_from_sw4_case(
    case_path,
    input_file=None,
    output_name="motions.h5drm",
    use_filter=False,
    freqmin=0.25,
    freqmax=10.0,
    corners=4,
    zerophase=True,
    move_2_shakermaker_coor=False,
):
    """
    Build an .h5drm file from an SW4 case directory.

    Parameters
    ----------
    case_path : str or Path
        Path to the case directory (contains sw4/ and shakermakerexports/).
    input_file : str or Path, optional
        SW4 input file name (default: sw4/shakermaker2sw4.in).
    output_name : str, optional
        Output .h5drm file name (default: motions.h5drm).
    use_filter : bool, optional
        Apply bandpass filter to signals (default: False).
    freqmin, freqmax : float, optional
        Filter corner frequencies (default: 0.25, 10.0 Hz).
    corners : int, optional
        Filter corners (default: 4).
    zerophase : bool, optional
        Zero-phase filter (default: True).
    move_2_shakermaker_coor : bool, optional
        If True, convert coordinates from SW4 local to ShakerMaker/UTM
        by adding the domain origin offset read from model_summary.h5.
        If False, keep SW4 local km (default: False).
    """
    case_path    = Path(case_path)
    sw4_dir      = case_path / "sw4"
    exports_dir  = case_path / "shakermakerexports"
    geometry_h5  = exports_dir / "model_geometry.h5"
    summary_h5   = exports_dir / "model_summary.h5"
    output_h5drm = exports_dir / output_name

    exports_dir.mkdir(parents=True, exist_ok=True)

    if input_file is None:
        input_file = sw4_dir / "shakermaker2sw4.in"
    else:
        input_file = Path(input_file)
        if not input_file.is_absolute():
            input_file = sw4_dir / input_file

    print(f"\n{'='*55}")
    print(f"case   : {case_path}")
    print(f"output : {output_h5drm}")
    print(f"{'='*55}")

    # ------------------------------------------------------------------
    # 1. Read geometry.h5 (ShakerMaker DRM stations)
    # ------------------------------------------------------------------
    with h5py.File(geometry_h5, "r") as f:
        sm_xyz      = f["DRM_Data/xyz"][:]
        sm_internal = f["DRM_Data/internal"][:].astype(bool)
        qa_xyz      = f["DRM_QA_Data/xyz"][:]

        rec_files     = [v.decode() if isinstance(v, bytes) else str(v)
                         for v in f["SW4_Receivers/file"][:]]
        rec_model_idx = f["SW4_Receivers/model_index"][:].astype(int)
        rec_is_qa     = f["SW4_Receivers/is_qa"][:].astype(bool)

    sm_file_map = {
        fname: (int(midx), bool(iq))
        for fname, midx, iq in zip(rec_files, rec_model_idx, rec_is_qa)
    }
    sm_fnames = set(sm_file_map.keys())
    n_sm = len(sm_xyz)

    # ------------------------------------------------------------------
    # 1b. Read SW4 domain origin for coordinate conversion
    # ------------------------------------------------------------------
    domain_origin_km = np.zeros(3)
    if move_2_shakermaker_coor:
        if summary_h5.exists():
            with h5py.File(summary_h5, "r") as f:
                sw4 = f["SW4"]
                x_origin = float(sw4["x_origin"][()])
                y_origin = float(sw4["y_origin"][()])
                z_origin = float(sw4["z_origin"][()])
            # config stores origin_m = -domain_origin_m
            domain_origin_km = -np.array([x_origin, y_origin, z_origin]) / 1000.0
            print(f"  Origin offset: +({domain_origin_km[0]:.2f}, "
                  f"{domain_origin_km[1]:.2f}, {domain_origin_km[2]:.2f}) km")
        else:
            print("  WARNING: model_summary.h5 not found — keeping SW4 local coords")

    # Apply coordinate offset
    sm_xyz = sm_xyz + domain_origin_km
    qa_xyz = qa_xyz + domain_origin_km

    # ------------------------------------------------------------------
    # 2. Read .in -> full ordered receiver list
    # ------------------------------------------------------------------
    fileio_path, topo_rel_file, rec_lines = _read_sw4_input(input_file)
    station_dir = (input_file.parent / fileio_path).resolve()
    topo_path   = (input_file.parent / topo_rel_file) if topo_rel_file else None
    topo_lookup = _build_topo_lookup(topo_path)

    drm_stations = []
    qa_station   = None

    for rec in rec_lines:
        fname = Path(rec["file"]).name
        txt   = station_dir / f"{fname}.txt"

        if fname in sm_fnames:
            model_idx, is_qa = sm_file_map[fname]
            if is_qa:
                qa_station = {"fname": fname, "txt": txt}
                continue
            drm_stations.append({
                "fname":    fname,
                "txt":      txt,
                "xyz_km":   sm_xyz[model_idx].tolist(),
                "internal": bool(sm_internal[model_idx]),
                "kind":     "shakermaker",
            })
        else:
            if not txt.exists():
                print(f"  WARNING: {fname}.txt not found — skipping")
                continue
            xyz = _xyz_from_rec(rec, topo_lookup)
            xyz = (np.asarray(xyz) + domain_origin_km).tolist()
            drm_stations.append({
                "fname":    fname,
                "txt":      txt,
                "xyz_km":   xyz,
                "internal": True,
                "kind":     _kind_from_rec(rec),
            })

    n_total = len(drm_stations)
    n_extra = n_total - n_sm
    print(f"  SM stations    : {n_sm}")
    print(f"  Extra stations : {n_extra}")
    print(f"  Total          : {n_total}")

    # ------------------------------------------------------------------
    # 3. Time vector from the first .txt
    # ------------------------------------------------------------------
    first_data = _fast_loadtxt(drm_stations[0]["txt"])
    t  = first_data[:, 0]
    nt = len(t)
    dt = float((t[-1] - t[0]) / (nt - 1))
    print(f"  nt={nt}  dt={dt:.6g}s  duration={t[-1]:.3f}s")

    # ------------------------------------------------------------------
    # 4. Build the .h5drm file
    # ------------------------------------------------------------------
    shutil.copyfile(geometry_h5, output_h5drm)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    xyz_all      = np.array([s["xyz_km"]  for s in drm_stations], dtype=np.float64)
    internal_all = np.array([s["internal"] for s in drm_stations], dtype=bool)
    kind_all     = np.array([s["kind"]    for s in drm_stations], dtype=object)
    name_all     = np.array([s["fname"]   for s in drm_stations], dtype=object)
    data_loc_all = np.arange(n_total, dtype=np.int32) * 3

    with h5py.File(output_h5drm, "r+") as f:
        grp     = f["DRM_Data"]
        grp_qa  = f["DRM_QA_Data"]

        # Replace geometry (may have grown with extra stations)
        for key in ("xyz", "internal", "data_location", "kind", "name"):
            if key in grp:
                del grp[key]
        grp.create_dataset("xyz",           data=xyz_all)
        grp.create_dataset("internal",      data=internal_all, dtype=bool)
        grp.create_dataset("data_location", data=data_loc_all, dtype=np.int32)
        grp.create_dataset("kind",          data=kind_all, dtype=string_dtype)
        grp.create_dataset("name",          data=name_all, dtype=string_dtype)

        # Re-create signal datasets — contiguous layout
        for key in ("velocity", "displacement", "acceleration"):
            if key in grp:
                del grp[key]
            if key in grp_qa:
                del grp_qa[key]
        grp.create_dataset("velocity",     shape=(3 * n_total, nt), dtype=np.float64)
        grp.create_dataset("displacement", shape=(3 * n_total, nt), dtype=np.float64)
        grp.create_dataset("acceleration", shape=(3 * n_total, nt), dtype=np.float64)
        grp_qa.create_dataset("velocity",     shape=(3, nt), dtype=np.float64)
        grp_qa.create_dataset("displacement", shape=(3, nt), dtype=np.float64)
        grp_qa.create_dataset("acceleration", shape=(3, nt), dtype=np.float64)

        # Metadata
        meta = f["DRM_Metadata"]
        for key in ("dt", "tstart", "tend", "nt", "created_on", "writer_mode",
                    "receiver_count", "signal_units", "component_order", "component_map",
                    "filter_enabled", "filter_freqmin", "filter_freqmax",
                    "filter_corners", "filter_zerophase",
                    "domain_origin_km", "coordinate_system"):
            if key in meta:
                del meta[key]

        meta.create_dataset("dt",               data=dt)
        meta.create_dataset("tstart",           data=float(t[0]))
        meta.create_dataset("tend",             data=float(t[-1]))
        meta.create_dataset("nt",               data=int(nt))
        meta.create_dataset("receiver_count",   data=int(n_total))
        meta.create_dataset("created_on",       data=datetime.datetime.now().isoformat(), dtype=string_dtype)
        meta.create_dataset("writer_mode",      data="sw4_txt_filled",  dtype=string_dtype)
        meta.create_dataset("component_order",  data="E,N,Z",           dtype=string_dtype)
        meta.create_dataset("component_map",    data="E=SW4_Y, N=SW4_X, Z=SW4_Z(down+)", dtype=string_dtype)
        meta.create_dataset("signal_units",
                            data="velocity from SW4 txt; displacement integrated; acceleration derived",
                            dtype=string_dtype)
        meta.create_dataset("filter_enabled",   data=bool(use_filter))
        meta.create_dataset("filter_freqmin",   data=float(freqmin))
        meta.create_dataset("filter_freqmax",   data=float(freqmax))
        meta.create_dataset("filter_corners",   data=int(corners))
        meta.create_dataset("filter_zerophase", data=bool(zerophase))
        meta.create_dataset("domain_origin_km", data=domain_origin_km, dtype=np.float64)
        meta.create_dataset("coordinate_system",
                            data="shakermaker_utm_km" if move_2_shakermaker_coor
                                 else "sw4_local_km",
                            dtype=string_dtype)

        # ------------------------------------------------------------------
        # 5. Write signals station by station
        # ------------------------------------------------------------------
        for i, sta in enumerate(drm_stations):
            if not sta["txt"].exists():
                print(f"  WARNING: {sta['fname']}.txt not found — zeroed row")
                continue
            data = _fast_loadtxt(sta["txt"])
            if data.shape[0] != nt:
                raise ValueError(f"nt mismatch in {sta['fname']}.txt")
            vel, disp, acc = _signals(data, t, dt, use_filter, freqmin, freqmax, corners, zerophase)
            row = i * 3
            f["DRM_Data/velocity"]    [row:row+3] = vel
            f["DRM_Data/displacement"][row:row+3] = disp
            f["DRM_Data/acceleration"][row:row+3] = acc
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{n_total} ({100*(i+1)/n_total:.1f}%)", flush=True)

        # ------------------------------------------------------------------
        # 6. QA signal
        # ------------------------------------------------------------------
        if qa_station and qa_station["txt"].exists():
            print(f"  QA station: {qa_station['fname']}")
            qa_data = _fast_loadtxt(qa_station["txt"])
        else:
            nearest = int(np.argmin(np.sum((xyz_all - qa_xyz[0]) ** 2, axis=1)))
            print(f"  QA station (nearest fallback): {drm_stations[nearest]['fname']}")
            qa_data = _fast_loadtxt(drm_stations[nearest]["txt"])

        vel, disp, acc = _signals(qa_data, t, dt, use_filter, freqmin, freqmax, corners, zerophase)
        f["DRM_QA_Data/velocity"]    [:] = vel
        f["DRM_QA_Data/displacement"][:] = disp
        f["DRM_QA_Data/acceleration"][:] = acc
        f.flush()

    coord_label = "ShakerMaker UTM km" if move_2_shakermaker_coor else "SW4 local km"
    print(f"\n  -> {output_h5drm}  ({coord_label})")
    print(f"  DRM stations: {n_total}  (SM={n_sm}, extra={n_extra})")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build .h5drm from SW4 case directory")
    parser.add_argument("case_path", help="Path to the SW4 case directory")
    parser.add_argument("--input-file", help="SW4 input file (default: sw4/shakermaker2sw4.in)")
    parser.add_argument("--output-name", default="motions.h5drm", help="Output .h5drm name")
    parser.add_argument("--use-filter", action="store_true", help="Apply bandpass filter")
    parser.add_argument("--freqmin", type=float, default=0.25, help="Filter low corner (Hz)")
    parser.add_argument("--freqmax", type=float, default=10.0, help="Filter high corner (Hz)")
    parser.add_argument("--corners", type=int, default=4, help="Filter corners")
    parser.add_argument("--no-zerophase", dest="zerophase", action="store_false",
                        help="Disable zero-phase filtering")
    parser.add_argument("--move-2-shakermaker-coor", action="store_true",
                        help="Convert from SW4 local to ShakerMaker/UTM coordinates")
    args = parser.parse_args()

    build_h5drm_from_sw4_case(
        case_path=args.case_path,
        input_file=args.input_file,
        output_name=args.output_name,
        use_filter=args.use_filter,
        freqmin=args.freqmin,
        freqmax=args.freqmax,
        corners=args.corners,
        zerophase=args.zerophase,
        move_2_shakermaker_coor=args.move_2_shakermaker_coor,
    )



    # # ---------------------------------------------------------------------------
    # # Ejecución
    # # ---------------------------------------------------------------------------

    # build_h5drm_from_sw4_case(
    #     case_path="/mnt/deadmanschest/pxpalacios/SW4/shakermaker_2_sw4/C2_sw4_LOH1_02_Plane_Z_0_gauus",
    #     output_name="C2_sw4_LOH1_02_Plane_Z_0_gauus.h5drm",
    #     use_filter=False,
    #     move_2_shakermaker_coor=False,
    # )