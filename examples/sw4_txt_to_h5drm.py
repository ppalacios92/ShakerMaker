#!/usr/bin/env python3
"""Convert SW4 ASCII station output to a simple ShakerMaker-style .h5drm file.

The SW4 station text files are treated as velocity histories. Output rows follow
ShakerMaker's DRM convention: E, N, Z. For this LOH.1 setup, SW4 x is North and
SW4 y is East, so the text columns are mapped as E=Y, N=X, Z=Z.
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import numpy as np


DEFAULT_BASE = Path(
    r"C:\Dropbox\01. Brain\10. Ph.D U ANDES\04. Clases\02. Semestre02 2025-2"
    r"\01. SAIC\04. SW4\LOH.1\example\topo_gauss_stations"
)


def str_to_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "si"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def read_topography_xyz(topo_path: Path) -> np.ndarray:
    with topo_path.open("r", encoding="utf-8") as f:
        header = f.readline().split()
        if len(header) < 2:
            raise ValueError(f"Invalid topography header in {topo_path}")
        nx, ny = int(header[0]), int(header[1])
        points = []
        for line in f:
            if not line.strip():
                continue
            cols = line.split()
            if len(cols) >= 3:
                points.append((float(cols[0]), float(cols[1]), float(cols[2])))

    expected = nx * ny
    if len(points) != expected:
        raise ValueError(f"Topography has {len(points)} points, expected {expected}")
    return np.asarray(points, dtype=np.float64)


def read_sw4_velocity_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"{path} must have at least 4 columns: t, X, Y, Z")

    t = data[:, 0]
    x_v = data[:, 1]
    y_v = data[:, 2]
    z_v = data[:, 3]

    # ShakerMaker DRM row convention: E, N, Z. In this SW4 model: y=E, x=N.
    velocity_enz = np.vstack((y_v, x_v, z_v))
    return t, velocity_enz


def check_uniform_time(t: np.ndarray, path: Path) -> float:
    if t.size < 2:
        raise ValueError(f"{path} has fewer than two time samples")
    dt = float((t[-1] - t[0]) / (t.size - 1))

    # SW4 ASCII output prints time with limited precision, so adjacent printed
    # differences alternate slightly even when the true simulation dt is uniform.
    expected = t[0] + np.arange(t.size, dtype=np.float64) * dt
    if not np.allclose(t, expected, rtol=1.0e-7, atol=max(1.0e-6, abs(dt) * 1.0e-3)):
        max_error = float(np.max(np.abs(t - expected)))
        raise ValueError(f"Non-uniform time step in {path}; max time error={max_error:g}")
    return dt


def maybe_filter_velocity(
    velocity_enz: np.ndarray,
    dt: float,
    do_filter: bool,
    freqmin: float,
    freqmax: float,
    corners: int,
    zerophase: bool,
) -> np.ndarray:
    if not do_filter:
        return velocity_enz

    try:
        from obspy import Stream, Trace
    except ImportError as exc:
        raise ImportError(
            "Filtering requires obspy. Install obspy or run with --filter false."
        ) from exc

    traces = []
    for component in velocity_enz:
        tr = Trace(data=component.astype(np.float32))
        tr.stats.delta = dt
        traces.append(tr)

    st = Stream(traces)
    st.filter(
        "bandpass",
        freqmin=freqmin,
        freqmax=freqmax,
        corners=corners,
        zerophase=zerophase,
    )
    return np.vstack([tr.data.astype(np.float64) for tr in st])


def integrate_velocity(velocity: np.ndarray, dt: float) -> np.ndarray:
    displacement = np.zeros_like(velocity)
    increments = 0.5 * (velocity[:, 1:] + velocity[:, :-1]) * dt
    displacement[:, 1:] = np.cumsum(increments, axis=1)
    return displacement


def derive_acceleration(velocity: np.ndarray, dt: float) -> np.ndarray:
    acceleration = np.empty_like(velocity)
    acceleration[:, 1:-1] = (velocity[:, 2:] - velocity[:, :-2]) / (2.0 * dt)
    acceleration[:, 0] = (velocity[:, 1] - velocity[:, 0]) / dt
    acceleration[:, -1] = (velocity[:, -1] - velocity[:, -2]) / dt
    return acceleration


def find_center_node(xyz_m: np.ndarray) -> int:
    center_xy = 0.5 * (xyz_m[:, :2].min(axis=0) + xyz_m[:, :2].max(axis=0))
    dist2 = np.sum((xyz_m[:, :2] - center_xy) ** 2, axis=1)
    return int(np.argmin(dist2))


def write_h5drm(
    out_file: Path,
    xyz_km: np.ndarray,
    velocity: np.ndarray,
    displacement: np.ndarray,
    acceleration: np.ndarray,
    t: np.ndarray,
    qa_idx: int,
    metadata: dict[str, object],
) -> None:
    import h5py

    n_nodes = xyz_km.shape[0]
    nt = t.size
    data_location = np.arange(n_nodes, dtype=np.int32) * 3
    internal = np.ones(n_nodes, dtype=bool)
    dt = float(t[1] - t[0])

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_file, "w") as f:
        g = f.create_group("DRM_Data")
        g.create_dataset("xyz", data=xyz_km.astype(np.float64))
        g.create_dataset("internal", data=internal)
        g.create_dataset("data_location", data=data_location)
        g.create_dataset("velocity", data=velocity, chunks=(3, nt))
        g.create_dataset("displacement", data=displacement, chunks=(3, nt))
        g.create_dataset("acceleration", data=acceleration, chunks=(3, nt))

        q = f.create_group("DRM_QA_Data")
        qa_rows = slice(3 * qa_idx, 3 * qa_idx + 3)
        q.create_dataset("xyz", data=xyz_km[qa_idx : qa_idx + 1, :])
        q.create_dataset("velocity", data=velocity[qa_rows, :])
        q.create_dataset("displacement", data=displacement[qa_rows, :])
        q.create_dataset("acceleration", data=acceleration[qa_rows, :])

        m = f.create_group("DRM_Metadata")
        base_metadata = {
            "dt": dt,
            "tstart": float(t[0]),
            "tend": float(t[-1]),
            "nt": int(nt),
            "created_on": _dt.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)"),
            "program_used": "sw4_txt_to_h5drm.py",
            "component_order": "E,N,Z",
            "component_map": "E=SW4_Y, N=SW4_X, Z=SW4_Z",
            "coordinate_units": "km",
            "signal_units": "velocity from SW4 txt; displacement integrated; acceleration derived",
            "internal_policy": "all true",
            "qa_index": int(qa_idx),
        }
        base_metadata.update(metadata)
        for key, value in base_metadata.items():
            m.create_dataset(key, data=value)


def build_h5drm(args: argparse.Namespace) -> None:
    station_dir = args.station_dir
    topo_file = args.topo_file
    out_file = args.out_file

    xyz_m = read_topography_xyz(topo_file)
    xyz_km = xyz_m / 1000.0
    n_nodes = xyz_km.shape[0]

    station_files = [station_dir / f"{args.prefix}{i:05d}.txt" for i in range(1, n_nodes + 1)]
    missing = [p for p in station_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} station files; first missing: {missing[0]}")

    t0, v0 = read_sw4_velocity_txt(station_files[0])
    dt = check_uniform_time(t0, station_files[0])
    nt = t0.size

    velocity = np.zeros((3 * n_nodes, nt), dtype=np.float64)
    velocity[0:3, :] = maybe_filter_velocity(
        v0, dt, args.filter, args.freqmin, args.freqmax, args.corners, args.zerophase
    )

    for i, path in enumerate(station_files[1:], start=1):
        t, v = read_sw4_velocity_txt(path)
        check_uniform_time(t, path)
        if t.size != nt or not np.allclose(t, t0, rtol=1.0e-5, atol=1.0e-9):
            raise ValueError(f"Time vector mismatch in {path}")
        velocity[3 * i : 3 * i + 3, :] = maybe_filter_velocity(
            v, dt, args.filter, args.freqmin, args.freqmax, args.corners, args.zerophase
        )

    displacement = integrate_velocity(velocity, dt)
    acceleration = derive_acceleration(velocity, dt)
    qa_idx = find_center_node(xyz_m)

    metadata = {
        "source_station_dir": str(station_dir),
        "source_topography": str(topo_file),
        "filter_enabled": bool(args.filter),
        "filter_freqmin": float(args.freqmin),
        "filter_freqmax": float(args.freqmax),
        "filter_corners": int(args.corners),
        "filter_zerophase": bool(args.zerophase),
    }
    write_h5drm(out_file, xyz_km, velocity, displacement, acceleration, t0, qa_idx, metadata)

    print(f"Wrote: {out_file}")
    print(f"Nodes: {n_nodes}")
    print(f"nt: {nt}")
    print(f"dt: {dt}")
    print(f"QA index: {qa_idx}")
    print(f"Filter: {args.filter}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--station-dir",
        type=Path,
        default=DEFAULT_BASE / "LOH1_topo_ifile",
        help="Directory containing sfNNNNN.txt files.",
    )
    parser.add_argument(
        "--topo-file",
        type=Path,
        default=DEFAULT_BASE / "topography_LOH1_gauss.topo",
        help="SW4 cartesian topography file used to generate the stations.",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=DEFAULT_BASE / "LOH1_topo_ifile.h5drm",
        help="Output .h5drm file.",
    )
    parser.add_argument("--prefix", default="sf", help="Station file prefix.")
    parser.add_argument("--filter", type=str_to_bool, default=False, help="true/false.")
    parser.add_argument("--freqmin", type=float, default=0.25)
    parser.add_argument("--freqmax", type=float, default=10.0)
    parser.add_argument("--corners", type=int, default=4)
    parser.add_argument("--zerophase", type=str_to_bool, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    build_h5drm(parse_args())
