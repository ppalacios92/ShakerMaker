#!/usr/bin/env python3
"""
Build a final .h5drm from a ShakerMaker/SW4 geometry skeleton and SW4 sf*.txt files.

Edit the inputs at the bottom of this file, then run:

    python simple_sw4_to_h5drm.py
"""

from pathlib import Path
import shutil
import datetime

import h5py
import numpy as np


def build_h5drm_from_sw4(
    geometry_h5,
    station_dir,
    output_h5drm,
    receiver_file_prefix="sf",
    use_filter=False,
    freqmin=0.25,
    freqmax=10.0,
    corners=4,
    zerophase=True,
):
    geometry_h5 = Path(geometry_h5)
    station_dir = Path(station_dir)
    output_h5drm = Path(output_h5drm)

    output_h5drm.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(geometry_h5, output_h5drm)

    with h5py.File(output_h5drm, "r+") as f:
        xyz = f["DRM_Data/xyz"][:]
        internal = f["DRM_Data/internal"][:]
        data_location = f["DRM_Data/data_location"][:]
        nstations = xyz.shape[0]

        first_file = station_dir / f"{receiver_file_prefix}00001.txt"
        first = np.loadtxt(first_file, comments="#")
        t = first[:, 0]
        nt = len(t)
        dt = float((t[-1] - t[0]) / (nt - 1))

        for group_name, rows in (("DRM_Data", 3 * nstations), ("DRM_QA_Data", 3)):
            group = f[group_name]
            for name in ("velocity", "displacement", "acceleration"):
                if name in group:
                    del group[name]
                group.create_dataset(name, shape=(rows, nt), dtype=np.float64, chunks=(3, nt))

        meta = f["DRM_Metadata"]
        for key in (
            "dt", "tstart", "tend", "nt", "created_on", "writer_mode",
            "receiver_file_prefix", "receiver_count", "signal_units",
            "component_order", "component_map", "filter_enabled",
            "filter_freqmin", "filter_freqmax", "filter_corners",
            "filter_zerophase",
        ):
            if key in meta:
                del meta[key]

        string_dtype = h5py.string_dtype(encoding="utf-8")
        meta.create_dataset("dt", data=dt)
        meta.create_dataset("tstart", data=float(t[0]))
        meta.create_dataset("tend", data=float(t[-1]))
        meta.create_dataset("nt", data=int(nt))
        meta.create_dataset("created_on", data=datetime.datetime.now().isoformat(), dtype=string_dtype)
        meta.create_dataset("writer_mode", data="sw4_txt_filled", dtype=string_dtype)
        meta.create_dataset("receiver_file_prefix", data=receiver_file_prefix, dtype=string_dtype)
        meta.create_dataset("receiver_count", data=int(nstations))
        meta.create_dataset("signal_units", data="velocity from SW4 txt; displacement integrated; acceleration derived", dtype=string_dtype)
        meta.create_dataset("component_order", data="E,N,Z", dtype=string_dtype)
        meta.create_dataset("component_map", data="E=SW4_Y, N=SW4_X, Z=SW4_Z", dtype=string_dtype)
        meta.create_dataset("filter_enabled", data=bool(use_filter))
        meta.create_dataset("filter_freqmin", data=float(freqmin))
        meta.create_dataset("filter_freqmax", data=float(freqmax))
        meta.create_dataset("filter_corners", data=int(corners))
        meta.create_dataset("filter_zerophase", data=bool(zerophase))

        for i in range(nstations):
            station_file = station_dir / f"{receiver_file_prefix}{i + 1:05d}.txt"
            data = np.loadtxt(station_file, comments="#")
            if data.shape[0] != nt or not np.allclose(data[:, 0], t, rtol=1e-5, atol=1e-6):
                raise ValueError(f"Time vector mismatch in {station_file}")

            # SW4 txt columns are time, X, Y, Z. For this convention:
            # E = SW4 Y, N = SW4 X, Z = SW4 Z.
            z_v = data[:, 3]
            e_v = data[:, 2]
            n_v = data[:, 1]

            if use_filter:
                from obspy import Trace, Stream

                tr_z = Trace(data=z_v.astype(np.float32))
                tr_e = Trace(data=e_v.astype(np.float32))
                tr_n = Trace(data=n_v.astype(np.float32))

                for tr in (tr_z, tr_e, tr_n):
                    tr.stats.delta = dt

                st = Stream([tr_z, tr_e, tr_n])
                st.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=corners, zerophase=zerophase)

                z_v = st[0].data.astype(np.float64)
                e_v = st[1].data.astype(np.float64)
                n_v = st[2].data.astype(np.float64)

            velocity = np.vstack((e_v, n_v, z_v)).astype(np.float64)
            displacement = np.zeros_like(velocity)
            displacement[:, 1:] = np.cumsum(0.5 * (velocity[:, 1:] + velocity[:, :-1]) * dt, axis=1)
            acceleration = np.empty_like(velocity)
            acceleration[:, 1:-1] = (velocity[:, 2:] - velocity[:, :-2]) / (2.0 * dt)
            acceleration[:, 0] = (velocity[:, 1] - velocity[:, 0]) / dt
            acceleration[:, -1] = (velocity[:, -1] - velocity[:, -2]) / dt

            row = int(data_location[i])
            f["DRM_Data/velocity"][row:row + 3, :] = velocity
            f["DRM_Data/displacement"][row:row + 3, :] = displacement
            f["DRM_Data/acceleration"][row:row + 3, :] = acceleration

        qa_index = int(meta["qa_index"][()]) if "qa_index" in meta else nstations
        qa_file_number = qa_index + 1
        qa_file = station_dir / f"{receiver_file_prefix}{qa_file_number:05d}.txt"

        if not qa_file.exists():
            qa_xyz = f["DRM_QA_Data/xyz"][:]
            nearest = int(np.argmin(np.sum((xyz - qa_xyz) ** 2, axis=1)))
            qa_file = station_dir / f"{receiver_file_prefix}{nearest + 1:05d}.txt"

        data = np.loadtxt(qa_file, comments="#")
        z_v = data[:, 3]
        e_v = data[:, 2]
        n_v = data[:, 1]

        if use_filter:
            from obspy import Trace, Stream

            tr_z = Trace(data=z_v.astype(np.float32))
            tr_e = Trace(data=e_v.astype(np.float32))
            tr_n = Trace(data=n_v.astype(np.float32))
            for tr in (tr_z, tr_e, tr_n):
                tr.stats.delta = dt
            st = Stream([tr_z, tr_e, tr_n])
            st.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=corners, zerophase=zerophase)
            z_v = st[0].data.astype(np.float64)
            e_v = st[1].data.astype(np.float64)
            n_v = st[2].data.astype(np.float64)

        velocity = np.vstack((e_v, n_v, z_v)).astype(np.float64)
        displacement = np.zeros_like(velocity)
        displacement[:, 1:] = np.cumsum(0.5 * (velocity[:, 1:] + velocity[:, :-1]) * dt, axis=1)
        acceleration = np.empty_like(velocity)
        acceleration[:, 1:-1] = (velocity[:, 2:] - velocity[:, :-2]) / (2.0 * dt)
        acceleration[:, 0] = (velocity[:, 1] - velocity[:, 0]) / dt
        acceleration[:, -1] = (velocity[:, -1] - velocity[:, -2]) / dt

        f["DRM_QA_Data/velocity"][:, :] = velocity
        f["DRM_QA_Data/displacement"][:, :] = displacement
        f["DRM_QA_Data/acceleration"][:, :] = acceleration
        f.flush()

    print(f"Wrote {output_h5drm}")
    print(f"Receivers: {nstations}")
    print(f"Samples: {nt}")
    print(f"dt: {dt}")
    print(f"Filter: {use_filter}")


if __name__ == "__main__":
    build_h5drm_from_sw4(
        geometry_h5=r"C:\path\to\shakermakerexports\model_geometry.h5",
        station_dir=r"C:\path\to\sw4\shakermaker2sw4_fileio",
        output_h5drm=r"C:\path\to\output\motions.h5drm",
        receiver_file_prefix="sf",
        use_filter=True,
        freqmin=0.25,
        freqmax=10.0,
        corners=4,
        zerophase=True,
    )
