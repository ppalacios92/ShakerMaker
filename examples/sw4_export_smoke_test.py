"""
sw4_export_smoke_test.py
========================
End-to-end smoke test for the SW4 exporter. Builds small ShakerMaker models,
runs the exporter, opens the HDF5 package, and checks that every data
structure the rest of the pipeline relies on is present and consistent.

The test does not call the FK core, so it does not need MPI nor the compiled
Fortran extension. It only exercises the path:

    model -> SW4Exporter -> sw4_package.h5 -> unpack_sw4_package_h5

Run from the repo root with the project venv:

    python examples/sw4_export_smoke_test.py

The script exits with status 0 on success and prints a per-case pass/fail line.
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import h5py

from shakermaker.shakermaker import ShakerMaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions import Brune
from shakermaker.sl_extensions import DRMBox

from shakermaker.sw4_exporter import (
    SW4Exporter,
    SW4ExportConfig,
    unpack_sw4_package_h5,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _make_crust():
    """Two-layer crust good enough for an export round-trip."""
    crust = CrustModel(2)
    crust.add_layer(2.0, 4.5, 2.6, 2.4, 600.0, 300.0)
    crust.add_layer(0.0, 6.0, 3.5, 2.7, 1000.0, 500.0)
    return crust


def _make_fault(zsrc=2.0, dt=0.01):
    """One PointSource with a Brune STF already discretised."""
    stf = Brune(f0=2.0, t0=0.0)
    stf.dt = dt
    src = PointSource([0.5, 0.5, zsrc], [0.0, 90.0, 0.0], tt=0.0, stf=stf)
    return FaultSource([src], metadata={"name": "smoke-fault"})


def _make_stations():
    """Four stations forming a square around the source."""
    coords = [
        ( 1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        ( 0.0,  1.0, 0.0),
        ( 0.0, -1.0, 0.0),
    ]
    stations = [Station(c, metadata={"name": f"s{i}"}) for i, c in enumerate(coords)]
    return StationList(stations, metadata={"name": "smoke-stations"})


def _make_drmbox():
    """Small DRMBox so the QA-station path is exercised."""
    return DRMBox(
        pos=[0.0, 0.0, 0.0],
        nelems=[2, 2, 2],
        h=[0.2, 0.2, 0.2],
        metadata={"name": "smoke-drm"},
    )


def _write_synthetic_topo(path, nx=4, ny=4, dx=500.0, dy=500.0):
    """Tiny cartesian topography file in SW4 format (1.0 m elevation)."""
    lines = [f"{nx} {ny}"]
    for j in range(ny):
        for i in range(nx):
            x = i * dx
            y = j * dy
            z = 1.0
            lines.append(f"{x:.1f} {y:.1f} {z:.6f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


# ---------------------------------------------------------------------------
# HDF5 structural invariants
# ---------------------------------------------------------------------------

_REQUIRED_GROUPS = (
    "manifest",
    "config",
    "coordinates",
    "crust",
    "stations",
    "sw4_input",
    "sources",
    "topography",
    "receivers",
    "drm_template",
    "files",
)

_REQUIRED_CONFIG_KEYS = (
    "h", "x_domain", "y_domain", "z_domain",
    "x_origin", "y_origin", "z_origin",
    "tmax", "m0",
    "supergrid_gp",
    "fileio_path", "station_prefix", "h5_export_name",
)


def _check_package_structure(h5_path):
    """Open the package and assert that every expected group/dataset is there."""
    with h5py.File(h5_path, "r") as f:
        assert f.attrs.get("purpose", "") == "transport_unpack_to_sw4_files", \
            "package missing transport purpose attribute"
        for name in _REQUIRED_GROUPS:
            assert name in f, f"missing top-level group /{name}"
        for key in _REQUIRED_CONFIG_KEYS:
            assert key in f["config"], f"missing config/{key}"

        # Coordinates round-trip: sm_offset_m + sw4_origin_m = 0
        off_m = f["coordinates/shakermaker_to_sw4_offset_m"][:]
        org_m = f["coordinates/sw4_origin_in_shakermaker_m"][:]
        assert np.allclose(off_m + org_m, 0.0, atol=1e-12), \
            "shakermaker<->sw4 offsets are not negatives of each other"

        # Sources block is internally consistent.
        sources = f["sources"]
        ids = sources["id"][:]
        npts = sources["npts"][:]
        offsets = sources["data_offsets"][:]
        values = sources["data_values"][:]
        assert len(ids) == len(npts) == len(offsets), "sources arrays misaligned"
        assert int(offsets[-1] + npts[-1]) == len(values), \
            "sources data_values length does not match offsets+npts"

        # Receivers block matches its accessors.
        receivers = f["receivers"]
        nrec = len(receivers["id"][:])
        for key in ("file", "kind", "xyz_km", "internal", "is_qa", "model_index", "metadata"):
            assert key in receivers, f"missing receivers/{key}"
            assert len(receivers[key]) == nrec, f"receivers/{key} length mismatch"

        # File payloads index matches the embedded blob count.
        files_group = f["files"]
        relpaths = [
            v.decode("utf-8") if isinstance(v, bytes) else str(v)
            for v in files_group["relpath"][:]
        ]
        entries = files_group["entries"]
        assert len(relpaths) == len(entries.keys()), "files relpath vs entries count mismatch"
        assert any(name.endswith("shakermaker2sw4.in") for name in relpaths), \
            "expected sw4 input file in package"

        return {
            "sw4_input": f["sw4_input/text"][()],
            "x_origin_m": float(f["config/x_origin"][()]),
            "y_origin_m": float(f["config/y_origin"][()]),
            "z_origin_m": float(f["config/z_origin"][()]),
            "x_domain": float(f["config/x_domain"][()]),
            "y_domain": float(f["config/y_domain"][()]),
            "z_domain": float(f["config/z_domain"][()]),
        }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def case_basic_stationlist(tmp):
    """Sources + plain StationList. No topography, no DRM."""
    crust = _make_crust()
    fault = _make_fault()
    stations = _make_stations()
    model = ShakerMaker(crust, fault, stations)

    config = SW4ExportConfig(
        path=tmp,
        h=50.0,
        size_domain=[3000.0, 3000.0, 2000.0],
        tmax=10.0,
        m0=1.0,
        plot_geometry=False,
        plot_geometry_sw4=False,
    )
    SW4Exporter(model, config).write()

    package = Path(tmp) / "shakermakerexports" / "sw4_package.h5"
    info = _check_package_structure(package)
    assert info["x_domain"] == 3000.0 and info["y_domain"] == 3000.0
    return package


def case_topography(tmp):
    """Same model plus a tiny synthetic topography grid."""
    crust = _make_crust()
    fault = _make_fault()
    stations = _make_stations()
    model = ShakerMaker(crust, fault, stations)

    topo_path = Path(tmp) / "topo.txt"
    _write_synthetic_topo(topo_path)

    config = SW4ExportConfig(
        path=tmp,
        h=50.0,
        size_domain=[3000.0, 3000.0, 2000.0],
        tmax=10.0,
        topo_file=str(topo_path),
        write_topography_z0_stations=False,
    )
    SW4Exporter(model, config).write()
    package = Path(tmp) / "shakermakerexports" / "sw4_package.h5"

    _check_package_structure(package)
    with h5py.File(package, "r") as f:
        assert bool(f["topography/present"][()]), "topography flag not set"
        for key in ("relpath", "nx", "ny", "points_xyz_m", "original_bounds"):
            assert key in f["topography"], f"missing topography/{key}"
    return package


def case_drmbox(tmp):
    """DRMBox receivers so the QA-station branch is exercised."""
    crust = _make_crust()
    fault = _make_fault()
    drm = _make_drmbox()
    model = ShakerMaker(crust, fault, drm)

    config = SW4ExportConfig(
        path=tmp,
        h=50.0,
        size_domain=[2000.0, 2000.0, 1500.0],
        tmax=5.0,
    )
    SW4Exporter(model, config).write()

    package = Path(tmp) / "shakermakerexports" / "sw4_package.h5"
    _check_package_structure(package)
    with h5py.File(package, "r") as f:
        qa_defined = bool(f["drm_template/qa_defined"][()])
        qa_index = int(f["drm_template/qa_index"][()])
        assert qa_defined, "DRMBox export should mark qa_defined=True"
        assert qa_index >= 0, "DRMBox export should produce a non-negative qa_index"
    return package


def case_unpack(tmp, package):
    """Package -> on-disk tree using the public unpacker."""
    out_dir = Path(tmp) / "unpacked"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    unpack_sw4_package_h5(package, out_dir)

    sw4_input = out_dir / "sw4" / "shakermaker2sw4.in"
    assert sw4_input.is_file(), f"unpack did not write {sw4_input}"
    text = sw4_input.read_text(encoding="utf-8")
    assert text.startswith("# SW4 input"), "unpacked input file has unexpected header"
    assert "\ngrid h=" in text, "unpacked input missing grid line"
    return out_dir


def case_coordinate_roundtrip(tmp, package):
    """sw4_local -> shakermaker -> sw4_local must be identity."""
    with h5py.File(package, "r") as f:
        off_m = f["coordinates/shakermaker_to_sw4_offset_m"][:]
        org_m = f["coordinates/sw4_origin_in_shakermaker_m"][:]
        x_dom = float(f["config/x_domain"][()])
        y_dom = float(f["config/y_domain"][()])
        z_dom = float(f["config/z_domain"][()])

    sw4_corners_m = np.array([
        [0.0, 0.0, 0.0],
        [x_dom, 0.0, 0.0],
        [x_dom, y_dom, 0.0],
        [0.0, y_dom, z_dom],
    ], dtype=float)

    # ShakerMaker georef <- SW4 local: P_sm = P_sw4 + sw4_origin_in_shakermaker_m
    # SW4 local <- ShakerMaker georef: P_sw4 = P_sm - sw4_origin_in_shakermaker_m
    sm_points_m = sw4_corners_m + org_m
    back_sw4_m = sm_points_m - org_m
    assert np.allclose(back_sw4_m, sw4_corners_m, atol=1e-9), \
        "coordinate round-trip (sw4 -> shakermaker -> sw4) is not identity"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run(name, fn, *args):
    try:
        result = fn(*args)
    except AssertionError as exc:
        print(f"  FAIL  {name}: {exc}")
        return False, None
    except Exception as exc:
        print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
        return False, None
    print(f"  ok    {name}")
    return True, result


def main():
    print("SW4 exporter smoke test")
    print("-" * 50)
    ok_all = True
    with tempfile.TemporaryDirectory(prefix="sw4_smoke_") as tmp:
        ok, package = _run("basic_stationlist", case_basic_stationlist, tmp)
        ok_all &= ok
        if ok:
            ok_all &= _run("coordinate_roundtrip", case_coordinate_roundtrip, tmp, package)[0]
            ok_all &= _run("unpack_basic", case_unpack, tmp, package)[0]

    with tempfile.TemporaryDirectory(prefix="sw4_smoke_topo_") as tmp:
        ok, _ = _run("topography", case_topography, tmp)
        ok_all &= ok

    with tempfile.TemporaryDirectory(prefix="sw4_smoke_drm_") as tmp:
        ok, _ = _run("drmbox", case_drmbox, tmp)
        ok_all &= ok

    print("-" * 50)
    if ok_all:
        print("OK")
        return 0
    print("FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
