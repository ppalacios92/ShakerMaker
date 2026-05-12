from pathlib import Path
import argparse

import h5py
import numpy as np


PACKAGE_VERSION = "1.0"


def write_sw4_package_h5(path, model, config, paths, source_rows, receiver_records,
                         file_payloads, input_text, topography_relpath=None,
                         topography_shape=None, topography_points=None,
                         topography_original_bounds=None):
    """Write a serial HDF5 transport package for an SW4 export."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(path, "w", locking=False) as hf:
        hf.attrs["package_version"] = PACKAGE_VERSION
        hf.attrs["generator"] = "ShakerMaker.sw4_exporter"
        hf.attrs["purpose"] = "transport_unpack_to_sw4_files"

        _write_manifest(hf, config, paths, file_payloads.keys(), string_dtype)
        _write_config(hf, config, string_dtype)
        _write_coordinates(hf, config)
        _write_crust(hf, model)
        _write_stations(hf, model, string_dtype)
        _write_sw4_input(hf, input_text, string_dtype)
        _write_sources(hf, source_rows, string_dtype)
        _write_topography(
            hf, topography_relpath, topography_shape, topography_points,
            topography_original_bounds, string_dtype)
        _write_receivers(hf, receiver_records, string_dtype)
        _write_drm_template(hf, config, receiver_records, string_dtype)
        _write_files(hf, file_payloads, string_dtype)


def write_unpack_script(path):
    """Copy the standalone unpacker next to the exported HDF5 package."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_UNPACK_SCRIPT, encoding="utf-8", newline="\n")


def unpack_sw4_package_h5(package_path, output_dir):
    """Recreate the SW4 export tree from a transport HDF5 package."""
    package_path = Path(package_path)
    output_dir = Path(output_dir)
    with h5py.File(package_path, "r") as hf:
        if "files" not in hf:
            raise ValueError(f"{package_path} does not contain a /files group.")
        relpaths = _read_string_array(hf["files/relpath"])
        entries = hf["files/entries"]
        for index, relpath in enumerate(relpaths):
            target = output_dir / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bytes(entries[f"file_{index:06d}/content"][:]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unpack a ShakerMaker SW4 HDF5 transport package.")
    parser.add_argument("package", nargs="?", help="Path to sw4_package.h5. If omitted, search current directory.")
    parser.add_argument("--out", default=None, help="Output directory. Default: parent of shakermakerexports, otherwise current directory.")
    args = parser.parse_args(argv)
    search_dir = Path.cwd()
    if args.package:
        candidate = Path(args.package)
        if candidate.is_dir():
            search_dir = candidate
            package = _find_package(candidate)
        else:
            package = candidate
    else:
        package = _find_package(search_dir)
    output = Path(args.out) if args.out is not None else _default_output_dir(search_dir)
    unpack_sw4_package_h5(package, output)
    print(f"Unpacked {package} into {output.resolve()}")


def _write_manifest(hf, config, paths, relpaths, string_dtype):
    grp = hf.create_group("manifest")
    grp.create_dataset("input_relpath", data="sw4/shakermaker2sw4.in", dtype=string_dtype)
    grp.create_dataset("fileio_path", data=str(config.fileio_path), dtype=string_dtype)
    grp.create_dataset(
        "created_files",
        data=np.array(list(relpaths), dtype=object),
        dtype=string_dtype,
    )


def _write_config(hf, config, string_dtype):
    grp = hf.create_group("config")
    for key in (
        "h", "x_domain", "y_domain", "z_domain", "x_origin", "y_origin", "z_origin",
        "tmax", "m0", "topo_zmax", "domain_sw4_x", "domain_sw4_y", "domain_sw4_z",
    ):
        value = getattr(config, key)
        if value is None:
            grp.create_dataset(key, data=np.nan)
        else:
            grp.create_dataset(key, data=float(value))
    for key in ("supergrid_gp",):
        grp.create_dataset(key, data=int(getattr(config, key)))
    for key in (
        "write_topography_z0_stations", "shakermaker_stations",
        "shakermaker_stations_to_surface", "domain_sw4",
    ):
        grp.create_dataset(key, data=bool(getattr(config, key)))
    for key in ("fileio_path", "station_prefix", "h5_export_name"):
        grp.create_dataset(key, data=str(getattr(config, key)), dtype=string_dtype)


def _write_coordinates(hf, config):
    grp = hf.create_group("coordinates")
    shakermaker_to_sw4_offset_m = np.asarray([
        float(config.x_origin),
        float(config.y_origin),
        float(config.z_origin),
    ], dtype=np.float64)
    grp.create_dataset("shakermaker_to_sw4_offset_m", data=shakermaker_to_sw4_offset_m)
    grp.create_dataset("sw4_origin_in_shakermaker_m", data=-shakermaker_to_sw4_offset_m)
    grp.create_dataset("shakermaker_to_sw4_offset_km", data=shakermaker_to_sw4_offset_m / 1000.0)
    grp.create_dataset("sw4_origin_in_shakermaker_km", data=-shakermaker_to_sw4_offset_m / 1000.0)


def _write_crust(hf, model):
    crust = model._crust
    grp = hf.create_group("crust")
    grp.create_dataset("thickness_km", data=np.asarray(crust.d, dtype=np.float64))
    grp.create_dataset("vp_km_s", data=np.asarray(crust.a, dtype=np.float64))
    grp.create_dataset("vs_km_s", data=np.asarray(crust.b, dtype=np.float64))
    grp.create_dataset("rho_g_cm3", data=np.asarray(crust.rho, dtype=np.float64))
    grp.create_dataset("qp", data=np.asarray(crust.qa, dtype=np.float64))
    grp.create_dataset("qs", data=np.asarray(crust.qb, dtype=np.float64))
    depth_top = np.concatenate(([0.0], np.cumsum(np.asarray(crust.d[:-1], dtype=np.float64))))
    grp.create_dataset("depth_top_km", data=depth_top)


def _write_stations(hf, model, string_dtype):
    grp = hf.create_group("stations")
    stations = list(model._receivers)
    grp.create_dataset("id", data=np.arange(len(stations), dtype=np.int32))
    xyz_km = np.asarray([station.x for station in stations], dtype=np.float64)
    if xyz_km.size == 0:
        xyz_km = np.empty((0, 3), dtype=np.float64)
    grp.create_dataset("xyz_km", data=xyz_km)
    grp.create_dataset("internal", data=np.asarray([station.is_internal for station in stations], dtype=bool))
    grp.create_dataset(
        "metadata",
        data=np.asarray([repr(station.metadata) for station in stations], dtype=object),
        dtype=string_dtype,
    )


def _write_sw4_input(hf, input_text, string_dtype):
    grp = hf.create_group("sw4_input")
    grp.create_dataset("relpath", data="sw4/shakermaker2sw4.in", dtype=string_dtype)
    grp.create_dataset("text", data=input_text, dtype=string_dtype)


def _write_sources(hf, source_rows, string_dtype):
    grp = hf.create_group("sources")
    grp.create_dataset("id", data=np.array([int(row["id"]) for row in source_rows], dtype=np.int32))
    for key in (
        "x_km", "y_km", "z_km", "x_m", "y_m", "z_m",
        "strike_deg", "dip_deg", "rake_deg", "trigger_time_s", "stf_local_t0_s", "dt",
    ):
        grp.create_dataset(key, data=np.array([float(row[key]) for row in source_rows], dtype=np.float64))
    grp.create_dataset("stf_type", data=np.array([row["stf_type"] for row in source_rows], dtype=object), dtype=string_dtype)
    grp.create_dataset("dfile", data=np.array([row["dfile"] for row in source_rows], dtype=object), dtype=string_dtype)
    npts = []
    offsets = []
    values = []
    offset = 0
    for row in source_rows:
        data = np.asarray(row["stf"].data, dtype=np.float64).reshape(-1)
        npts.append(len(data))
        offsets.append(offset)
        values.extend(data)
        offset += len(data)
    grp.create_dataset("npts", data=np.asarray(npts, dtype=np.int64))
    grp.create_dataset("data_offsets", data=np.asarray(offsets, dtype=np.int64))
    _create_array(grp, "data_values", np.asarray(values, dtype=np.float64), compress=True)


def _write_topography(hf, topography_relpath, topography_shape, topography_points,
                      topography_original_bounds, string_dtype):
    grp = hf.create_group("topography")
    grp.create_dataset("present", data=topography_relpath is not None)
    if topography_relpath is None:
        return
    grp.create_dataset("relpath", data=str(topography_relpath), dtype=string_dtype)
    grp.create_dataset("nx", data=int(topography_shape[0]))
    grp.create_dataset("ny", data=int(topography_shape[1]))
    _create_array(grp, "points_xyz_m", np.asarray(topography_points, dtype=np.float64), compress=True)
    if topography_original_bounds is not None:
        grp.create_dataset("original_bounds", data=np.asarray(topography_original_bounds, dtype=np.float64))


def _write_receivers(hf, receiver_records, string_dtype):
    grp = hf.create_group("receivers")
    grp.create_dataset("id", data=np.arange(len(receiver_records), dtype=np.int32))
    grp.create_dataset("file", data=np.array([r["file"] for r in receiver_records], dtype=object), dtype=string_dtype)
    grp.create_dataset("kind", data=np.array([r["kind"] for r in receiver_records], dtype=object), dtype=string_dtype)
    xyz_km = np.asarray([r["xyz_km"] for r in receiver_records], dtype=np.float64)
    if xyz_km.size == 0:
        xyz_km = np.empty((0, 3), dtype=np.float64)
    _create_array(grp, "xyz_km", xyz_km, compress=True)
    grp.create_dataset("internal", data=np.asarray([r["internal"] for r in receiver_records], dtype=bool))
    grp.create_dataset("is_qa", data=np.asarray([r["is_qa"] for r in receiver_records], dtype=bool))
    grp.create_dataset("model_index", data=np.asarray([r["model_index"] for r in receiver_records], dtype=np.int32))
    grp.create_dataset("metadata", data=np.array([r["metadata"] for r in receiver_records], dtype=object), dtype=string_dtype)


def _write_drm_template(hf, config, receiver_records, string_dtype):
    grp = hf.create_group("drm_template")
    non_qa = [record for record in receiver_records if not record["is_qa"]]
    qa_index = -1
    qa_record = None
    for index, record in enumerate(receiver_records):
        if record["is_qa"]:
            qa_index = int(index)
            qa_record = record
            break
    xyz_km = np.asarray([record["xyz_km"] for record in non_qa], dtype=np.float64)
    if xyz_km.size == 0:
        xyz_km = np.empty((0, 3), dtype=np.float64)
    internal = np.asarray([record["internal"] for record in non_qa], dtype=bool)
    data_location = np.arange(len(non_qa), dtype=np.int32) * 3

    if qa_record is not None:
        qa_xyz = np.asarray(qa_record["xyz_km"], dtype=np.float64).reshape(1, 3)
        qa_defined = True
        qa_file = str(qa_record["file"])
    else:
        qa_xyz = _center_xy_z0(xyz_km, config).reshape(1, 3)
        qa_defined = False
        qa_file = ""

    grp.create_dataset("xyz_km", data=xyz_km)
    grp.create_dataset("internal", data=internal)
    grp.create_dataset("data_location", data=data_location)
    grp.create_dataset("qa_xyz_km", data=qa_xyz)
    grp.create_dataset("qa_defined", data=qa_defined)
    grp.create_dataset("qa_index", data=qa_index)
    grp.create_dataset("qa_file", data=qa_file, dtype=string_dtype)
    grp.create_dataset("component_order", data="E,N,Z", dtype=string_dtype)
    grp.create_dataset("coordinate_units", data="km", dtype=string_dtype)


def _center_xy_z0(xyz_km, config):
    if len(xyz_km):
        xy = 0.5 * (xyz_km[:, :2].min(axis=0) + xyz_km[:, :2].max(axis=0))
        return np.asarray([xy[0], xy[1], 0.0], dtype=np.float64)
    else:
        sw4_origin_km = -np.asarray([
            float(config.x_origin),
            float(config.y_origin),
            float(config.z_origin),
        ], dtype=float) / 1000.0
        local_center_km = np.asarray([float(config.x_domain), float(config.y_domain), 0.0], dtype=float) / 2000.0
        return sw4_origin_km + local_center_km


def _write_files(hf, file_payloads, string_dtype):
    grp = hf.create_group("files")
    entries = grp.create_group("entries")
    relpaths = []
    for index, (relpath, content) in enumerate(file_payloads.items()):
        relpaths.append(str(relpath).replace("\\", "/"))
        entry = entries.create_group(f"file_{index:06d}")
        if isinstance(content, str):
            content = content.encode("utf-8")
        data = np.frombuffer(content, dtype=np.uint8)
        _create_array(entry, "content", data, compress=True)
    grp.create_dataset("relpath", data=np.array(relpaths, dtype=object), dtype=string_dtype)


def _find_package(directory):
    candidates = sorted(Path(directory).glob("*.h5"))
    matches = []
    for candidate in candidates:
        try:
            with h5py.File(candidate, "r") as hf:
                if hf.attrs.get("purpose", "") == "transport_unpack_to_sw4_files":
                    matches.append(candidate)
        except OSError:
            continue
    if not matches:
        raise FileNotFoundError(f"No ShakerMaker SW4 package .h5 found in {Path(directory).resolve()}")
    if len(matches) > 1:
        names = ", ".join(str(path.name) for path in matches)
        raise RuntimeError(f"More than one SW4 package found: {names}. Pass the package path explicitly.")
    return matches[0]


def _default_output_dir(directory):
    directory = Path(directory)
    if directory.name.lower() == "shakermakerexports":
        return directory.parent
    return directory


def _create_array(group, name, data, compress=False):
    data = np.asarray(data)
    if compress and data.size:
        return group.create_dataset(name, data=data, compression="gzip", compression_opts=4)
    return group.create_dataset(name, data=data)


def _read_string_array(dataset):
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in dataset[:]
    ]


_UNPACK_SCRIPT = r'''#!/usr/bin/env python3
from pathlib import Path
import argparse

import h5py


def unpack_sw4_package_h5(package_path, output_dir):
    package_path = Path(package_path)
    output_dir = Path(output_dir)
    with h5py.File(package_path, "r") as hf:
        if "files" not in hf:
            raise ValueError(f"{package_path} does not contain a /files group.")
        relpaths = _read_string_array(hf["files/relpath"])
        entries = hf["files/entries"]
        for index, relpath in enumerate(relpaths):
            target = output_dir / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bytes(entries[f"file_{index:06d}/content"][:]))


def find_package(directory):
    candidates = sorted(Path(directory).glob("*.h5"))
    matches = []
    for candidate in candidates:
        try:
            with h5py.File(candidate, "r") as hf:
                if hf.attrs.get("purpose", "") == "transport_unpack_to_sw4_files":
                    matches.append(candidate)
        except OSError:
            continue
    if not matches:
        raise FileNotFoundError(f"No ShakerMaker SW4 package .h5 found in {Path(directory).resolve()}")
    if len(matches) > 1:
        names = ", ".join(str(path.name) for path in matches)
        raise RuntimeError(f"More than one SW4 package found: {names}. Pass the package path explicitly.")
    return matches[0]


def default_output_dir(directory):
    directory = Path(directory)
    if directory.name.lower() == "shakermakerexports":
        return directory.parent
    return directory


def _read_string_array(dataset):
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in dataset[:]
    ]


def main():
    parser = argparse.ArgumentParser(description="Unpack a ShakerMaker SW4 HDF5 transport package.")
    parser.add_argument("package", nargs="?", help="Path to sw4_package.h5. If omitted, search current directory.")
    parser.add_argument("--out", default=None, help="Output directory. Default: parent of shakermakerexports, otherwise current directory.")
    args = parser.parse_args()
    search_dir = Path.cwd()
    if args.package:
        candidate = Path(args.package)
        if candidate.is_dir():
            search_dir = candidate
            package = find_package(candidate)
        else:
            package = candidate
    else:
        package = find_package(search_dir)
    output = Path(args.out) if args.out is not None else default_output_dir(search_dir)
    unpack_sw4_package_h5(package, output)
    print(f"Unpacked {package} into {output.resolve()}")


if __name__ == "__main__":
    main()
'''


if __name__ == "__main__":
    main()
