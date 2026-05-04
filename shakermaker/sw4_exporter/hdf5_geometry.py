import numpy as np
import h5py

from .receivers import domain_receiver_bounds


def write_geometry_h5(path, model, transform, dt, tmax, config, has_receiver_qa,
                      n_drm_stations, qa_index, receiver_records=None):
    if receiver_records is None:
        receiver_records = _records_from_model(model, transform, has_receiver_qa, n_drm_stations)

    drm_records = [record for record in receiver_records if not record["is_qa"]]
    qa_records = [record for record in receiver_records if record["is_qa"]]

    xyz = _xyz_array(drm_records)
    internal = np.asarray([record["internal"] for record in drm_records], dtype=bool)
    n_drm_stations = len(drm_records)

    if qa_records:
        qa_xyz = np.asarray(qa_records[0]["xyz_km"], dtype=np.float64).reshape(1, 3)
        qa_defined = True
        qa_index_out = _record_index(receiver_records, qa_records[0])
    else:
        qa_xyz = _center_xy_z0(xyz, config).reshape(1, 3)
        qa_defined = False
        qa_index_out = -1

    nt = int(round(float(tmax) / float(dt))) + 1
    data_location = np.arange(0, n_drm_stations, dtype=np.int32) * 3
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(path, "w", locking=False) as hf:
        grp_data = hf.create_group("DRM_Data")
        grp_qa = hf.create_group("DRM_QA_Data")
        grp_meta = hf.create_group("DRM_Metadata")
        grp_receivers = hf.create_group("SW4_Receivers")
        grp_data.create_dataset("xyz", data=xyz, dtype=np.float64)
        grp_data.create_dataset("internal", data=internal, dtype=bool)
        grp_data.create_dataset("data_location", data=data_location, dtype=np.int32)
        grp_data.create_dataset("velocity", shape=(3 * n_drm_stations, nt), dtype=np.float64, fillvalue=0.0)
        grp_data.create_dataset("displacement", shape=(3 * n_drm_stations, nt), dtype=np.float64, fillvalue=0.0)
        grp_data.create_dataset("acceleration", shape=(3 * n_drm_stations, nt), dtype=np.float64, fillvalue=0.0)
        grp_qa.create_dataset("xyz", data=qa_xyz, dtype=np.float64)
        grp_qa.create_dataset("velocity", shape=(3, nt), dtype=np.float64, fillvalue=0.0)
        grp_qa.create_dataset("displacement", shape=(3, nt), dtype=np.float64, fillvalue=0.0)
        grp_qa.create_dataset("acceleration", shape=(3, nt), dtype=np.float64, fillvalue=0.0)
        grp_meta.create_dataset("dt", data=float(dt))
        grp_meta.create_dataset("tstart", data=0.0)
        grp_meta.create_dataset("tend", data=float(tmax))
        grp_meta.create_dataset("nt", data=nt)
        grp_meta.create_dataset("component_order", data="E,N,Z", dtype=string_dtype)
        grp_meta.create_dataset("coordinate_units", data="km", dtype=string_dtype)
        grp_meta.create_dataset("program_used", data="ShakerMaker", dtype=string_dtype)
        grp_meta.create_dataset("writer_mode", data="geometry_only", dtype=string_dtype)
        grp_meta.create_dataset("qa_defined", data=qa_defined)
        grp_meta.create_dataset("qa_index", data=qa_index_out)
        grp_meta.create_dataset("h", data=float(config.h))
        grp_meta.create_dataset("x_domain", data=float(config.x_domain))
        grp_meta.create_dataset("y_domain", data=float(config.y_domain))
        grp_meta.create_dataset("z_domain", data=float(config.z_domain))
        grp_meta.create_dataset("write_topography_z0_stations", data=bool(config.write_topography_z0_stations))
        grp_meta.create_dataset("domain_sw4", data=bool(config.domain_sw4))
        grp_meta.create_dataset("domain_sw4_size", data=_domain_sw4_size_array(config))
        grp_meta.create_dataset("domain_sw4_bounds", data=_domain_sw4_bounds_array(config))

        grp_receivers.create_dataset("id", data=np.arange(len(receiver_records), dtype=np.int32))
        grp_receivers.create_dataset("file", data=np.array([r["file"] for r in receiver_records], dtype=object), dtype=string_dtype)
        grp_receivers.create_dataset("kind", data=np.array([r["kind"] for r in receiver_records], dtype=object), dtype=string_dtype)
        grp_receivers.create_dataset("xyz_km", data=_xyz_array(receiver_records))
        grp_receivers.create_dataset("internal", data=np.asarray([r["internal"] for r in receiver_records], dtype=bool))
        grp_receivers.create_dataset("is_qa", data=np.asarray([r["is_qa"] for r in receiver_records], dtype=bool))
        grp_receivers.create_dataset("model_index", data=np.asarray([r["model_index"] for r in receiver_records], dtype=np.int32))
        grp_receivers.create_dataset("metadata", data=np.array([r["metadata"] for r in receiver_records], dtype=object), dtype=string_dtype)


def _records_from_model(model, transform, has_receiver_qa, n_drm_stations):
    records = []
    qa_index = n_drm_stations if has_receiver_qa else -1
    for i_station, station in enumerate(model._receivers):
        records.append({
            "file": f"sf{i_station + 1:05d}",
            "kind": "shakermaker",
            "xyz_km": transform.from_shakermaker_km_to_sw4_km(station.x),
            "internal": bool(station.is_internal),
            "is_qa": bool(i_station == qa_index),
            "model_index": int(i_station),
            "metadata": repr(station.metadata),
        })
    return records


def _center_xy_z0(xyz, config):
    if len(xyz):
        xy = 0.5 * (xyz[:, :2].min(axis=0) + xyz[:, :2].max(axis=0))
    else:
        xy = np.asarray([float(config.x_domain), float(config.y_domain)], dtype=float) / 2000.0
    return np.asarray([xy[0], xy[1], 0.0], dtype=np.float64)


def _xyz_array(records):
    if not records:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray([record["xyz_km"] for record in records], dtype=np.float64)


def _record_index(records, target):
    for index, record in enumerate(records):
        if record is target:
            return int(index)
    return -1


def _domain_sw4_size_array(config):
    if config.domain_sw4_size is None:
        return np.asarray([config.x_domain, config.y_domain, config.z_domain], dtype=np.float64)
    return np.asarray(config.domain_sw4_size, dtype=np.float64)


def _domain_sw4_bounds_array(config):
    bounds = domain_receiver_bounds(
        config.x_domain, config.y_domain, config.z_domain, config.domain_sw4_size)
    return np.asarray(bounds, dtype=np.float64)
