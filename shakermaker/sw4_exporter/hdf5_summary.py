import numpy as np
import h5py

from .receivers import domain_receiver_bounds


def write_summary_h5(path, model, config, paths, source_rows, transform,
                     has_receiver_qa, n_drm_stations, qa_index, receiver_records=None):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    station_count = model._receivers.nstations
    if receiver_records is None:
        receiver_records = []
    exported_drm_count = len([r for r in receiver_records if not r["is_qa"]]) if receiver_records else n_drm_stations
    exported_qa = [r for r in receiver_records if r["is_qa"]]
    exported_qa_index = _record_index(receiver_records, exported_qa[0]) if exported_qa else -1
    with h5py.File(path, "w", locking=False) as hf:
        grp_paths = hf.create_group("Paths")
        for key, value in paths.items():
            grp_paths.create_dataset(key, data=str(value), dtype=string_dtype)

        grp_sw4 = hf.create_group("SW4")
        for key in ("h", "x_domain", "y_domain", "z_domain", "x_origin", "y_origin", "z_origin", "tmax", "m0"):
            grp_sw4.create_dataset(key, data=float(getattr(config, key)))
        grp_sw4.create_dataset("fileio_path", data=config.fileio_path, dtype=string_dtype)
        grp_sw4.create_dataset("supergrid_gp", data=int(config.supergrid_gp))
        grp_sw4.create_dataset("write_topography_z0_stations", data=bool(config.write_topography_z0_stations))
        grp_sw4.create_dataset("domain_sw4", data=bool(config.domain_sw4))
        grp_sw4.create_dataset("domain_sw4_size", data=_domain_sw4_size_array(config))
        grp_sw4.create_dataset("domain_sw4_bounds", data=_domain_sw4_bounds_array(config))

        grp_model = hf.create_group("Model")
        grp_model.create_dataset("source_model", data=type(model._source).__name__, dtype=string_dtype)
        grp_model.create_dataset("source_metadata", data=repr(model._source.metadata), dtype=string_dtype)
        grp_model.create_dataset("receiver_model", data=type(model._receivers).__name__, dtype=string_dtype)
        grp_model.create_dataset("receiver_metadata", data=repr(model._receivers.metadata), dtype=string_dtype)
        grp_model.create_dataset("number_of_sources", data=model._source.nsources)
        grp_model.create_dataset("number_of_receivers", data=station_count)
        grp_model.create_dataset("number_of_model_drm_receivers", data=n_drm_stations)
        grp_model.create_dataset("number_of_drm_receivers", data=exported_drm_count)
        grp_model.create_dataset("number_of_exported_receivers", data=len(receiver_records))
        grp_model.create_dataset("has_receiver_qa", data=bool(exported_qa))
        grp_model.create_dataset("qa_index", data=exported_qa_index)

        grp_crust = hf.create_group("Crust")
        depth_top = np.concatenate(([0.0], np.cumsum(model._crust.d[:-1])))
        grp_crust.create_dataset("depth_top_km", data=depth_top, dtype=np.float64)
        grp_crust.create_dataset("thickness_km", data=model._crust.d, dtype=np.float64)
        grp_crust.create_dataset("vp_km_s", data=model._crust.a, dtype=np.float64)
        grp_crust.create_dataset("vs_km_s", data=model._crust.b, dtype=np.float64)
        grp_crust.create_dataset("rho_g_cm3", data=model._crust.rho, dtype=np.float64)
        grp_crust.create_dataset("qp", data=model._crust.qa, dtype=np.float64)
        grp_crust.create_dataset("qs", data=model._crust.qb, dtype=np.float64)

        grp_sources = hf.create_group("Sources")
        grp_sources.create_dataset("id", data=np.arange(len(source_rows), dtype=np.int32))
        for key in ("x_km", "y_km", "z_km", "x_m", "y_m", "z_m",
                    "x_sw4_m", "y_sw4_m", "z_sw4_m",
                    "strike_deg", "dip_deg", "rake_deg", "trigger_time_s", "stf_local_t0_s", "dt"):
            grp_sources.create_dataset(key, data=np.array([float(row[key]) for row in source_rows]))
        grp_sources.create_dataset("stf_type", data=np.array([row["stf_type"] for row in source_rows], dtype=object), dtype=string_dtype)
        grp_sources.create_dataset("dfile", data=np.array([row["dfile"] for row in source_rows], dtype=object), dtype=string_dtype)

        grp_receivers = hf.create_group("Receivers")
        xyz_km = np.array([np.asarray(st.x, dtype=float) for st in model._receivers])
        xyz_sw4_km = xyz_km + transform.origin_km
        grp_receivers.create_dataset("id", data=np.arange(station_count, dtype=np.int32))
        grp_receivers.create_dataset("xyz_km", data=xyz_km, dtype=np.float64)
        grp_receivers.create_dataset("xyz_sw4_km", data=xyz_sw4_km, dtype=np.float64)
        grp_receivers.create_dataset("internal", data=np.array([st.is_internal for st in model._receivers], dtype=bool))
        grp_receivers.create_dataset("metadata", data=np.array([repr(st.metadata) for st in model._receivers], dtype=object), dtype=string_dtype)

        if receiver_records:
            grp_exported = hf.create_group("SW4_Receivers")
            grp_exported.create_dataset("id", data=np.arange(len(receiver_records), dtype=np.int32))
            grp_exported.create_dataset("file", data=np.array([r["file"] for r in receiver_records], dtype=object), dtype=string_dtype)
            grp_exported.create_dataset("kind", data=np.array([r["kind"] for r in receiver_records], dtype=object), dtype=string_dtype)
            grp_exported.create_dataset("xyz_km", data=np.asarray([r["xyz_km"] for r in receiver_records], dtype=np.float64))
            grp_exported.create_dataset("internal", data=np.asarray([r["internal"] for r in receiver_records], dtype=bool))
            grp_exported.create_dataset("is_qa", data=np.asarray([r["is_qa"] for r in receiver_records], dtype=bool))
            grp_exported.create_dataset("model_index", data=np.asarray([r["model_index"] for r in receiver_records], dtype=np.int32))
            grp_exported.create_dataset("metadata", data=np.array([r["metadata"] for r in receiver_records], dtype=object), dtype=string_dtype)


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
