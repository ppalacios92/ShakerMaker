import numpy as np
import h5py


def write_summary_h5(path, model, config, paths, source_rows, transform,
                     has_receiver_qa, n_drm_stations, qa_index):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    station_count = model._receivers.nstations
    with h5py.File(path, "w", locking=False) as hf:
        grp_paths = hf.create_group("Paths")
        for key, value in paths.items():
            grp_paths.create_dataset(key, data=str(value), dtype=string_dtype)

        grp_sw4 = hf.create_group("SW4")
        for key in ("h", "x_domain", "y_domain", "z_domain", "x_origin", "y_origin", "z_origin", "tmax", "m0"):
            grp_sw4.create_dataset(key, data=float(getattr(config, key)))
        grp_sw4.create_dataset("fileio_path", data=config.fileio_path, dtype=string_dtype)
        grp_sw4.create_dataset("supergrid_gp", data=int(config.supergrid_gp))

        grp_model = hf.create_group("Model")
        grp_model.create_dataset("source_model", data=type(model._source).__name__, dtype=string_dtype)
        grp_model.create_dataset("source_metadata", data=repr(model._source.metadata), dtype=string_dtype)
        grp_model.create_dataset("receiver_model", data=type(model._receivers).__name__, dtype=string_dtype)
        grp_model.create_dataset("receiver_metadata", data=repr(model._receivers.metadata), dtype=string_dtype)
        grp_model.create_dataset("number_of_sources", data=model._source.nsources)
        grp_model.create_dataset("number_of_receivers", data=station_count)
        grp_model.create_dataset("number_of_drm_receivers", data=n_drm_stations)
        grp_model.create_dataset("has_receiver_qa", data=has_receiver_qa)
        grp_model.create_dataset("qa_index", data=qa_index)

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
