import numpy as np
import h5py


def write_geometry_h5(path, model, transform, dt, tmax, config, has_receiver_qa, n_drm_stations, qa_index):
    origin_km = transform.origin_km
    xyz = np.zeros((n_drm_stations, 3), dtype=np.float64)
    internal = np.zeros(n_drm_stations, dtype=bool)
    for i_station in range(n_drm_stations):
        station = model._receivers.get_station_by_id(i_station)
        xyz[i_station, :] = np.asarray(station.x, dtype=float) + origin_km
        internal[i_station] = station.is_internal

    if has_receiver_qa:
        qa_station = model._receivers.get_station_by_id(n_drm_stations)
        qa_xyz = (np.asarray(qa_station.x, dtype=float) + origin_km).reshape(1, 3)
    else:
        qa_xyz = origin_km.reshape(1, 3)

    nt = int(round(float(tmax) / float(dt))) + 1
    data_location = np.arange(0, n_drm_stations, dtype=np.int32) * 3
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(path, "w", locking=False) as hf:
        grp_data = hf.create_group("DRM_Data")
        grp_qa = hf.create_group("DRM_QA_Data")
        grp_meta = hf.create_group("DRM_Metadata")
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
        grp_meta.create_dataset("qa_defined", data=has_receiver_qa)
        grp_meta.create_dataset("qa_index", data=qa_index)
        grp_meta.create_dataset("h", data=float(config.h))
        grp_meta.create_dataset("x_domain", data=float(config.x_domain))
        grp_meta.create_dataset("y_domain", data=float(config.y_domain))
        grp_meta.create_dataset("z_domain", data=float(config.z_domain))
