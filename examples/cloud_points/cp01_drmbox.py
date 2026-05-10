from shakermaker.sl_extensions import DRMBox

x0 = [10., 10., 0.]
nx, ny, nz = 8, 8, 4
dx = 0.1

drm = DRMBox(x0, [nx, ny, nz], [dx, dx, dx], metadata={"name": "drm_box"})
print(f"DRM box stations: {len(drm.stations.list)}")
print(f"Internal: {sum(drm.stationlist_internal)}")
print(f"Boundary: {sum(drm.stationlist_boundary)}")
