import numpy as np


def model_receiver_lines(receivers, transform, prefix="sf"):
    lines = []
    for i_station, station in enumerate(receivers):
        x_m = transform.from_shakermaker_km_to_sw4_m(station.x)
        lines.append(
            f"rec x={x_m[0]:.16g} y={x_m[1]:.16g} depth={x_m[2]:.16g} "
            f"file={prefix}{i_station + 1:05d} usgsformat=1"
        )
    return lines


def topography_receiver_lines(local_points, start_index=1, prefix="sf", depth_from_topography=False):
    lines = []
    for offset, (x, y, z) in enumerate(np.asarray(local_points, dtype=float), start=start_index):
        if depth_from_topography:
            lines.append(f"rec x={x:.1f} y={y:.1f} z={z:.6f} file={prefix}{offset:05d} usgsformat=1")
        else:
            lines.append(f"rec x={x:.1f} y={y:.1f} depth=0 file={prefix}{offset:05d} usgsformat=1")
    return lines
