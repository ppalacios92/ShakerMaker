import numpy as np


def model_receiver_lines(receivers, transform, prefix="sf", start_index=1, topo_xy_z0=None):
    topo_xy_z0 = topo_xy_z0 or set()
    lines = []
    offset = int(start_index)
    for station in receivers:
        x, y, z = transform.from_shakermaker_km_to_sw4_m(station.x)
        if abs(float(z)) < 1.0e-9 and (round(float(x)), round(float(y))) in topo_xy_z0:
            continue
        lines.append(
            f"rec x={float(x):.16g} y={float(y):.16g} z={float(z):.16g} "
            f"file={prefix}{offset:05d} usgsformat=1"
        )
        offset += 1
    return lines


def model_receiver_surface_lines(receivers, transform, prefix="sf", start_index=1):
    lines = []
    offset = int(start_index)
    for station in receivers:
        x, y, _z = transform.from_shakermaker_km_to_sw4_m(station.x)
        lines.append(
            f"rec x={float(x):.16g} y={float(y):.16g} depth=0 "
            f"file={prefix}{offset:05d} usgsformat=1"
        )
        offset += 1
    return lines


def topography_receiver_lines(local_points, start_index=1, prefix="sf"):
    lines = []
    for offset, (x, y, _z) in enumerate(np.asarray(local_points, dtype=float), start=start_index):
        lines.append(f"rec x={x:.1f} y={y:.1f} depth=0 file={prefix}{offset:05d} usgsformat=1")
    return lines


def topography_z0_receiver_lines(local_points, h, start_index=1, prefix="sf"):
    lines = []
    offset = int(start_index)
    h = float(h)
    for x, y, topo_z in np.asarray(local_points, dtype=float):
        for z in _values_between_topography_and_z0(topo_z, h):
            lines.append(
                f"rec x={x:.1f} y={y:.1f} z={z:.16g} "
                f"file={prefix}{offset:05d} usgsformat=1"
            )
            offset += 1
    return lines


def domain_receiver_lines(h, domain_size, start_index=1, prefix="sf", topo_xy_z0=None,
                          origin_xy=(0.0, 0.0)):
    topo_xy_z0 = topo_xy_z0 or set()
    lx, ly, lz = [float(value) for value in domain_size]
    ox, oy = [float(value) for value in origin_xy]
    h = float(h)
    xs = ox + _grid_values(0.0, lx, h)
    ys = oy + _grid_values(0.0, ly, h)
    zs = _grid_values(h, lz, h)
    lines = []
    offset = int(start_index)
    for z in zs:
        for y in ys:
            for x in xs:
                if abs(float(z)) < 1.0e-9 and (round(float(x)), round(float(y))) in topo_xy_z0:
                    continue
                lines.append(
                    f"rec x={x:.1f} y={y:.1f} z={z:.1f} "
                    f"file={prefix}{offset:05d} usgsformat=1"
                )
                offset += 1
    return lines


def domain_receiver_bounds(x_domain, y_domain, z_domain, domain_sw4_size=None):
    if domain_sw4_size is not None and len(domain_sw4_size) == 3:
        lx, ly, lz = [float(value) for value in domain_sw4_size]
        ox = 0.5 * (float(x_domain) - lx)
        oy = 0.5 * (float(y_domain) - ly)
        return [ox, ox + lx, oy, oy + ly, 0.0, lz]
    else:
        lx, ly, lz = float(x_domain), float(y_domain), float(z_domain)
        return [0.0, lx, 0.0, ly, 0.0, lz]


def _grid_values(start, stop, step):
    values = np.arange(float(start), float(stop) + 0.5 * float(step), float(step))
    return values[values <= float(stop) + 1.0e-9]


def _values_between_topography_and_z0(topo_z, h):
    topo_z = float(topo_z)
    h = float(h)
    if abs(topo_z) < 1.0e-9:
        return []
    values = np.arange(h, abs(topo_z), h)
    if len(values) == 0:
        return values
    if topo_z > 0.0:
        return -values
    return values
