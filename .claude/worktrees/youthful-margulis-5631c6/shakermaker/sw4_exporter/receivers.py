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


def topography_receiver_lines(local_points, start_index=1, prefix="sf"):
    """Receivers at the topography free surface (depth=0 in SW4)."""
    lines = []
    for offset, (x, y, _z) in enumerate(np.asarray(local_points, dtype=float), start=start_index):
        lines.append(f"rec x={x:.1f} y={y:.1f} depth=0 file={prefix}{offset:05d} usgsformat=1")
    return lines


def topography_z0_receiver_lines(local_points, h, start_index=1, prefix="sf"):
    """Receivers between the topography surface and the z=0 plane, stepping by h.

    For each topo point with height tz > 0, stations are placed at elevations
    tz, tz-h, ..., h (stopping before z=0).  Written as z=-elevation so that
    SW4 sees them as coordinates above the z=0 datum (negative z in SW4).
    """
    lines = []
    offset = start_index
    h = float(h)
    for (x, y, tz) in np.asarray(local_points, dtype=float):
        if tz <= 0.0:
            continue
        z_values = np.arange(tz, 0.0, -h)
        for z_val in z_values:
            if z_val > 0.0:
                lines.append(
                    f"rec x={x:.1f} y={y:.1f} z={-z_val:.6f} "
                    f"file={prefix}{offset:05d} usgsformat=1"
                )
                offset += 1
    return lines


def domain_receiver_lines(h, domain_size, start_index=1, prefix="sf"):
    """Receivers on a regular grid within [0,Lx] x [0,Ly] x [0,Lz], spaced by h.

    Written as depth= so SW4 interprets them as depths below the free surface.
    """
    lx, ly, lz = float(domain_size[0]), float(domain_size[1]), float(domain_size[2])
    h = float(h)
    xs = np.arange(0.0, lx + 0.5 * h, h)
    ys = np.arange(0.0, ly + 0.5 * h, h)
    zs = np.arange(0.0, lz + 0.5 * h, h)
    lines = []
    offset = start_index
    for z in zs:
        for y in ys:
            for x in xs:
                lines.append(
                    f"rec x={x:.1f} y={y:.1f} depth={z:.1f} "
                    f"file={prefix}{offset:05d} usgsformat=1"
                )
                offset += 1
    return lines
