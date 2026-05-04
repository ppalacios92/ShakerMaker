import numpy as np


def model_receiver_lines(receivers, transform, prefix="sf", start_index=1, topo_xy_z0=None):
    """ShakerMaker model stations written with z= (absolute SW4 coordinates).

    topo_xy_z0: set of (round(x), round(y)) topo positions where topo_z≈0.
    Stations at z=0 that coincide with those positions are skipped because
    the topography depth=0 station already covers them.
    """
    topo_xy_z0 = topo_xy_z0 or set()
    lines = []
    offset = start_index
    for station in receivers:
        x_m = transform.from_shakermaker_km_to_sw4_m(station.x)
        x, y, z = float(x_m[0]), float(x_m[1]), float(x_m[2])
        if z == 0.0 and (round(x), round(y)) in topo_xy_z0:
            continue
        lines.append(
            f"rec x={x:.16g} y={y:.16g} z={z:.16g} "
            f"file={prefix}{offset:05d} usgsformat=1"
        )
        offset += 1
    return lines


def topography_receiver_lines(local_points, start_index=1, prefix="sf"):
    """Topography free-surface stations — always depth=0 in SW4."""
    lines = []
    for offset, (x, y, _z) in enumerate(np.asarray(local_points, dtype=float), start=start_index):
        lines.append(f"rec x={x:.1f} y={y:.1f} depth=0 file={prefix}{offset:05d} usgsformat=1")
    return lines


def topography_z0_receiver_lines(local_points, h, start_index=1, prefix="sf"):
    """Stations between the topography surface and z=0, stepping by h.

    Uses depth=k*h (relative to the free surface) so SW4 Laplacian smoothing
    of the topography cannot push any station above the surface.
    Stations go from depth=h to depth=floor((tz - h/2) / h) * h.
    The z=0 plane is never written here (covered by the domain grid starting at z=h).
    """
    lines = []
    offset = start_index
    h = float(h)
    for (x, y, tz) in np.asarray(local_points, dtype=float):
        if tz <= 0.0:
            continue
        # Number of h-steps that fit strictly below the surface and above z=0
        # (discard the last step if it would land within h/2 of z=0)
        n_steps = max(0, int((tz - h / 2.0) / h))
        for k in range(1, n_steps + 1):
            depth = k * h
            lines.append(
                f"rec x={x:.1f} y={y:.1f} depth={depth:.1f} "
                f"file={prefix}{offset:05d} usgsformat=1"
            )
            offset += 1
    return lines


def domain_receiver_bounds(x_domain, y_domain, z_domain, domain_sw4_size=None):
    """Return [xmin, xmax, ymin, ymax, zmin, zmax] of the receiver sub-domain."""
    if domain_sw4_size is not None and len(domain_sw4_size) == 3:
        lx, ly, lz = float(domain_sw4_size[0]), float(domain_sw4_size[1]), float(domain_sw4_size[2])
    else:
        lx, ly, lz = float(x_domain), float(y_domain), float(z_domain)
    return [0.0, lx, 0.0, ly, 0.0, lz]


def domain_receiver_lines(h, domain_size, start_index=1, prefix="sf", topo_xy_z0=None):
    """Regular grid of receivers in [0,Lx]x[0,Ly]x[0,Lz] spaced by h.

    Written with z= (absolute SW4 coordinates, positive = below datum).
    Stations at z=0 that coincide with topo_xy_z0 positions are skipped
    because the topography depth=0 station already covers those points.
    """
    topo_xy_z0 = topo_xy_z0 or set()
    lx, ly, lz = float(domain_size[0]), float(domain_size[1]), float(domain_size[2])
    h = float(h)
    xs = np.arange(0.0, lx + 0.5 * h, h)
    ys = np.arange(0.0, ly + 0.5 * h, h)
    zs = np.arange(h, lz + 0.5 * h, h)    # start at h, never at z=0
    lines = []
    offset = start_index
    for z in zs:
        for y in ys:
            for x in xs:
                if z == 0.0 and (round(x), round(y)) in topo_xy_z0:
                    continue
                lines.append(
                    f"rec x={x:.1f} y={y:.1f} z={z:.1f} "
                    f"file={prefix}{offset:05d} usgsformat=1"
                )
                offset += 1
    return lines
