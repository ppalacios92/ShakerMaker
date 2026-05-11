from pathlib import Path
import numpy as np

SEPARATOR = "-" * 50


def read_cartesian_topography(path):
    path = Path(path)
    lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty topography file: {path}")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Invalid topography header in {path}: {lines[0]}")

    nx, ny = int(header[0]), int(header[1])
    points = np.array([[float(v) for v in line.split()[:3]] for line in lines[1:]], dtype=float)
    expected = nx * ny
    if points.shape[0] != expected:
        raise ValueError(f"Topography point count mismatch in {path}: expected {expected}, got {points.shape[0]}")

    return nx, ny, points


def rotate_topography_to_shakermaker(points):
    points = np.asarray(points, dtype=float)
    rotated = points.copy()
    rotated[:, 0] = points[:, 1]
    rotated[:, 1] = points[:, 0]
    return rotated


def rebuild_cartesian_topography(points):
    points = np.asarray(points, dtype=float)
    xs = np.unique(points[:, 0])
    ys = np.unique(points[:, 1])
    z_by_xy = {(float(x), float(y)): float(z) for x, y, z in points}

    rebuilt = []
    for y in ys:
        for x in xs:
            key = (float(x), float(y))
            if key not in z_by_xy:
                raise ValueError("Topography points must form a complete cartesian grid.")
            rebuilt.append((x, y, z_by_xy[key]))

    return len(xs), len(ys), np.asarray(rebuilt, dtype=float)


def bounds(points):
    points = np.asarray(points, dtype=float)
    return (
        float(points[:, 0].min()), float(points[:, 0].max()),
        float(points[:, 1].min()), float(points[:, 1].max()),
        float(points[:, 2].min()), float(points[:, 2].max()),
    )


def minimum_corner(points):
    points = np.asarray(points, dtype=float)
    return np.array([
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 2].min()),
    ], dtype=float)


def cartesian_topography_text(nx, ny, points):
    lines = [f"{nx} {ny}"]
    for x, y, z in np.asarray(points, dtype=float):
        lines.append(f"{x:.1f} {y:.1f} {z:.6f}")
    return "\n".join(lines) + "\n"


def extend_topography_to_domain(nx, ny, points, x_domain, y_domain):
    points = np.asarray(points, dtype=float)
    xs = np.unique(points[:, 0])
    ys = np.unique(points[:, 1])
    if len(xs) != nx or len(ys) != ny:
        raise ValueError("Topography points must form a regular cartesian grid.")

    dx = float(np.median(np.diff(xs))) if len(xs) > 1 else float(x_domain)
    dy = float(np.median(np.diff(ys))) if len(ys) > 1 else float(y_domain)
    new_xs = _axis_to_domain(0.0, x_domain, dx)
    new_ys = _axis_to_domain(0.0, y_domain, dy)

    z_grid = points[:, 2].reshape(ny, nx)
    extended = []
    for y in new_ys:
        y_clamped = min(max(y, ys[0]), ys[-1])
        old_j = int(np.abs(ys - y_clamped).argmin())
        for x in new_xs:
            x_clamped = min(max(x, xs[0]), xs[-1])
            old_i = int(np.abs(xs - x_clamped).argmin())
            extended.append((x, y, z_grid[old_j, old_i]))

    print(f"Extended topography spacing dx={dx:.6g} dy={dy:.6g}")
    return len(new_xs), len(new_ys), np.asarray(extended, dtype=float)


def _axis_to_domain(start, end, step):
    values = list(np.arange(float(start), float(end) + 0.5 * step, float(step)))
    if values[-1] < float(end):
        values.append(float(end))
    elif values[-1] > float(end):
        values[-1] = float(end)
    return np.asarray(values, dtype=float)


def _grid_size(length, h):
    return int(round(float(length) / float(h))) + 1


def print_topography_diagnostics(original_points, local_points, x_domain, y_domain, z_domain,
                                 h=None, topo_nx=None, topo_ny=None, topo_zmax=None):
    original_centroid = np.asarray(original_points, dtype=float).mean(axis=0)
    local_centroid = np.asarray(local_points, dtype=float).mean(axis=0)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds(original_points)
    lxmin, lxmax, lymin, lymax, lzmin, lzmax = bounds(local_points)
    print(SEPARATOR)
    print("Topography bounds")
    print(SEPARATOR)
    print(f"Original  x=[{xmin:.1f}, {xmax:.1f}]  y=[{ymin:.1f}, {ymax:.1f}]  z=[{zmin:.2f}, {zmax:.2f}]")
    print(f"Centroid  [{original_centroid[0]:.1f}, {original_centroid[1]:.1f}, {original_centroid[2]:.2f}]")
    print("")
    print(f"Local     x=[{lxmin:.1f}, {lxmax:.1f}]  y=[{lymin:.1f}, {lymax:.1f}]  z=[{lzmin:.2f}, {lzmax:.2f}]")
    print(f"Centroid  [{local_centroid[0]:.1f}, {local_centroid[1]:.1f}, {local_centroid[2]:.2f}]")
    print("")
    print(f"SW4 grid  x=[0.0, {x_domain:.1f}]  y=[0.0, {y_domain:.1f}]  z=[0.0, {z_domain:.1f}]")
    if h is not None:
        nx = _grid_size(x_domain, h)
        ny = _grid_size(y_domain, h)
        nz = _grid_size(z_domain, h)
        print("")
        print("Grid estimate")
        if topo_nx is not None and topo_ny is not None:
            print(f"Topography samples  Nx={int(topo_nx)}  Ny={int(topo_ny)}")
        if topo_zmax is not None:
            print(f"Topography zmax     {float(topo_zmax):.1f}")
        print(f"SW4 grid            h={float(h):.1f}  Nx={nx}  Ny={ny}  Nz={nz}  Points={nx * ny * nz}")
    if lxmin > 0.0 or lymin > 0.0:
        print("WARNING: local topography starts above x=0 or y=0; SW4 may require coverage at the grid minimum.")
    if lxmin < 0.0 or lymin < 0.0:
        print("WARNING: local topography has negative x/y coordinates.")
    if lxmax < x_domain or lymax < y_domain:
        print("WARNING: local topography does not cover the full SW4 grid domain.")


def print_domain_diagnostics(x_domain, y_domain, z_domain, h=None):
    print(SEPARATOR)
    print("SW4 grid domain")
    print(SEPARATOR)
    print(f"SW4 grid domain     x=[0.0, {float(x_domain):.1f}], y=[0.0, {float(y_domain):.1f}], z=[0.0, {float(z_domain):.1f}]")
    if h is not None:
        nx = _grid_size(x_domain, h)
        ny = _grid_size(y_domain, h)
        nz = _grid_size(z_domain, h)
        print(f"Grid estimate       h={float(h):.1f}, Nx={nx}, Ny={ny}, Nz={nz}, Points={nx * ny * nz}")


def print_active_geometry_bounds(points):
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return
    xmin, xmax, ymin, ymax, zmin, zmax = bounds(points)
    centroid = points.mean(axis=0)
    print(SEPARATOR)
    print("Active geometry bounds")
    print(SEPARATOR)
    print(f"Active    x=[{xmin:.1f}, {xmax:.1f}]  y=[{ymin:.1f}, {ymax:.1f}]  z=[{zmin:.2f}, {zmax:.2f}]")
    print(f"Centroid  [{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.2f}]")
