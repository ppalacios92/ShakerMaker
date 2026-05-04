from pathlib import Path

import numpy as np


def _token_value(tokens, key):
    prefix = f"{key}="
    for token in tokens:
        if token.startswith(prefix):
            return float(token[len(prefix):])
    return None


def _token_str(tokens, key):
    prefix = f"{key}="
    for token in tokens:
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def _read_grid(tokens):
    x = _token_value(tokens, "x")
    y = _token_value(tokens, "y")
    z = _token_value(tokens, "z")
    if None in (x, y, z):
        raise ValueError("SW4 grid line must include x, y, and z.")
    return x, y, z


def _read_point(tokens):
    """Read a source or rec line.

    Convention (matching SW4 z-axis):
      z > 0  →  below the datum (depth, inside the domain box)
      z < 0  →  above the datum (elevation, above the domain box)
      depth= d  →  d metres below the free surface (always >= 0)
    """
    x = _token_value(tokens, "x")
    y = _token_value(tokens, "y")
    z = _token_value(tokens, "z")
    depth = _token_value(tokens, "depth")
    if x is None or y is None:
        return None
    # depth= means "below the free surface", positive going down → same sign as z
    if depth is not None:
        return x, y, depth      # depth=0 → z=0 (at the datum surface)
    if z is not None:
        return x, y, z          # z<0 means above datum, z>0 below datum
    return x, y, 0.0


def read_sw4_geometry(path):
    """Parse a SW4 .in file and return grid, source points, receiver points,
    and optionally the topography points read from the referenced .topo file.

    Z convention in the returned arrays:
      z > 0  →  depth below the datum (inside the domain)
      z < 0  →  elevation above the datum (topography zone)
    """
    path = Path(path)
    grid = None
    sources = []
    receivers = []
    topo_rel_file = None

    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        tokens = line.split()
        if tokens[0] == "grid":
            grid = _read_grid(tokens)
        elif tokens[0] == "source":
            point = _read_point(tokens)
            if point is not None:
                sources.append(point)
        elif tokens[0] == "rec":
            point = _read_point(tokens)
            if point is not None:
                receivers.append(point)
        elif tokens[0] == "topography":
            topo_rel_file = _token_str(tokens, "file")

    if grid is None:
        raise ValueError(f"No grid line found in {path}.")

    # Read the topo file referenced in the .in (path is relative to the .in folder)
    topo_points = None
    if topo_rel_file is not None:
        topo_abs = path.parent / topo_rel_file
        try:
            from .topography import read_cartesian_topography
            _nx, _ny, topo_points = read_cartesian_topography(topo_abs)
        except Exception:
            pass

    return (
        grid,
        np.asarray(sources, dtype=float) if sources else np.empty((0, 3)),
        np.asarray(receivers, dtype=float) if receivers else np.empty((0, 3)),
        topo_points,
    )


def plot_sw4_geometry(path):
    try:
        import pyvista as pv
        from pyvistaqt import QtInteractor
        from PyQt5 import QtWidgets
    except ImportError as exc:
        raise ImportError(
            "plot_geometry=True requires pyvista, pyvistaqt, and PyQt5."
        ) from exc

    path = Path(path)
    (grid_x, grid_y, grid_z), sources, receivers, topo_points = read_sw4_geometry(path)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    window = QtWidgets.QMainWindow()
    window.setWindowTitle(f"SW4 geometry - {path.name}")
    window.resize(1200, 800)

    frame = QtWidgets.QFrame()
    layout = QtWidgets.QVBoxLayout(frame)
    plotter = QtInteractor(frame)
    layout.addWidget(plotter.interactor)
    window.setCentralWidget(frame)

    # Domain box: z=0 at the datum surface, z=+grid_z at the base
    # Topo surface will appear at z<0 (above the datum)
    box = pv.Box(bounds=(0.0, grid_x, 0.0, grid_y, 0.0, grid_z))
    plotter.add_mesh(box, color="lightgray", opacity=0.12, show_edges=False)
    plotter.add_mesh(box, color="gray", style="wireframe", line_width=3)

    # Topography surface read from the .topo file.
    # Topo z values are elevations (positive = above datum).
    # In the SW4/plot convention they sit at z = -elevation (above the box top).
    if topo_points is not None and len(topo_points):
        topo_z = topo_points[:, 2]
        topo_plot = np.column_stack([
            topo_points[:, 0],
            topo_points[:, 1],
            -topo_z,            # negative: above z=0 datum
        ])
        topo_cloud = pv.PolyData(topo_plot.astype(float))
        topo_cloud["Elevation [m]"] = topo_z.astype(float)
        plotter.add_points(
            topo_cloud,
            scalars="Elevation [m]",
            cmap="terrain",
            point_size=3,
            render_points_as_spheres=False,
            show_scalar_bar=True,
            scalar_bar_args={"title": "Topo elevation [m]"},
        )

    # Receivers from the .in file.
    # depth=0 lines → z=0 (at datum, top of box face).
    # z=<negative> lines → above the box (between topo and datum).
    # z=<positive> lines → inside the box (below datum).
    if len(receivers):
        rec_cloud = pv.PolyData(receivers.astype(float))
        rec_cloud["Z [m]"] = receivers[:, 2].astype(float)
        plotter.add_points(
            rec_cloud,
            scalars="Z [m]",
            cmap="coolwarm",
            point_size=4,
            render_points_as_spheres=False,
            show_scalar_bar=True,
            scalar_bar_args={"title": "Receiver Z [m]  (- above datum, + below)"},
        )

    # Sources
    if len(sources):
        source_cloud = pv.PolyData(sources.astype(float))
        plotter.add_points(source_cloud, color="red", point_size=18, render_points_as_spheres=True)

    plotter.add_text("SW4 local Cartesian", position="upper_left", font_size=11)
    plotter.show_bounds(
        grid="back",
        location="outer",
        xtitle="X [m]",
        ytitle="Y [m]",
        ztitle="Z [m]  (+ down)",
        fmt="%.1f",
    )
    plotter.show_axes()
    plotter.reset_camera()

    window.show()
    app.exec_()
