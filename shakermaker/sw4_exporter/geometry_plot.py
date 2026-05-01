from pathlib import Path

import numpy as np


def _token_value(tokens, key):
    prefix = f"{key}="
    for token in tokens:
        if token.startswith(prefix):
            return float(token[len(prefix):])
    return None


def _read_grid(tokens):
    x = _token_value(tokens, "x")
    y = _token_value(tokens, "y")
    z = _token_value(tokens, "z")
    if None in (x, y, z):
        raise ValueError("SW4 grid line must include x, y, and z.")
    return x, y, z


def _read_point(tokens, depth_sign):
    x = _token_value(tokens, "x")
    y = _token_value(tokens, "y")
    z = _token_value(tokens, "z")
    depth = _token_value(tokens, "depth")
    if x is None or y is None:
        return None
    if depth is not None:
        return x, y, depth_sign * depth
    if z is not None:
        return x, y, depth_sign * z
    return x, y, 0.0


def read_sw4_geometry(path):
    path = Path(path)
    grid = None
    sources = []
    receivers = []

    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        tokens = line.split()
        if tokens[0] == "grid":
            grid = _read_grid(tokens)
        elif tokens[0] == "source":
            point = _read_point(tokens, depth_sign=-1.0)
            if point is not None:
                sources.append(point)
        elif tokens[0] == "rec":
            point = _read_point(tokens, depth_sign=1.0)
            if point is not None:
                receivers.append(point)

    if grid is None:
        raise ValueError(f"No grid line found in {path}.")

    return grid, np.asarray(sources, dtype=float), np.asarray(receivers, dtype=float)


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
    (grid_x, grid_y, grid_z), sources, receivers = read_sw4_geometry(path)

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

    box = pv.Box(bounds=(0.0, grid_x, 0.0, grid_y, -grid_z, 0.0))
    plotter.add_mesh(box, color="lightgray", opacity=0.12, show_edges=False)
    plotter.add_mesh(box, color="gray", style="wireframe", line_width=3)

    if len(receivers):
        receiver_cloud = pv.PolyData(receivers)
        receiver_cloud["Elevation"] = receivers[:, 2]
        plotter.add_points(
            receiver_cloud,
            scalars="Elevation",
            cmap="terrain",
            point_size=4,
            render_points_as_spheres=False,
            show_scalar_bar=True,
            scalar_bar_args={"title": "Receiver elevation [m]"},
        )

    if len(sources):
        source_cloud = pv.PolyData(sources)
        plotter.add_points(source_cloud, color="red", point_size=18, render_points_as_spheres=True)

    plotter.add_text("SW4 local Cartesian", position="upper_left", font_size=11)
    plotter.show_bounds(
        grid="back",
        location="outer",
        xtitle="X [m]",
        ytitle="Y [m]",
        ztitle="Z [m]",
        fmt="%.1f",
    )
    plotter.show_axes()
    plotter.reset_camera()

    window.show()
    app.exec_()
