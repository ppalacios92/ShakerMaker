from pathlib import Path

import numpy as np

_DEPTH_ENCODE_OFFSET = 1e9   # depth=d encoded as z = -(d + 1e9); decoded in _resolve_receivers


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
    """Parse one source or rec line into (x, y, z).

    Convention:
      z > 0  → below datum (depth, inside domain box)
      z < 0  → above datum (elevation, topo zone)
      depth= lines encoded as z = -(depth + _DEPTH_ENCODE_OFFSET); resolved later.
    """
    x = _token_value(tokens, "x")
    y = _token_value(tokens, "y")
    if x is None or y is None:
        return None
    depth = _token_value(tokens, "depth")
    if depth is not None:
        # Encode depth in z: -(depth + offset) — always negative and large enough
        # to distinguish from real z values; resolved later from the topo file.
        return x, y, -(depth + _DEPTH_ENCODE_OFFSET)
    z = _token_value(tokens, "z")
    if z is not None:
        return x, y, z
    return x, y, 0.0


def read_sw4_geometry(path):
    """Parse a SW4 .in file.

    Returns
    -------
    grid : (gx, gy, gz)
    sources : ndarray (N,3)
    receivers : ndarray (N,3) — z=NaN means depth= (topo surface, needs resolution)
    topo_points : ndarray (M,3) or None — raw topo file, z = positive elevation
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
            pt = _read_point(tokens)
            if pt is not None:
                sources.append(pt)
        elif tokens[0] == "rec":
            pt = _read_point(tokens)
            if pt is not None:
                receivers.append(pt)
        elif tokens[0] == "topography":
            topo_rel_file = _token_str(tokens, "file")

    if grid is None:
        raise ValueError(f"No grid line found in {path}.")

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


def _resolve_receivers(receivers, topo_points):
    """Resolve depth=-encoded z values to their actual SW4 z position.

    Encoding: z = -(depth + _DEPTH_ENCODE_OFFSET)
      depth=0  → station at the free surface        → z = -topo_z
      depth=d  → station d metres below free surface → z = -(topo_z - d)

    Topo file stores positive elevation; SW4 z-axis has z < 0 above datum.
    Falls back to z=0 when no topo match is found.
    """
    if len(receivers) == 0:
        return receivers.copy()

    resolved = receivers.copy()
    depth_mask = resolved[:, 2] < -(_DEPTH_ENCODE_OFFSET / 2.0)
    if not depth_mask.any():
        return resolved

    topo_lookup = {}
    if topo_points is not None and len(topo_points):
        for tx, ty, tz in topo_points:
            topo_lookup[(round(float(tx)), round(float(ty)))] = float(tz)

    for i in np.where(depth_mask)[0]:
        x, y = resolved[i, 0], resolved[i, 1]
        depth = -(resolved[i, 2]) - _DEPTH_ENCODE_OFFSET   # recover depth from encoding
        tz = topo_lookup.get((round(x), round(y)), 0.0)
        # SW4 convention: positive topo elevation tz → SW4 z = -tz at surface
        # depth d below surface → SW4 z = -(tz - d)
        resolved[i, 2] = -(tz - depth)

    return resolved


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

    # Resolve depth=0 receivers to their actual topo elevation
    receivers = _resolve_receivers(receivers, topo_points)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    window = QtWidgets.QMainWindow()
    window.setWindowTitle(f"SW4 geometry — {path.name}")
    window.resize(1200, 800)

    frame = QtWidgets.QFrame()
    layout = QtWidgets.QVBoxLayout(frame)
    plotter = QtInteractor(frame)
    layout.addWidget(plotter.interactor)
    window.setCentralWidget(frame)

    # Domain box: z=0 at datum, z=+grid_z at bottom
    box = pv.Box(bounds=(0.0, grid_x, 0.0, grid_y, 0.0, grid_z))
    plotter.add_mesh(box, color="lightgray", opacity=0.12, show_edges=False)
    plotter.add_mesh(box, color="gray", style="wireframe", line_width=2)

    # Topo surface as background (no colorbar) — plotted at z=-tz (above datum)
    if topo_points is not None and len(topo_points):
        topo_plot = np.column_stack([
            topo_points[:, 0],
            topo_points[:, 1],
            -topo_points[:, 2],
        ]).astype(float)
        plotter.add_points(
            pv.PolyData(topo_plot),
            color="tan",
            point_size=2,
            render_points_as_spheres=False,
            opacity=0.4,
        )

    # All receivers — ONE colorbar showing Z
    # z < 0 : above datum (topo surface stations, z0 stations)
    # z = 0 : at datum
    # z > 0 : below datum (domain grid, shakermaker subsurface)
    if len(receivers):
        rec_cloud = pv.PolyData(receivers.astype(float))
        rec_cloud["Z [m]"] = receivers[:, 2].astype(float)
        plotter.add_points(
            rec_cloud,
            scalars="Z [m]",
            cmap="RdYlBu_r",
            point_size=4,
            render_points_as_spheres=False,
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "Z [m]   (- above datum / + below)",
                "vertical": True,
            },
        )

    # Sources — red spheres
    if len(sources):
        plotter.add_points(
            pv.PolyData(sources.astype(float)),
            color="red",
            point_size=18,
            render_points_as_spheres=True,
        )

    plotter.add_text("SW4 local Cartesian", position="upper_left", font_size=11)
    plotter.show_bounds(
        grid="back",
        location="outer",
        xtitle="X [m]",
        ytitle="Y [m]",
        ztitle="Z [m]  (+ down)",
        fmt="%.0f",
    )
    plotter.show_axes()
    plotter.reset_camera()

    window.show()
    app.exec_()
