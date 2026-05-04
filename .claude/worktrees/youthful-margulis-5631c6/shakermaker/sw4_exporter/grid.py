def domain_from_topography(local_points, h, z_domain):
    x_domain = float(local_points[:, 0].max() - local_points[:, 0].min())
    y_domain = float(local_points[:, 1].max() - local_points[:, 1].min())
    return x_domain, y_domain, z_domain


def grid_line(h, x_domain, y_domain, z_domain):
    return f"grid h={h:.16g} x={x_domain:.16g} y={y_domain:.16g} z={z_domain:.16g}"
