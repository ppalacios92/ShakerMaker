import numpy as np


def material_lines(crust):
    lines = []
    depth_tops = np.concatenate(([0.0], np.cumsum(crust.d[:-1])))
    for i in range(crust.nlayers - 1, -1, -1):
        vp = crust.a[i] * 1000.0
        vs = crust.b[i] * 1000.0
        rho = crust.rho[i] * 1000.0
        line = f"block vp={vp:.16g} vs={vs:.16g} rho={rho:.16g}"
        if crust.d[i] > 0:
            z1 = depth_tops[i] * 1000.0
            z2 = (depth_tops[i] + crust.d[i]) * 1000.0
            if z1 > 0:
                line += f" z1={z1:.16g}"
            line += f" z2={z2:.16g}"
        lines.append(line)
    return lines
