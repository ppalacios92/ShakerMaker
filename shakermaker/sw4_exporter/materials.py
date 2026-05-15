"""SW4 ``block`` (material) lines from a ShakerMaker crust.

The crust is given to ShakerMaker as layers of (thickness, vp, vs, rho).
SW4 wants the same information expressed as semi-infinite slabs with
absolute depth caps (``z1``, ``z2``). This module converts one to the other,
in SI units and from bottom layer up so the SW4 reader applies them
in the right order.
"""

import numpy as np


def material_lines(crust):
    """Convert a ShakerMaker crust into SW4 ``block`` lines.

    The crust is iterated bottom-up so the SW4 input lists the deepest layer
    first. The bottommost layer is left without ``z1``/``z2`` so SW4 treats
    it as a halfspace.

    Inputs
    ------
    crust : CrustModel
        Must expose ``d`` (layer thicknesses, km), ``a`` (Vp, km/s),
        ``b`` (Vs, km/s), ``rho`` (g/cm^3) and ``nlayers``.

    Returns
    -------
    list of str
        One ``block vp=... vs=... rho=... [z1=... z2=...]`` line per layer.
        Values are in SI units (m/s and kg/m^3) as SW4 expects.
    """
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
