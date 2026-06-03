"""SW4 ``block`` (material) lines from a ShakerMaker crust.

The crust is given to ShakerMaker as layers of (thickness, vp, vs, rho).
SW4 wants the same information expressed as semi-infinite slabs with
absolute depth caps (``z1``, ``z2``). This module converts one to the other,
in SI units and from bottom layer up so the SW4 reader applies them
in the right order.

Optionally, the converter appends one thin ``block`` per internal interface
holding effective-medium properties (harmonic average of the Lame parameters
mu/lambda, arithmetic average of density). On a node-based finite-difference
grid the node sitting exactly on a material discontinuity must carry a single
value; the effective medium centres the interface instead of staircasing it,
which markedly improves accuracy at no extra cost. See the SW4 User's Guide
(LOH.1 test case) for the recipe.
"""

import warnings

import numpy as np

# Relative tolerance used to decide whether an interface depth coincides with
# a grid node (z_k / h must be (close to) an integer).
_NODE_TOL = 1.0e-6


def deepest_interface(crust):
    """Depth (m) of the deepest internal interface, i.e. the top of the
    bottom (half-space) layer.

    The exporter uses this both to size the SW4 box in z (the domain must
    reach below this interface so the half-space material is present) and to
    place the deepest harmonic interface node. Keeping the value in one place
    stops the material and domain logic from disagreeing.

    Inputs
    ------
    crust : CrustModel
        Must expose ``d`` (layer thicknesses, km) and ``nlayers``.

    Returns
    -------
    float
        Depth in metres. ``0.0`` for a single-layer (pure half-space) crust.
    """
    if crust.nlayers < 2:
        return 0.0
    return float(np.sum(crust.d[:-1])) * 1000.0


def material_lines(crust, h=None, interface_blocks=False, interface_block_delta=1.0):
    """Convert a ShakerMaker crust into SW4 ``block`` lines.

    The crust is iterated bottom-up so the SW4 input lists the deepest layer
    first. The bottommost layer is left without ``z1``/``z2`` so SW4 treats
    it as a halfspace.

    When ``interface_blocks`` is set, one extra thin ``block`` per internal
    interface is appended *after* the layer blocks (SW4 applies materials in
    order, last one wins, so the interface node correctly overrides the single
    grid plane at the discontinuity).

    Inputs
    ------
    crust : CrustModel
        Must expose ``d`` (layer thicknesses, km), ``a`` (Vp, km/s),
        ``b`` (Vs, km/s), ``rho`` (g/cm^3) and ``nlayers``.
    h : float, optional
        Grid spacing in metres. Required when ``interface_blocks`` is set so
        the interface depth can be checked against the grid nodes.
    interface_blocks : bool
        When ``True``, append effective-medium interface nodes. Default
        ``False``.
    interface_block_delta : float
        Half-thickness (m) of each interface block: ``z1 = z_k - delta``,
        ``z2 = z_k + delta``. Must be ``< h/2`` so the block captures exactly
        the one grid plane on the interface. Default ``1.0``.

    Returns
    -------
    list of str
        One ``block vp=... vs=... rho=... [z1=... z2=...]`` line per layer,
        optionally followed by one interface ``block`` per internal interface.
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

    if interface_blocks:
        lines += _interface_block_lines(crust, depth_tops, h, interface_block_delta)
    return lines


def _interface_block_lines(crust, depth_tops, h, delta):
    """Effective-medium ``block`` lines, one per internal interface.

    Harmonic average for the Lame parameters (mu, lambda), arithmetic average
    for density. Interfaces that do not coincide with a grid node are skipped
    with a warning (the node trick only works when ``z_k`` is a multiple of
    ``h``).

    Inputs
    ------
    crust : CrustModel
    depth_tops : ndarray
        Per-layer top depths in km (``[0, d0, d0+d1, ...]``).
    h : float or None
        Grid spacing in metres.
    delta : float
        Interface block half-thickness in metres.

    Returns
    -------
    list of str
    """
    if h is None:
        warnings.warn(
            "interface_blocks requested but grid spacing h is unknown; "
            "skipping harmonic interface nodes.",
            stacklevel=3,
        )
        return []

    h = float(h)
    delta = float(delta)
    if delta >= 0.5 * h:
        warnings.warn(
            f"interface_block_delta={delta:g} is not < h/2 ({0.5 * h:g}); the "
            "interface block may capture neighbouring grid planes. Use a "
            "smaller delta.",
            stacklevel=3,
        )

    out = []
    for i in range(1, crust.nlayers):
        # Interface between layer i-1 (above) and layer i (below).
        z_k = float(depth_tops[i]) * 1000.0

        ratio = z_k / h
        if abs(ratio - round(ratio)) > _NODE_TOL:
            warnings.warn(
                f"Interface at z={z_k:g} m does not fall on a grid node "
                f"(h={h:g}); skipping its harmonic node. Choose an h that "
                f"divides {z_k:g}.",
                stacklevel=3,
            )
            continue

        vp_t, vs_t, rho_t = crust.a[i - 1] * 1000.0, crust.b[i - 1] * 1000.0, crust.rho[i - 1] * 1000.0
        vp_b, vs_b, rho_b = crust.a[i] * 1000.0, crust.b[i] * 1000.0, crust.rho[i] * 1000.0

        mu_t = rho_t * vs_t ** 2
        mu_b = rho_b * vs_b ** 2
        lam_t = rho_t * (vp_t ** 2 - 2.0 * vs_t ** 2)
        lam_b = rho_b * (vp_b ** 2 - 2.0 * vs_b ** 2)
        if min(mu_t, mu_b, lam_t, lam_b) <= 0.0:
            warnings.warn(
                f"Interface at z={z_k:g} m has non-positive Lame parameters; "
                "skipping harmonic node.",
                stacklevel=3,
            )
            continue

        mu_i = 2.0 / (1.0 / mu_t + 1.0 / mu_b)
        lam_i = 2.0 / (1.0 / lam_t + 1.0 / lam_b)
        rho_i = 0.5 * (rho_t + rho_b)
        vs_i = np.sqrt(mu_i / rho_i)
        vp_i = np.sqrt((lam_i + 2.0 * mu_i) / rho_i)

        z1 = z_k - delta
        z2 = z_k + delta
        out.append(
            f"block vp={vp_i:.16g} vs={vs_i:.16g} rho={rho_i:.16g} "
            f"z1={z1:.16g} z2={z2:.16g}"
        )
    return out
