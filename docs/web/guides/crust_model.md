# Crust model

The medium: a 1-D layered, anelastic half-space. Build it layer by layer,
top to bottom. The last layer (zero thickness) is the half-space.

## Input: `add_layer`

```python
from shakermaker.crustmodel import CrustModel

crust = CrustModel(2)                                  # number of layers
crust.add_layer(1.0, 4.0, 2.0, 2.6, 10000., 10000.)   # soft surface layer
crust.add_layer(0.0, 6.0, 3.464, 2.7, 10000., 10000.) # half-space (d = 0)
```

`add_layer(d, vp, vs, rho, qp, qs)`, positional, in this exact order:

| Arg | Symbol | Units | Meaning | Typical |
|---|---|---|---|---|
| `d` | $d$ | km | layer thickness (**0 = half-space**) | 0.1 – 10 |
| `vp` | $V_P$ | km/s | P-wave velocity | 1.5 – 8 |
| `vs` | $V_S$ | km/s | S-wave velocity (must be `< vp`) | 0.5 – 4.5 |
| `rho` | $\rho$ | g/cm³ | density | 1.8 – 3.3 |
| `qp` | $Q_P$ | – | P quality factor (↑ = less damping) | 50 – 10000 |
| `qs` | $Q_S$ | – | S quality factor | 50 – 10000 |

`add_layer` validates the physics (positive speeds/density/Q, $V_P > V_S$).
Use a large `Q` (e.g. `10000`) for a near-elastic medium.

## Input: pre-packaged models (`shakermaker.cm_library`)

Skip the hand-building with a benchmark model:

```python
from shakermaker.cm_library.LOH import SCEC_LOH_1
crust = SCEC_LOH_1()        # 1 km soft layer over a half-space
```

| Constructor | Model |
|---|---|
| `SCEC_LOH_1()` | SCEC LOH.1 benchmark |
| `AbellThesis()` | regional Chilean subduction crust |

`plot_profile()` draws the velocity and density structure versus depth:

![SCEC LOH.1 profile](../assets/images/crust_loh1.png){ width=460 }

## Result: inspect the model

```python
crust.plot_profile()        # velocity / density vs depth
```

Quick queries (no plotting): `crust.nlayers`, and the per-layer arrays
`crust.d`, `crust.a` (Vp), `crust.b` (Vs), `crust.rho`, `crust.qa`, `crust.qb`.

| Method | Returns |
|---|---|
| `properties_at_depths(z)` | $(V_P, V_S, \rho, Q)$ sampled at depth(s) `z` |
| `get_layer(z)` | index of the layer containing depth `z` |
| `split_at_depth(z)` | inserts an interface at depth `z` |
| `modify_layer(i, vp=…, …)` | edits layer `i` in place |
| `plot()` / `plot_profile()` | layer plot / full profile |

## Reference

[`CrustModel` API →](../api/crustmodel.md)
