# `supplementary/` — auxiliary tools & datasets

This folder holds **auxiliary tools and reference datasets** used by the examples and by
research workflows around ShakerMaker.

> ⚠️ **Not part of the package.** Nothing here is compiled or installed. `setup.py` only
> packages the `shakermaker.*` subpackages, so `supplementary/` is invisible to the build and
> to `pip install`. It lives in the repository purely as a convenience so the data is in **one
> place** instead of scattered across projects.

## Contents

### `crust1/` — CRUST 1.0 global crustal model

[CRUST 1.0](https://igppweb.ucsd.edu/~gabi/crust1.html) (Laske et al., 2013) is a global
crustal model on a 1° × 1° grid (9 layers: water, ice, 3 sediments, 3 crystalline, mantle).

| Path | What |
|------|------|
| `crust1/crust1.py` | Self-contained `Crust1` reader (numpy + matplotlib only) |
| `crust1/crust1.0/`  | The CRUST 1.0 binary grids (`crust1.vp/.vs/.rho/.bnds`) + type add-on |

The reader auto-detects the data because it sits **next to** `crust1.py`:

```python
import sys, pathlib
REPO_ROOT = pathlib.Path("…/ShakerMaker")
sys.path.insert(0, str(REPO_ROOT / "supplementary" / "crust1"))

from crust1 import Crust1

crust = Crust1()                 # zero-config: finds crust1.0/ next to crust1.py
# crust = Crust1(data_dir=...)   # or point it elsewhere

profile = crust.profile_at(-33.42, -70.61)     # 9-layer column at a lat/lon
crust.print_shakermaker((-33.42, -70.61))      # ready-to-paste CrustModel snippet
```

Used by `examples/crustal_models_experiment/`.

**Source / license:** CRUST 1.0 is distributed freely by UCSD (G. Laske et al.). It is
redistributed here unmodified for convenience.
