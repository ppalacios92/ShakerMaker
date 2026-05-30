# Outputs & writers

Where the computed motion goes: in memory, to NumPy archives, to aggregate
HDF5, or to H5DRM for external solvers.

## In-memory results

After `run()`, each `Station` holds the three velocity components and the time
vector:

```python
z, e, n, t = sta.get_response()      # NumPy arrays, length nfft (× smth)
```

The natural FK output is **velocity** (cm/s). Integrate for displacement,
differentiate for acceleration, built into [plotting](plotting.md).

## Per-station NumPy archive (`.npz`)

The simplest persistent format, one file per station:

```python
sta.save("STA01.npz")
sta2 = Station(); sta2.load("STA01.npz")
```

Good for one-off explorations and unit tests.

## Aggregate HDF5 / H5DRM (`slw_extensions`)

For many stations, pass a `writer=` to `run()` so results stream to a single
file as they are computed (O(1) memory):

| Writer | Format | Use case |
|---|---|---|
| `DRMHDF5StationListWriter` | H5DRM | DRM boundary motions for OpenSees / SW4 |
| `HDF5StationListWriter` | HDF5 | generic station-list archive |

```python
from shakermaker.slw_extensions import DRMHDF5StationListWriter

writer = DRMHDF5StationListWriter("motions.h5drm")
model.run(..., writer=writer)
```

### The `StationListWriter` interface

All writers share one contract (subclass it for a custom format):

| Member | Purpose |
|---|---|
| `initialize(station_list, num_samples)` | open + allocate |
| `write_station(station, index)` | write one station |
| `write_metadata(metadata)` | write run metadata |
| `write(station_list, num_samples)` | write the whole list |
| `close()` | flush + close |

Properties: `filename`, `transform_function` (an optional coordinate
transform applied on write).

## SW4 export

ShakerMaker can hand sources and motions to the SW4 finite-difference solver:

| Method | Writes |
|---|---|
| `model.export_sw4(path=...)` | sources / motions for SW4 |
| `model.export_sw4_topo(path=...)` | the same, with topography |

The reverse path, building an `.h5drm` *from* an SW4 case, is
`other_utils/build_h5drm_from_sw4_case.py`.

## H5DRM and OpenSees

The H5DRM layout (coordinates, time, velocity/displacement/acceleration, and a
`DRM_Information` group) is read directly by the OpenSees `H5DRMLoadPattern`.
See the [DRM guide](drm.md) for the full structure.

## Reference

[Plotting & visualisation](plotting.md) · [ShakerMaker engine API](../api/shakermaker.md)
