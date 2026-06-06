# Running a simulation

`ShakerMaker(crust, source, receivers)` pairs every source with every
receiver and computes the motion. Two ways to run it: a single `run()` call,
or the three-stage MPI pipeline.

## Input: `run()`

```python
from shakermaker.shakermaker import ShakerMaker

model = ShakerMaker(crust, fault, stations)
model.run(dt=0.005, nfft=2048, dk=0.1, tb=500)
```

### The numerical parameters

These come straight from the FK sampling theory, each one has a physical
meaning, not just a knob. Defaults cover the standard case ($f_\text{max}
\approx 10$ Hz, $r \le 50$ km). For the full derivation of every parameter,
the source-line citations, and the `check_parameters` pre-flight, see the
[**FK parameters reference**](parameters.md).

| Param | Default | Units | Meaning | Turn it ↑ when… |
|---|---|---|---|---|
| `dt` | 0.05 | s | output time step | you need higher frequencies (↓ it) |
| `nfft` | 4096 | – | FFT length → record length $T = N\cdot dt$ | the wave train is cut off (wrap-around) |
| `tb` | 1000 | samples | pre-arrival zero padding | the first arrival is clipped |
| `dk` | 0.3 | π/r | wavenumber spacing | a long oscillatory tail appears (↓ it) |
| `kc` | 15.0 | 1/h | wavenumber cutoff | high-frequency S is missing |
| `sigma` | 2 | 1/T | trace damping $e^{-\sigma t}$ (anti wrap-around) | tail looks noisy (↓ it) |
| `smth` | 1 | – | output densification factor | you want finer sampling |
| `taper` | 0.9 | – | spectral taper (0–1) | high-frequency content is lost |
| `pmin`,`pmax` | 0, 1 | 1/Vs | slowness window | rarely changed |
| `tmin`,`tmax` | 0, 100 | s | output time window |  |
| `writer` | None | – | [`StationListWriter`](outputs.md) for direct HDF5 | writing many stations |

## Result: where the motion goes

After `run()`, each `Station` holds three velocity components and a time
vector, read with `s.get_response()` (see [Receivers](receivers.md)) or pass
a `writer=` to stream to disk.

## The OP pipeline (`run_nearest`)

`run()` computes every (source, receiver) pair independently, fine for a
handful of stations, but a DRM box or surface grid has thousands of receivers
and a finite fault has thousands of subfaults, so the naïve product explodes.

The key observation: the FK kernel depends only on the **geometry** of a pair
,  the horizontal distance, the source depth, and the receiver depth. Two pairs
with (nearly) the same geometry produce the **same** Green's functions; only
the azimuth and the mechanism differ afterwards. So instead of computing every
pair, you compute every *unique geometry* once and reuse it. That is the
"optimised" (OP) pipeline, exposed as `run_nearest`.

### The three stages

`run_nearest(stage='all', ...)` runs them in sequence; you can also call them
individually (`stage=0/1/2`) to extend a run incrementally.

| Stage | What it does | Writes |
|---|---|---|
| **0**, pair clustering | groups all pairs into *unique-geometry slots* | `<db>_map.h5` |
| **1**, Green's functions | runs the FK kernel **once per slot** | `<db>_gf.h5` |
| **2**, assembly | per pair: recombine mechanism + azimuth, convolve the STF, accumulate | the `writer` output |

You pass a single database root via `h5_database_name`; the pipeline appends
`_map.h5` and `_gf.h5` itself.

### Stage-0 clustering tolerances

These control *how close* two pairs must be to share a slot, the central
trade-off of the whole method:

| Argument | Groups by | Effect |
|---|---|---|
| `delta_h` | horizontal distance | larger → fewer slots, more reuse, faster |
| `delta_v_rec` | receiver depth | how finely receiver depths are distinguished |
| `delta_v_src` | source depth | how finely source depths are distinguished |
| `npairs_max` |  | max pairs processed per batch (memory cap) |

Smaller tolerances mean more, more-exact slots (higher fidelity, more kernel
calls); larger tolerances collapse near-identical geometries together (fewer
kernel calls, faster). For a flat surface grid at a single depth, even a small
`delta_h` already collapses thousands of receivers into a handful of distance
rings.

### Calling it

```python
model = ShakerMaker(crust, fault, stations)

model.run_nearest(
    stage='all',
    h5_database_name='./run/gf_database.h5',   # writes _map.h5 and _gf.h5
    # Stage 0: clustering
    delta_h=2.5e-3, delta_v_rec=2.5e-3, delta_v_src=2.5e-3, npairs_max=100000,
    # Core numerical parameters (as in run())
    dt=0.005, nfft=4096, dk=0.05, tb=20, smth=1,
    # Stage 2: output
    writer=writer, writer_mode='progressive',
    showProgress=True,
)
```

```bash
mpirun -n 64 python my_simulation.py    # one rank per CPU core
```

`writer_mode='progressive'` flushes each station to disk and frees its memory
immediately, so even a huge DRM box runs in O(1) RAM. The full workflow is
walked through in
[Exercise 7](../exercises/07_receivers_pipeline.md).

MPI helpers: `enable_mpi(rank, nprocs)`, `mpi_is_master_process()`,
`mpi_rank()`, `mpi_nprocs()`.

## Diagnostic checklist

When the seismogram disagrees with expectation, the symptom points to the
parameter:

| Symptom | Cause | Fix |
|---|---|---|
| Long oscillatory tail from $t\sim r/V_S$ | spatial aliasing, `dk` too large | halve `dk` |
| Tail looks like noise | `sigma` too large, signal decayed | reduce `sigma` |
| Wave train ends abruptly at $t\sim T$ | wrap-around, record too short | increase `nfft` (or `sigma`) |
| High-frequency content missing | taper too aggressive | raise `taper` cutoff |
| Wrong arrival times | wrong crust or source depth | check `Vp, Vs, z` |
| Wrong polarity on Z/R/T | strike/dip/rake convention | re-check `angles` |
| `NaN` at a receiver | receiver exactly at source ($r=0$) | move it slightly |
| Saturation at high freq | `kc` too small | increase `kc` |

## Reference

[ShakerMaker engine API →](../api/shakermaker.md)
