# Exercises

A graded path through ShakerMaker. Each exercise is a **complete, runnable
script** plus the result you should see and how to read it. They build on
each other, do them in order the first time.

| # | Exercise | You learn | New inputs |
|---|---|---|---|
| 1 | [First run & the four arrivals](01_first_run.md) | the full pipeline; reading a seismogram | `CrustModel`, `PointSource`, `Station`, `run` |
| 2 | [Numerical convergence](02_convergence.md) | what `dk` and `nfft` actually control | the numerical parameters |
| 3 | [A sedimentary basin](03_basin.md) | how layering shapes motion; resonance | multi-layer `CrustModel`, `Brune` |
| 4 | [Source time functions compared](04_stf.md) | choosing an STF for a target band | `Dirac`, `Brune`, `Gaussian`, `Discrete` |
| 5 | [DRM box → H5DRM](05_drm.md) | boundary motions for OpenSees | `DRMBox`, `DRMHDF5StationListWriter` |
| 6 | [FFSP stochastic rupture](06_ffsp.md) | an ensemble of admissible ruptures | `FFSPSource` |

!!! tip "How to run"
    Each script is self-contained, copy it into a `.py` file and run with
    `python exercise.py`. On a workstation the FK examples finish in seconds;
    the FFSP and DRM ones take longer. For large jobs, launch under MPI:
    `mpirun -n 8 python exercise.py`.
