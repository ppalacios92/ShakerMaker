# Finite faults & FFSP

From a single point to an extended rupture, and from one deterministic
rupture to a stochastic ensemble.

## Point source → finite fault

A real earthquake is not a point: it is slip over an extended fault plane.
The finite-fault response is the **linear superposition** of many point
sources (subfaults), each with its own location, mechanism, slip amount, and
**rupture time**, the instant the rupture front reaches it:

$$
u_n(\mathbf{x}_r, t) \;=\; \sum_{\alpha=1}^{N_\text{sub}}
\; G_{ni}(\mathbf{x}_r,\, t - t_{0,\alpha};\, \mathbf{\xi}_\alpha)\,
* \, \dot{M}_{ij,\alpha}(t)\,\hat n_j
$$

Each subfault is a [`PointSource`](../guides/sources.md); the collection is a
[`FaultSource`](../guides/sources.md). The rupture-time delay $t_{0,\alpha}$
is the `tt` argument on each `PointSource`, and the moment release shape is
its [source time function](../guides/source_time_functions.md). Get the
rupture timing and the slip distribution right and the superposition does the
rest.

The resolution rule: subfaults must be small versus the shortest wavelength,
$\Delta\xi,\Delta\eta \lesssim V_S^\text{min} / (N_p\,f_\text{max})$ with
$N_p \approx 5$ subfaults per wavelength.

## Why stochastic? (FFSP)

The slip distribution of a *future* earthquake is unknowable, even the
best-recorded past events admit many slip inversions that fit the data
equally well (~30% irreducible uncertainty in local slip). The honest
forward approach is therefore an **ensemble**: generate many physically
admissible ruptures, compute each, and report a *distribution* of ground
motions rather than a single number.

The **Finite Fault Stochastic Process (FFSP)** tool does exactly this. It
splits the rupture into:

- **Constrained (deterministic):** moment / magnitude, fault dimensions
  $L\times W$, orientation (strike, dip, rake), hypocentre, target corner
  frequencies.
- **Random fields:** slip, rise time, peak time, rupture velocity, and
  dip/rake perturbations, drawn with magnitude-scaled correlation lengths
  and prescribed cross-correlations.

Each realisation is scored against the targets (a PDF score); you keep either
the single **best** realisation (deterministic analysis) or the **full
ensemble** (probabilistic analysis).

In ShakerMaker this is the [`FFSPSource`](../guides/ffsp.md) class, every
constrained quantity and every random-field control is a constructor
argument.

## The efficiency payoff

For an FFSP ensemble over a fixed geometry, the **FK Green's functions are
shared across all realisations**, only the slip and rupture times change.
ShakerMaker computes the Green's functions once and re-runs only the cheap
recombination per realisation, so the amortised cost of an extra realisation
is small.

## References

- Aki, K. & Richards, P. G. (2002). *Quantitative Seismology*, §§4.3, 10.
- Liu, P., Archuleta, R. J. & Hartzell, S. (2006). *BSSA* **96**, 2118–2130.
- Graves, R. & Pitarka, A. (2010, 2016). *BSSA*.
- Atkinson, G. (1993), double-corner-frequency source spectrum.
