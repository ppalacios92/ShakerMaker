import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from shakermaker.core import subgreen
from shakermaker.stf_extensions.brune import Brune

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "images")

mb, src, rcv, stype, updn = 3, 3, 1, 2, 0
d = [1., 4., 0.]
a = [4., 6., 6.]
b = [2., 3.464, 3.464]
rho = [2.6, 2.7, 2.7]
qa = [54.65, 69.3, 69.3]
qb = [137.95, 120., 120.]
dt, nfft, tb, nx = 0.005, 2048, 0, 1
sigma, smth, wc1, wc2 = 2, 1, 1, 2
pmin, pmax, dk, kc, taper = 0, 1, 0.05, 15.0, 0.9
x, pf = 7.0, 0.0
df, lf = 0.7853981633974483, 1.5707963267948966
sx, sy, rx, ry = 0.0, 0.0, 0.0, 7.0

tdata, z, e, n, t0 = subgreen(mb, src, rcv, stype, updn, d, a, b, rho, qa, qb,
                              dt, nfft, tb, nx, sigma, smth, wc1, wc2, pmin, pmax,
                              dk, kc, taper, x, pf, df, lf, sx, sy, rx, ry)

gf = np.asarray(z, dtype=float)
Nt = len(gf)
t = np.arange(Nt) * dt + t0[0]

stf = Brune(f0=2.0, t0=0.0)
stf.dt = dt
s = np.asarray(stf.data, dtype=float)
ts = np.arange(len(s)) * dt

conv = fftconvolve(gf, s)[:Nt] * dt

fig, axs = plt.subplots(3, 1, figsize=(8, 7))
axs[0].plot(t, gf, color="#1f3a4d", linewidth=1.2)
axs[0].set_title("Green's function  G(t)  — impulse response, vertical")
axs[1].plot(ts, s, color="#c0392b", linewidth=1.5)
axs[1].set_title("Source time function  s(t)  — Brune, f0 = 2 Hz")
axs[2].plot(t, conv, color="#0b2540", linewidth=1.2)
axs[2].set_title("Ground motion  u(t) = G(t) * s(t)")
for ax in axs:
    ax.set_xlim(t[0], t[0] + 8)
    ax.set_xlabel("t (s)")
fig.tight_layout()
fig.savefig(os.path.join(IMG, "convolution.png"), dpi=150, bbox_inches="tight")
plt.close("all")

print("convolution OK")
