import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shakermaker.core import subgreen

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

Nt = len(z)
t = np.arange(Nt) * dt + t0[0]
labels = ["ZDD", "RDD", "ZDS", "RDS", "TDS", "ZSS", "RSS", "TSS", "EX"]

fig = plt.figure(figsize=(10, 8))
for i in range(9):
    ax = fig.add_subplot(3, 3, 1 + i)
    ax.plot(t, tdata[0, i, :], color="#c0392b", linewidth=1.0)
    ax.set_title(labels[i], fontsize=9)
    ax.set_xlim(t[0], t[0] + 8)
    ax.tick_params(labelsize=7)
fig.suptitle("Nine elementary Green's functions (subgreen), r = 7 km",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(IMG, "green_functions_9.png"), dpi=150, bbox_inches="tight")
plt.close("all")

print("green functions OK")
