"""
example_STF
===========
Plot ShakerMaker Source Time Functions (STF).
"""

import numpy as np
import matplotlib.pyplot as plt

from shakermaker.stf_extensions.dirac import Dirac
from shakermaker.stf_extensions.discrete import Discrete
from shakermaker.stf_extensions.brune import Brune
from shakermaker.stf_extensions.srf2 import SRF2
from shakermaker.stf_extensions.gaussian import Gaussian


plt.style.use("ggplot")

DT = 0.001

SAVE_DIR = "../docs/source/images"

# ---------------------------------------------------------------
# Brune (Original + Smoothed)
# ---------------------------------------------------------------
f0, t0 = 10.0, 0.5

stf = Brune(f0=f0, t0=t0, slip=1.0, smoothed=False)
stf.dt = DT
stf_s = Brune(f0=f0, t0=t0, slip=1.0, smoothed=True)
stf_s.dt = DT

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.plot(stf.t, stf.data, color="#E74C3C", linewidth=1.5, label="Original")
ax.plot(stf_s.t, stf_s.data, color="#3498DB", linewidth=1.5, label="Smoothed")
ax.set_title(f"Brune sources. f0={f0} (Hz) t0={t0} (s)")
ax.set_ylabel("STF")
ax.legend()
fig.tight_layout()
fig.savefig(f"{SAVE_DIR}/stf_brune.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------
stf = Gaussian(t0=0.1, freq=60.0, M0=1.0, derivative=False)
stf.dt = DT

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.plot(stf.t, stf.data, color="#E74C3C", linewidth=1.5)
ax.set_xlim(0, 0.3)
ax.set_title("Guassian source. f=10.0 (Hz)")
ax.set_ylabel("STF")
fig.tight_layout()
fig.savefig(f"{SAVE_DIR}/stf_gaussian.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------
# SRF2 (SRF2-like)
# ---------------------------------------------------------------
stf = SRF2(Tr=2.0, Tp=0.1, Te=1.5, dt=DT, slip=12.0, a=1.0, b=1.0)
stf.dt = DT

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.plot(stf.t, stf.data, color="#E74C3C", linewidth=1.5)
ax.set_title("SRF2")
ax.set_ylabel("STF")
fig.tight_layout()
fig.savefig(f"{SAVE_DIR}/SRF2.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------
# Dirac
# ---------------------------------------------------------------
stf = Dirac()
stf.dt = DT

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.stem(stf.t, stf.data, linefmt="#E74C3C", markerfmt="o", basefmt=" ")
ax.set_title("Dirac")
ax.set_ylabel("STF")
fig.tight_layout()
fig.savefig(f"{SAVE_DIR}/stf_dirac.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------
# Discrete
# ---------------------------------------------------------------
t_user = np.linspace(0, 0.3, 301)
data_user = np.exp(-((t_user - 0.1) / 0.02) ** 2)
stf = Discrete(data_user, t_user)
stf.dt = DT

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.plot(stf.t, stf.data, color="#E74C3C", linewidth=1.5)
ax.set_title("Discrete")
ax.set_ylabel("STF")
fig.tight_layout()
fig.savefig(f"{SAVE_DIR}/stf_discrete.png", dpi=150)
plt.close(fig)

print("All STF plots saved.")
