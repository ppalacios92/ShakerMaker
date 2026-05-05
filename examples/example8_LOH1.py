
# Shakermaker
from shakermaker import shakermaker
# Crustal Model
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
# Station
from shakermaker.station import Station
from shakermaker.stationlist import StationList
# STF
from shakermaker.stf_extensions.gaussian import Gaussian

# General
import matplotlib.pylab as plt
import numpy as np

# %%
# Generate STF
sigma=0.06
t0=6*sigma
stf=Gaussian(t0=t0, freq=1/sigma, M0=1 ,derivative=False)

# %%
#Initialize CrustModel
crust = CrustModel(2)

#Slow layer
vp=4.000
vs=2.000
rho=2.600
Qa=10000.
Qb=10000.
thickness = 1.0

crust.add_layer(thickness, vp, vs, rho, Qa, Qb)

#Halfspace
vp=6.000
vs=3.464
rho=2.700
Qa=10000.
Qb=10000.
thickness = 0   #Infinite thickness!
crust.add_layer(thickness, vp, vs, rho, Qa, Qb)

# %%
# crust.plot()

# %%
M0=1e18/5e14/2
#Create source
z = 2.0                 # Source depth (km)
s,d,r = 0., 90., 0.     # Fault plane angles (deg)
source = PointSource(   [0,0,z], 
                        [s,d,r],
                        stf = Gaussian(t0=t0, freq=1/sigma, M0=M0 , derivative=False),
                    )

fault = FaultSource([source], metadata={"name":"LOH1_source"})

# %%
#Create recording station
x1,y1 = 8.0, 8.0           # Station location
s1 = Station([x1,y1,0.0], metadata={"name":"sta01", "save_gf": True,
                                    })

x2,y2 = 6.0, 8.0           # Station location
s2 = Station([x2,y2,0.0], metadata={"name":"sta02",  "save_gf": True,
                                    })

x3,y3 = 4.0, 4.0           # Station location
s3 = Station([x3,y3,0.5], metadata={"name":"sta03",  "save_gf": True,
                                    })

stations = StationList([s1, s2 , s3], {})

# %%
#Create model
model = shakermaker.ShakerMaker(crust, fault, stations)
model.run(
 dt=0.005,          # Output time-step
 nfft=2048*2,       # N timesteps
 tb=20,             # Initial zero-padding
 smth=1, 
 dk=0.05/2,         # wavenumber discretization
 verbose=True,
 )

# %%
s1.save("sta01.npz")
s2.save("sta02.npz")
s3.save("sta03.npz")