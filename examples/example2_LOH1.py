from shakermaker import shakermaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions.gaussian import Gaussian

import matplotlib.pylab as plt
import numpy as np

sigma=0.06
t0=6*sigma
stf=Gaussian(t0=t0, freq=1/sigma, M0=1 ,derivative=False)

crust = CrustModel(2)

vp=4.000
vs=2.000
rho=2.600
Qa=10000.
Qb=10000.
thickness = 1.0
crust.add_layer(thickness, vp, vs, rho, Qa, Qb)

vp=6.000
vs=3.464
rho=2.700
Qa=10000.
Qb=10000.
thickness = 0
crust.add_layer(thickness, vp, vs, rho, Qa, Qb)

M0=1e18/5e14/2
z = 2.0
s,d,r = 0., 90., 0.
source = PointSource(   [0,0,z], 
                        [s,d,r],
                        stf = Gaussian(t0=t0, freq=1/sigma, M0=M0 , derivative=False),
                    )
fault = FaultSource([source], metadata={"name":"LOH1_source"})

x1,y1 = 8.0, 8.0
s1 = Station([x1,y1,0.0], metadata={"name":"sta01", "save_gf": True})
x2,y2 = 6.0, 8.0
s2 = Station([x2,y2,0.0], metadata={"name":"sta02",  "save_gf": True})
x3,y3 = 4.0, 4.0
s3 = Station([x3,y3,0.5], metadata={"name":"sta03",  "save_gf": True})
stations = StationList([s1, s2 , s3], {})

model = shakermaker.ShakerMaker(crust, fault, stations)
model.run(
 dt=0.005,
 nfft=2048*2,
 tb=20,
 smth=1, 
 dk=0.05/2,
 verbose=True,
 )

s1.save("sta01.npz")
s2.save("sta02.npz")
s3.save("sta03.npz")
