from shakermaker import shakermaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.tools.plotting import ZENTPlot
from shakermaker.stf_extensions import Brune
from math import sqrt

dt=0.05
nfft=4096/8
dk=0.02
tb=100

crust = CrustModel(2)
Qp=600.
Qs=600.

Vs, nu, rho=1.000, 0.25, 2.
G = rho*Vs**2
M = 2*G*(1-nu)/(1-2*nu)
Vp = sqrt(M/rho)

crust.add_layer(0.02, Vp, Vs, rho, Qp, Qs)

Vs, nu, rho=2.500, 0.25, 2.
G = rho*Vs**2
M = 2*G*(1-nu)/(1-2*nu)
Vp = sqrt(M/rho)
crust.add_layer(0, Vp, Vs, rho, Qp, Qs)

t0, f0 = 0., 2.
stf = Brune(f0=f0, t0=t0)

z = 3.
s,d,r = 0., 30., 90.

for z in [1.]:
	source = PointSource([0,0,z], [s,d,r], tt=0, stf=stf)
	fault = FaultSource([source], metadata={"name":"source"})

x0,y0 = 0.,10.

s = Station([x0,y0,0], 
        metadata={
        "name":"Your House", 
        "filter_results":False, 
        "filter_parameters":{"fmax":10.}
        })
stations = StationList([s], metadata=s.metadata)

model = shakermaker.ShakerMaker(crust, fault, stations)
model.run(
 dt=dt,
 nfft=nfft,
 dk=dk,
 tb=tb,
 )

s.save("mystation.npz")

print(s)

ZENTPlot(s, show=True)
