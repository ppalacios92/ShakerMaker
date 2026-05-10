from shakermaker.station import Station
from shakermaker.stationlist import StationList

stations = []
d = 0.2
n = 30
x0, y0 = 5., 5.

for k in range(-n, n + 1):
    stations.append(Station([x0 + k * d, y0, 0.], metadata={"name": f"ew_{k}"}))
    stations.append(Station([x0, y0 + k * d, 0.], metadata={"name": f"ns_{k}"}))

cross = StationList(stations, metadata={"name": "cross_array"})
print(f"Cross array stations: {len(cross.list)}")
