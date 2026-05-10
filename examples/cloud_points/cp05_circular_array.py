import numpy as np
from shakermaker.station import Station
from shakermaker.stationlist import StationList

stations = []
R = 3.0
n = 24
x0, y0 = 5., 5.

for k in range(n):
    theta = 2 * np.pi * k / n
    x = x0 + R * np.cos(theta)
    y = y0 + R * np.sin(theta)
    stations.append(Station([x, y, 0.], metadata={"name": f"c{k:02d}"}))

ring = StationList(stations, metadata={"name": "circular_array"})
print(f"Circular array stations: {len(ring.list)}")
