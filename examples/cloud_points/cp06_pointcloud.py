import numpy as np
from shakermaker.station import Station
from shakermaker.stationlist import StationList

np.random.seed(42)
stations = []
n = 40
x0, y0, L = 5., 5., 4.

for _ in range(n):
    x = x0 + L * (np.random.random() - 0.5)
    y = y0 + L * (np.random.random() - 0.5)
    stations.append(Station([x, y, 0.], metadata={"name": f"r{_:02d}"}))

cloud = StationList(stations, metadata={"name": "random_cloud"})
print(f"Random point cloud stations: {len(cloud.list)}")
