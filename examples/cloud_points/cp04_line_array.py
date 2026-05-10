from shakermaker.station import Station
from shakermaker.stationlist import StationList

stations = []
d = 0.1
n = 50

for k in range(n):
    x = k * d
    stations.append(Station([x, 0., 0.], metadata={"name": f"l{k:03d}"}))

line = StationList(stations, metadata={"name": "line_array"})
print(f"Line array stations: {len(line.list)}")
