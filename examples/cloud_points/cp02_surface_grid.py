from shakermaker.station import Station
from shakermaker.stationlist import StationList

stations = []
nx, ny = 10, 10
dx, dy = 0.5, 0.5
for i in range(nx):
    for j in range(ny):
        x = i * dx
        y = j * dy
        sta = Station([x, y, 0.], metadata={"name": f"s{i:02d}{j:02d}"})
        stations.append(sta)

grid = StationList(stations, metadata={"name": "surface_grid"})
print(f"Surface grid stations: {len(grid.list)}")
