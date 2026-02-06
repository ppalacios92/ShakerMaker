import numpy as np
from shakermaker.station import Station
from shakermaker.stationlist import StationList


class SurfaceGrid(StationList):
    """
    Surface grid of stations compatible with DRMBox interface.
    
    Parameters
    ----------
    x0 : array-like (3,)
        Center position [x, y, z] in km
    nelems : array-like (3,)
        Number of elements [nx, ny, nz]. Use nz=0 for single plane.
    h : float or array-like (3,)
        Spacing in km. Single value or [hx, hy, hz].
    mode : str
        'plane' - single surface at z=z0
        'hollow' - boundary surfaces only (like DRM box)
        'filled' - full 3D grid
    metadata : dict
        Additional metadata
    """
    
    def __init__(self, x0, nelems, h, mode='plane', metadata={}):
        StationList.__init__(self, [], metadata)
        
        self._x0 = np.array(x0, dtype=float)
        self._nelems = np.array(nelems, dtype=int)
        self._h = np.array([h, h, h] if np.isscalar(h) else h, dtype=float)
        self._mode = mode
        self._planes = []
        
        self._xmax = np.array([-np.inf, -np.inf, -np.inf])
        self._xmin = np.array([np.inf, np.inf, np.inf])
        
        self._create_stations()
        
        # QA station at center (required for DRM compatibility)
        self._new_station(self._x0, internal=False, name="QA")
        
        self._save_metadata()
    
    @property
    def nplanes(self):
        return len(self._planes)
    
    @property
    def planes(self):
        return self._planes
    
    def _new_station(self, x, internal=False, name=""):
        """Create and add a new station."""
        x = np.array(x, dtype=float)
        station = Station(x, internal=internal,
                         metadata={'id': self.nstations, 'name': name, 'internal': internal})
        self.add_station(station)
        
        self._xmax = np.maximum(self._xmax, x)
        self._xmin = np.minimum(self._xmin, x)
        
        return station
    
    def _create_stations(self):
        """Create stations based on mode."""
        nx, ny, nz = self._nelems
        hx, hy, hz = self._h
        
        # Grid dimensions
        lx, ly, lz = nx * hx, ny * hy, nz * hz
        
        # Origin (corner of grid)
        origin = self._x0 - np.array([lx/2, ly/2, 0])
        
        if self._mode == 'plane':
            self._create_plane(origin, nx, ny, hx, hy)
        elif self._mode == 'filled':
            self._create_filled(origin, nx, ny, nz, hx, hy, hz)
        elif self._mode == 'hollow':
            self._create_hollow(origin, nx, ny, nz, hx, hy, hz, lx, ly, lz)
    
    def _create_plane(self, origin, nx, ny, hx, hy):
        """Single plane at z=z0."""
        for i in range(nx + 1):
            for j in range(ny + 1):
                pos = origin + np.array([i * hx, j * hy, 0])
                self._new_station(pos, internal=False, name=f".{i}.{j}.0")
    
    def _create_filled(self, origin, nx, ny, nz, hx, hy, hz):
        """Full 3D grid."""
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    pos = origin + np.array([i * hx, j * hy, k * hz])
                    self._new_station(pos, internal=False, name=f".{i}.{j}.{k}")
    
    def _create_hollow(self, origin, nx, ny, nz, hx, hy, hz, lx, ly, lz):
        """Boundary surfaces only."""
        # Bottom face (z = z0)
        for i in range(nx + 1):
            for j in range(ny + 1):
                pos = origin + np.array([i * hx, j * hy, 0])
                self._new_station(pos, internal=False, name=f".{i}.{j}.bot")
        
        if nz > 0:
            # Top face (z = z0 + lz)
            for i in range(nx + 1):
                for j in range(ny + 1):
                    pos = origin + np.array([i * hx, j * hy, lz])
                    self._new_station(pos, internal=False, name=f".{i}.{j}.top")
            
            # Side faces (excluding edges already created)
            for k in range(1, nz):
                z = k * hz
                # Front (y=0) and Back (y=ly)
                for i in range(nx + 1):
                    self._new_station(origin + np.array([i * hx, 0, z]), False, f".{i}.front.{k}")
                    self._new_station(origin + np.array([i * hx, ly, z]), False, f".{i}.back.{k}")
                # Left (x=0) and Right (x=lx) - excluding corners
                for j in range(1, ny):
                    self._new_station(origin + np.array([0, j * hy, z]), False, f".left.{j}.{k}")
                    self._new_station(origin + np.array([lx, j * hy, z]), False, f".right.{j}.{k}")
    
    def _save_metadata(self):
        """Save metadata compatible with DRMBox."""
        self.metadata["h"] = self._h
        self.metadata["surfacegrid_x0"] = self._x0
        self.metadata["surfacegrid_nelems"] = self._nelems
        self.metadata["surfacegrid_mode"] = self._mode
        # DRMBox compatibility
        self.metadata["drmbox_x0"] = self._x0
        self.metadata["drmbox_xmax"] = self._xmax[0]
        self.metadata["drmbox_ymax"] = self._xmax[1]
        self.metadata["drmbox_zmax"] = self._xmax[2]
        self.metadata["drmbox_xmin"] = self._xmin[0]
        self.metadata["drmbox_ymin"] = self._xmin[1]
        self.metadata["drmbox_zmin"] = self._xmin[2]


StationList.register(SurfaceGrid)