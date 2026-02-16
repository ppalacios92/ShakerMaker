from shakermaker.stationlist import StationList
from shakermaker.stationlistwriter import StationListWriter
from shakermaker.station import Station
import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid


class HDF5StationListWriter(StationListWriter):
    """
    HDF5 writer for StationList results.
    
    Supports two writing modes:
    - 'legacy': Accumulates data in memory, writes at close()
    - 'progressive': Writes each station immediately to disk
    """

    def __init__(self, filename):
        StationListWriter.__init__(self, filename)
        
        self._h5file = None
        self.nstations = 0
        
        # Storage for legacy mode
        self._velocities = {}
        self._tstart = np.inf
        self._tend = -np.inf
        self._dt = 0.0
        
        # Progressive mode variables
        self._progressive_mode = False
        self._t_final = None

    def initialize(self, station_list, num_samples, 
                   tmin=None, tmax=None, dt=None, writer_mode='legacy'):
        """
        Initialize HDF5 writer.
        
        Parameters
        ----------
        station_list : StationList
            List of stations
        num_samples : int
            Number of samples (used only in legacy mode without tmin/tmax/dt)
        tmin, tmax, dt : float, optional
            Time parameters. Required for progressive mode.
        writer_mode : str
            'legacy' or 'progressive'
        """
        assert isinstance(station_list, StationList), \
            "HDF5StationListWriter.initialize - 'station_list' should be StationList"

        # Set filename
        if self._filename is None or self._filename == "":
            self._filename = "case.hdf5"

        self._h5file = h5py.File(self._filename, mode="w")
        self.nstations = station_list.nstations

        # Create groups
        grp_data = self._h5file.create_group("/Data")
        self._h5file.create_group("/Metadata")

        # Create xyz and internal datasets (known size)
        grp_data.create_dataset("xyz", (self.nstations, 3), dtype=np.double)
        grp_data.create_dataset("internal", (self.nstations,), dtype=bool)
        data_location = np.arange(0, self.nstations, dtype=np.int32) * 3
        grp_data.create_dataset("data_location", data=data_location)

        # Determine mode
        if writer_mode == 'progressive':
            if tmin is None or tmax is None or dt is None:
                raise ValueError(
                    "Progressive mode requires tmin, tmax, and dt parameters."
                )
            self._progressive_mode = True
            self._dt = dt
            self._tstart = tmin
            self._tend = tmax
            self._t_final = np.arange(tmin, tmax, dt)
            num_samples_final = len(self._t_final)
            
            # Pre-create velocity dataset with correct size
            grp_data.create_dataset("velocity", (3 * self.nstations, num_samples_final),
                                    dtype=np.double, chunks=(3, num_samples_final))
            
            # Write time metadata
            grp_meta = self._h5file['Metadata']
            grp_meta.create_dataset("dt", data=self._dt)
            grp_meta.create_dataset("tstart", data=self._tstart)
            grp_meta.create_dataset("tend", data=self._tend)
            
        else:
            # Legacy mode: velocity dataset created at close()
            self._progressive_mode = False

    def write_metadata(self, metadata):
        """Write metadata to HDF5."""
        assert self._h5file, "HDF5StationListWriter.write_metadata - uninitialized file"

        grp_metadata = self._h5file['Metadata']
        for key, value in metadata.items():
            if key not in grp_metadata:
                grp_metadata.create_dataset(key, data=value)

    def write_station(self, station, index):
        """Write a single station to HDF5."""
        assert self._h5file, "HDF5StationListWriter.write_station - uninitialized file"
        assert isinstance(station, Station), \
            "HDF5StationListWriter.write_station - 'station' should be Station"

        zz, ee, nn, t = station.get_response()
        
        # Write xyz and internal
        self._h5file['Data/xyz'][index, :] = station.x
        self._h5file['Data/internal'][index] = station.is_internal

        if self._progressive_mode:
            self._write_station_progressive(index, zz, ee, nn, t)
        else:
            # Legacy mode: accumulate in memory
            self._velocities[index] = (zz, ee, nn, t)
            self._tstart = min(t[0], self._tstart)
            self._tend = max(t[-1], self._tend)
            self._dt = t[1] - t[0]

    def _write_station_progressive(self, index, zz, ee, nn, t):
        """Write station directly to HDF5 (progressive mode)."""
        
        def interpolate(told, yold, tnew):
            return interp1d(told, yold,
                           fill_value=(yold[0], yold[-1]),
                           bounds_error=False)(tnew)
        
        t_final = self._t_final
        
        # Interpolate to final time grid
        ve = interpolate(t, ee, t_final)
        vn = interpolate(t, nn, t_final)
        vz = interpolate(t, zz, t_final)
        
        # Write to HDF5
        self._h5file['Data/velocity'][3 * index, :] = ve
        self._h5file['Data/velocity'][3 * index + 1, :] = vn
        self._h5file['Data/velocity'][3 * index + 2, :] = vz
        
        self._h5file.flush()

    def close(self):
        """Close HDF5 file."""
        assert self._h5file, "HDF5StationListWriter.close - uninitialized file"

        if self._progressive_mode:
            # Data already written
            self._h5file.close()
            return

        # Legacy mode: create velocity dataset and write all data
        t_final = np.arange(self._tstart, self._tend, self._dt)
        num_samples = len(t_final)

        grp_data = self._h5file['Data']
        grp_data.create_dataset("velocity", (3 * self.nstations, num_samples),
                                dtype=np.double, chunks=(3, num_samples))

        # Write time metadata
        grp_meta = self._h5file['Metadata']
        if 'dt' not in grp_meta:
            grp_meta.create_dataset("dt", data=self._dt)
        if 'tstart' not in grp_meta:
            grp_meta.create_dataset("tstart", data=self._tstart)
        if 'tend' not in grp_meta:
            grp_meta.create_dataset("tend", data=self._tend)

        def interpolate(told, yold, tnew):
            return interp1d(told, yold,
                           fill_value=(yold[0], yold[-1]),
                           bounds_error=False)(tnew)

        velocity = self._h5file['Data/velocity']
        
        for index, (zz, ee, nn, t) in self._velocities.items():
            ve = interpolate(t, ee, t_final)
            vn = interpolate(t, nn, t_final)
            vz = interpolate(t, zz, t_final)
            
            velocity[3 * index, :] = ve
            velocity[3 * index + 1, :] = vn
            velocity[3 * index + 2, :] = vz

        self._h5file.close()


StationListWriter.register(HDF5StationListWriter)