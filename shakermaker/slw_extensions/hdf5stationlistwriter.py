from shakermaker.stationlist import StationList
from shakermaker.stationlistwriter import StationListWriter
from shakermaker.station import Station
import h5py
import numpy as np
from scipy.interpolate import interp1d

# Try to import MPI for rank detection
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
except:
    rank = 0


class HDF5StationListWriter(StationListWriter):

    def __init__(self, filename):
        StationListWriter.__init__(self, filename)

        self._h5file = None
        
        # Attributes for progressive mode
        self._progressive_mode = False
        self._t_final = None
        self._dt = None
        self._tstart = None
        self._tend = None

    def initialize(self, station_list, num_samples, tmin=None, tmax=None, dt=None, writer_mode=None):
        assert isinstance(station_list, StationList), \
            "HDF5StationListWriter.initialize - 'station_list' Should be subclass of StationList"

        # Determine mode and calculate num_samples_actual
        if writer_mode == 'progressive' and tmin is not None and tmax is not None and dt is not None:
            # PROGRESSIVE MODE
            self._progressive_mode = True
            self._t_final = np.arange(tmin, tmax, dt)
            self._dt = dt
            self._tstart = tmin
            self._tend = tmax
            num_samples_actual = len(self._t_final)
            
            if rank == 0:
                print(f"[WRITER] Progressive mode enabled: tmin={tmin}, tmax={tmax}, dt={dt}, num_samples={num_samples_actual}")
        else:
            # LEGACY MODE
            self._progressive_mode = False
            num_samples_actual = num_samples
            
            if rank == 0:
                print(f"[WRITER] Legacy mode: num_samples={num_samples_actual}")

        # Form filename and create HDF5 dataset
        if self._filename is None or self._filename == "":
            self._filename = "case.hdf5"

        self._h5file = h5py.File(self._filename, mode="w")

        # Create groups
        grp_data = self._h5file.create_group("/Data")
        self._h5file.create_group("/Metadata")

        # Create data - velocity only (no displacement/acceleration for stations)
        grp_data.create_dataset("velocity", (3 * station_list.nstations, num_samples_actual), 
                                dtype=np.double, chunks=(3, num_samples_actual))
        grp_data.create_dataset("xyz", (station_list.nstations, 3), dtype=np.double)
        
        data_location = np.arange(0, station_list.nstations, dtype=np.int32) * 3
        grp_data.create_dataset("data_location", data=data_location)
        
        # Create GF group for progressive mode
        if self._progressive_mode:
            self._h5file.create_group('GF')

    def write_metadata(self, metadata):
        assert self._h5file, "HDF5StationListWriter.write_metadata uninitialized HDF5 file"

        grp_metadata = self._h5file['Metadata']
        
        # Write time metadata if in progressive mode
        if self._progressive_mode:
            if 'dt' not in grp_metadata:
                grp_metadata.create_dataset('dt', data=self._dt)
            if 'tstart' not in grp_metadata:
                grp_metadata.create_dataset('tstart', data=self._tstart)
            if 'tend' not in grp_metadata:
                grp_metadata.create_dataset('tend', data=self._tend)
        
        # Write other metadata
        for key, value in metadata.items():
            if key not in grp_metadata:
                grp_metadata.create_dataset(key, data=value)

    def write_station(self, station, index):
        assert self._h5file, "HDF5StationListWriter.write_station uninitialized HDF5 file"
        assert isinstance(station, Station), \
            "HDF5StationListWriter.write_station 'station Should be subclass of Station"

        velocity = self._h5file['Data/velocity']
        xyz = self._h5file['Data/xyz']

        zz, ee, nn, t = station.get_response()

        # Write velocity based on mode
        if self._progressive_mode:
            # Progressive mode: interpolate to common time grid
            self._write_station_progressive(station, index, zz, ee, nn, t)
        else:
            # Legacy mode: write directly without interpolation
            velocity[3 * index, :] = ee
            velocity[3 * index + 1, :] = nn
            velocity[3 * index + 2, :] = zz
            
            # Check if GFs need to be saved (not supported in legacy mode)
            gf_dict = station.get_greens_functions()
            if gf_dict and rank == 0:
                print(f"[WRITER] WARNING: save_gf=True detected but writer_mode='legacy'. "
                      f"Green's functions will not be saved. Use writer_mode='progressive' to save GFs.")

        # Write coordinates (common for both modes)
        xyz[index, :] = station.x

    def _write_station_progressive(self, station, index, zz, ee, nn, t):
        def interpolatorfun(told, yold, tnew):
            return interp1d(told, yold,
                fill_value=(yold[0], yold[-1]),
                bounds_error=False)(tnew)
        
        # Interpolate to common time grid
        ve = interpolatorfun(t, ee, self._t_final)
        vn = interpolatorfun(t, nn, self._t_final)
        vz = interpolatorfun(t, zz, self._t_final)
        
        # Write velocity to HDF5
        velocity = self._h5file['Data/velocity']
        velocity[3 * index, :] = ve
        velocity[3 * index + 1, :] = vn
        velocity[3 * index + 2, :] = vz
        
        # Write Green's functions if save_gf is enabled
        gf_dict = station.get_greens_functions()
        if gf_dict:
            self._write_station_gfs_progressive(index, gf_dict)
        
        # Flush to ensure data is written to disk
        self._h5file.flush()

    def _write_station_gfs_progressive(self, sta_idx, gf_dict):
        grp_gf = self._h5file['GF']
        
        # Check if group already exists
        sta_group_name = f'sta_{sta_idx}'
        if sta_group_name in grp_gf:
            if rank == 0:
                print(f"[WRITER] WARNING: Group {sta_group_name} already exists, skipping GF write")
            return
        
        # Create station group
        grp_sta = grp_gf.create_group(sta_group_name)
        
        # Write each subfault's Green's functions
        for sub_idx, (z, e, n, t, tdata, t0) in gf_dict.items():
            grp_sub = grp_sta.create_group(f'sub_{sub_idx}')
            grp_sub.create_dataset('z', data=z, compression='gzip')
            grp_sub.create_dataset('e', data=e, compression='gzip')
            grp_sub.create_dataset('n', data=n, compression='gzip')
            grp_sub.create_dataset('t', data=t, compression='gzip')
            grp_sub.create_dataset('tdata', data=tdata, compression='gzip')
            grp_sub.create_dataset('t0', data=t0)

    def close(self):
        """Close the HDF5 file"""
        assert self._h5file, "HDF5StationListWriter.close uninitialized HDF5 file"

        self._h5file.close()  # ← CORREGIDO: era writer.close()


StationListWriter.register(HDF5StationListWriter)