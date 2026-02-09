from shakermaker.slw_extensions.hdf5stationlistwriter import HDF5StationListWriter
from shakermaker.sl_extensions import DRMBox
from shakermaker.station import Station
from shakermaker.version import shakermaker_version
import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import datetime
import os
# test1
class DRMHDF5StationListWriter(HDF5StationListWriter):

    def __init__(self, filename):
        HDF5StationListWriter.__init__(self, filename)

        self._h5file = None

        # Variables for legacy mode (when tmin, tmax, dt are not passed)
        self._velocities = {}
        self._tstart = np.infty
        self._tend = -np.infty
        self._dt = 0.
        self._gfs = {}
        
        # Variables for progressive mode
        self._progressive_mode = False
        self._t_final = None


    def initialize(self, station_list, num_samples, tmin=None, tmax=None, dt=None):
        from shakermaker.sl_extensions.SurfaceGrid import SurfaceGrid
        assert isinstance(station_list, (DRMBox, SurfaceGrid)), \
            "DRMHDF5StationListWriter.initialize - 'station_list' Should be a DRMBox or SurfaceGrid"

        # Form filename and create HDF5 dataset
        if self._filename is None or self._filename == "":
            self._filename = "DRMcase.hdf5"

        self._h5file = h5py.File(self._filename, mode="w")

        # Create groups
        grp_drm_data = self._h5file.create_group("/DRM_Data")
        grp_drm_qa_data = self._h5file.create_group("/DRM_QA_Data")
        self._h5file.create_group("/DRM_Metadata")
        self.nstations = station_list.nstations-1
        self.station_list = station_list

        # Create data
        grp_drm_data.create_dataset("xyz", (self.nstations, 3), dtype=np.double)
        grp_drm_data.create_dataset("internal", [self.nstations], dtype=bool)
        data_location = np.arange(0, self.nstations, dtype=np.int32) * 3
        grp_drm_data.create_dataset("data_location", data=data_location)

        grp_drm_qa_data.create_dataset("xyz", (1, 3), dtype=np.double)
        
        # Progressive mode if tmin, tmax, dt are passed
        if tmin is not None and tmax is not None and dt is not None:
            self._progressive_mode = True
            self._dt = dt
            self._tstart = tmin
            self._tend = tmax
            self._t_final = np.arange(tmin, tmax, dt)
            num_samples_final = len(self._t_final)
            
            # Pre-create datasets with known size
            grp_drm_data.create_dataset("velocity", (3 * self.nstations, num_samples_final), 
                                        dtype=np.double, chunks=(3, num_samples_final))
            grp_drm_data.create_dataset("displacement", (3 * self.nstations, num_samples_final), 
                                        dtype=np.double, chunks=(3, num_samples_final))
            grp_drm_data.create_dataset("acceleration", (3 * self.nstations, num_samples_final), 
                                        dtype=np.double, chunks=(3, num_samples_final))
            grp_drm_qa_data.create_dataset("velocity", (3, num_samples_final), 
                                           dtype=np.double, chunks=(3, num_samples_final))
            grp_drm_qa_data.create_dataset("displacement", (3, num_samples_final), 
                                           dtype=np.double, chunks=(3, num_samples_final))
            grp_drm_qa_data.create_dataset("acceleration", (3, num_samples_final), 
                                           dtype=np.double, chunks=(3, num_samples_final))
            
            # Write time metadata now
            grp_metadata = self._h5file['DRM_Metadata']
            grp_metadata.create_dataset("dt", data=self._dt)
            grp_metadata.create_dataset("tstart", data=self._tstart)
            grp_metadata.create_dataset("tend", data=self._tend)
            
            # Create group for GFs
            self._h5file.create_group('GF')
            
            print(f"[WRITER] Progressive mode enabled: tmin={tmin}, tmax={tmax}, dt={dt}, num_samples={num_samples_final}")
        else:
            # Legacy mode: accumulate in memory
            self._progressive_mode = False
            
            # Phase 2: Warn about legacy mode usage
            estimated_memory_mb = (self.nstations * num_samples * 3 * 8) / (1024 * 1024)
            estimated_memory_gb = estimated_memory_mb / 1024
            
            print(f"[WRITER] Legacy mode: accumulating in memory")
            print(f"[WRITER] WARNING: Legacy mode will accumulate ~{estimated_memory_gb:.2f} GB in RAM")
            
            if estimated_memory_gb > 10:
                print(f"[WRITER] CRITICAL: Memory usage will be very high!")
                print(f"[WRITER] RECOMMENDATION: Enable progressive mode by passing tmin, tmax, dt")
                print(f"[WRITER] Example: writer.initialize(receivers, num_samples, tmin=0, tmax=100, dt=0.05)")


    def write_metadata(self, metadata):
        assert self._h5file, "DRMHDF5StationListWriter.write_metadata uninitialized HDF5 file"

        grp_metadata = self._h5file['DRM_Metadata']

        # More metadata
        metadata["created_by"] = "---"   # os.getlogin() produces an error on slurm
        metadata["program_used"] = f"ShakeMaker version {shakermaker_version}"
        metadata["created_on"] = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

        for key, value in metadata.items():
            if key not in grp_metadata:  # Avoid duplicates with dt, tstart, tend
                grp_metadata.create_dataset(key, data=value)
                print(f"key = {key} {value}")

    def write_station(self, station, index):
        assert self._h5file, "DRMHDF5StationListWriter.write_station uninitialized HDF5 file"
        assert isinstance(station, Station), \
            "DRMHDF5StationListWriter.write_station 'station Should be subclass of Station"

        xyz = self._h5file['DRM_Data/xyz']
        xyz_QA = self._h5file['DRM_QA_Data/xyz']
        internal = self._h5file['DRM_Data/internal']

        zz, ee, nn, t = station.get_response()
        if self.transform_function:
            zz, ee, nn, t = self.transform_function(zz, ee, nn, t)

        if index < self.nstations:
            xyz[index, :] = station.x
            internal[index] = station.is_internal
        else:
            xyz_QA[0, :] = station.x

        is_QA = False
        if station.metadata["name"] == "QA":
            is_QA = True
        
        # Progressive mode - write directly
        if self._progressive_mode:
            self._write_station_progressive(station, index, zz, ee, nn, t, is_QA)
        else:
            # Legacy mode: accumulate in memory
            self._velocities[index] = (zz, ee, nn, t, is_QA)
            self._tstart = min(t[0], self._tstart)
            self._tend = max(t[-1], self._tend)
            self._dt = t[1] - t[0]

            gf_dict = station.get_greens_functions()
            if gf_dict:
                self._gfs[index] = gf_dict

    def _write_station_progressive(self, station, index, zz, ee, nn, t, is_QA):
        """Write a station directly to HDF5 without accumulating in memory."""
        
        def interpolatorfun(told, yold, tnew):
            return interp1d(told, yold,
                fill_value=(yold[0], yold[-1]),
                bounds_error=False)(tnew)
        
        t_final = self._t_final
        dt = self._dt
        
        # Interpolate to t_final
        ve = interpolatorfun(t, ee, t_final)
        vn = interpolatorfun(t, nn, t_final)
        vz = interpolatorfun(t, zz, t_final)
        
        # Calculate acceleration and displacement
        Nt = len(ve)
        ae = np.zeros(Nt)
        ae[1:] = (ve[1:] - ve[0:-1]) / dt
        an = np.zeros(Nt)
        an[1:] = (vn[1:] - vn[0:-1]) / dt
        az = np.zeros(Nt)
        az[1:] = (vz[1:] - vz[0:-1]) / dt
        
        de = cumulative_trapezoid(ve, t_final, initial=0.)
        dn = cumulative_trapezoid(vn, t_final, initial=0.)
        dz = cumulative_trapezoid(vz, t_final, initial=0.)
        
        # Write directly to HDF5
        if not is_QA:
            self._h5file['DRM_Data/displacement'][3 * index, :] = de
            self._h5file['DRM_Data/displacement'][3 * index + 1, :] = dn
            self._h5file['DRM_Data/displacement'][3 * index + 2, :] = dz
            self._h5file['DRM_Data/velocity'][3 * index, :] = ve
            self._h5file['DRM_Data/velocity'][3 * index + 1, :] = vn
            self._h5file['DRM_Data/velocity'][3 * index + 2, :] = vz
            self._h5file['DRM_Data/acceleration'][3 * index, :] = ae
            self._h5file['DRM_Data/acceleration'][3 * index + 1, :] = an
            self._h5file['DRM_Data/acceleration'][3 * index + 2, :] = az
        else:
            self._h5file['DRM_QA_Data/velocity'][0, :] = ve
            self._h5file['DRM_QA_Data/velocity'][1, :] = vn
            self._h5file['DRM_QA_Data/velocity'][2, :] = vz
            self._h5file['DRM_QA_Data/displacement'][0, :] = de
            self._h5file['DRM_QA_Data/displacement'][1, :] = dn
            self._h5file['DRM_QA_Data/displacement'][2, :] = dz
            self._h5file['DRM_QA_Data/acceleration'][0, :] = ae
            self._h5file['DRM_QA_Data/acceleration'][1, :] = an
            self._h5file['DRM_QA_Data/acceleration'][2, :] = az
        
        # Write GFs directly if save_gf is enabled
        gf_dict = station.get_greens_functions()
        if gf_dict:
            self._write_station_gfs_progressive(index, gf_dict)
        
        # Flush to ensure data is written to disk
        self._h5file.flush()

    def _write_station_gfs_progressive(self, sta_idx, gf_dict):
        """Write Green's functions of a station directly to HDF5."""
        grp_gf = self._h5file['GF']
        grp_sta = grp_gf.create_group(f'sta_{sta_idx}')
        
        for sub_idx, (z, e, n, t, tdata, t0) in gf_dict.items():
            grp_sub = grp_sta.create_group(f'sub_{sub_idx}')
            grp_sub.create_dataset('z', data=z, compression='gzip')
            grp_sub.create_dataset('e', data=e, compression='gzip')
            grp_sub.create_dataset('n', data=n, compression='gzip')
            grp_sub.create_dataset('t', data=t, compression='gzip')
            grp_sub.create_dataset('tdata', data=tdata, compression='gzip')
            grp_sub.create_dataset('t0', data=t0)

    def close(self):
        
        # If progressive mode, just close
        if self._progressive_mode:
            # Save GF database info if exists (for stages created in run_fast_faster)
            if hasattr(self, 'gf_db_pairs') and self.gf_db_pairs is not None:
                grp_gf_db = self._h5file.create_group('GF_Database_Info')
                grp_gf_db.create_dataset('pairs_to_compute', data=self.gf_db_pairs, compression='gzip')
                grp_gf_db.create_dataset('dh_of_pairs', data=self.gf_db_dh, compression='gzip')
                grp_gf_db.create_dataset('zrec_of_pairs', data=self.gf_db_zrec, compression='gzip')
                grp_gf_db.create_dataset('zsrc_of_pairs', data=self.gf_db_zsrc, compression='gzip')
                grp_gf_db.attrs['delta_h'] = self.gf_db_delta_h
                grp_gf_db.attrs['delta_v_rec'] = self.gf_db_delta_v_rec
                grp_gf_db.attrs['delta_v_src'] = self.gf_db_delta_v_src
                print(f"[WRITER] GF Database info saved to output file")

            # Write node-to-donor mapping for GF lookup
            if hasattr(self, 'node_pair_mapping') and self.node_pair_mapping is not None:
                grp_map = self._h5file.create_group('Node_Mapping')
                grp_map.create_dataset('node_to_pair_mapping', data=self.node_pair_mapping, compression='gzip')
                grp_map.create_dataset('pairs_to_compute', data=self.pairs_to_compute_for_mapping, compression='gzip')
            
            print(f"[WRITER] Progressive mode: closing file (all data already written)")
            self._h5file.close()
            return
        
        # Legacy mode: original behavior
        t_final = np.arange(self._tstart, self._tend, self._dt)
        num_samples = len(t_final)

        grp_drm_data = self._h5file['DRM_Data/']
        grp_drm_qa_data = self._h5file['DRM_QA_Data/']

        grp_drm_data.create_dataset("velocity", (3 * self.nstations, num_samples), dtype=np.double, chunks=(3, num_samples))
        grp_drm_data.create_dataset("displacement", (3 * self.nstations, num_samples), dtype=np.double, chunks=(3, num_samples))
        grp_drm_data.create_dataset("acceleration", (3 * self.nstations, num_samples), dtype=np.double, chunks=(3, num_samples))
        grp_drm_qa_data.create_dataset("velocity", (3 , num_samples), dtype=np.double, chunks=(3, num_samples))
        grp_drm_qa_data.create_dataset("displacement", (3 , num_samples), dtype=np.double, chunks=(3, num_samples))
        grp_drm_qa_data.create_dataset("acceleration", (3 , num_samples), dtype=np.double, chunks=(3, num_samples))
        grp_metadata = self._h5file['DRM_Metadata']
        grp_metadata.create_dataset("dt", data=self._dt)
        grp_metadata.create_dataset("tstart", data=self._tstart)
        grp_metadata.create_dataset("tend", data=self._tend)

        def interpolatorfun(told, yold, tnew):
            return interp1d(told, yold,
                fill_value=(yold[0], yold[-1]),
                bounds_error=False)(tnew)

        velocity = self._h5file['DRM_Data/velocity']
        displacement = self._h5file['DRM_Data/displacement']
        acceleration = self._h5file['DRM_Data/acceleration']
        for index in self._velocities:
            zz, ee, nn, t, is_QA = self._velocities[index]
            ve = interpolatorfun(t, ee, t_final)
            vn = interpolatorfun(t, nn, t_final)
            vz = interpolatorfun(t, zz, t_final)
            dt = t_final[1] - t_final[0]
            Nt = len(ve)
            ae = np.zeros(Nt)
            ae[1:] = (ve[1:] - ve[0:-1]) / dt
            an = np.zeros(Nt)
            an[1:] = (vn[1:] - vn[0:-1]) / dt
            az = np.zeros(Nt)
            az[1:] = (vz[1:] - vz[0:-1]) / dt
            de = cumulative_trapezoid(ve, t_final, initial=0.)
            dn = cumulative_trapezoid(vn, t_final, initial=0.)
            dz = cumulative_trapezoid(vz, t_final, initial=0.)
            if not is_QA:
                displacement[3 * index, :] = de
                displacement[3 * index + 1, :] = dn
                displacement[3 * index + 2, :] = dz
                velocity[3 * index, :] = ve
                velocity[3 * index + 1, :] = vn
                velocity[3 * index + 2, :] = vz
                acceleration[3 * index, :] = ae
                acceleration[3 * index + 1, :] = an
                acceleration[3 * index + 2, :] = az
            else:
                self._h5file['DRM_QA_Data/velocity'][0, :] = ve
                self._h5file['DRM_QA_Data/velocity'][1, :] = vn
                self._h5file['DRM_QA_Data/velocity'][2, :] = vz
                self._h5file['DRM_QA_Data/displacement'][0, :] = de
                self._h5file['DRM_QA_Data/displacement'][1, :] = dn
                self._h5file['DRM_QA_Data/displacement'][2, :] = dz
                self._h5file['DRM_QA_Data/acceleration'][0, :] = ae
                self._h5file['DRM_QA_Data/acceleration'][1, :] = an
                self._h5file['DRM_QA_Data/acceleration'][2, :] = az

        if self._gfs:
            self._write_gfs()

        # Save GF database info if exists (for stages created in run_fast_faster)
        if hasattr(self, 'gf_db_pairs') and self.gf_db_pairs is not None:
            grp_gf_db = self._h5file.create_group('GF_Database_Info')
            grp_gf_db.create_dataset('pairs_to_compute', data=self.gf_db_pairs, compression='gzip')
            grp_gf_db.create_dataset('dh_of_pairs', data=self.gf_db_dh, compression='gzip')
            grp_gf_db.create_dataset('zrec_of_pairs', data=self.gf_db_zrec, compression='gzip')
            grp_gf_db.create_dataset('zsrc_of_pairs', data=self.gf_db_zsrc, compression='gzip')
            grp_gf_db.attrs['delta_h'] = self.gf_db_delta_h
            grp_gf_db.attrs['delta_v_rec'] = self.gf_db_delta_v_rec
            grp_gf_db.attrs['delta_v_src'] = self.gf_db_delta_v_src
            print(f"[WRITER] GF Database info saved to output file")

        # Write node-to-donor mapping for GF lookup
        if hasattr(self, 'node_pair_mapping') and self.node_pair_mapping is not None:
            grp_map = self._h5file.create_group('Node_Mapping')
            grp_map.create_dataset('node_to_pair_mapping', data=self.node_pair_mapping, compression='gzip')
            grp_map.create_dataset('pairs_to_compute', data=self.pairs_to_compute_for_mapping, compression='gzip')
                
        self._h5file.close()

    def _write_gfs(self):
        grp = self._h5file.create_group('GF')
        for sta_idx, gf_dict in self._gfs.items():
            grp_sta = grp.create_group(f'sta_{sta_idx}')
            for sub_idx, (z, e, n, t, tdata, t0) in gf_dict.items():
                grp_sub = grp_sta.create_group(f'sub_{sub_idx}')
                grp_sub.create_dataset('z', data=z, compression='gzip')
                grp_sub.create_dataset('e', data=e, compression='gzip')
                grp_sub.create_dataset('n', data=n, compression='gzip')
                grp_sub.create_dataset('t', data=t, compression='gzip')
                grp_sub.create_dataset('tdata', data=tdata, compression='gzip')
                grp_sub.create_dataset('t0', data=t0)
        print(f"[WRITER] _write_gfs() done!")


HDF5StationListWriter.register(DRMHDF5StationListWriter)