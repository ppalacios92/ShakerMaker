import os
import numpy as np
from typing import Optional, Dict, List
from .crustmodel import CrustModel
import matplotlib.pyplot as plt

# Finite Fault Stochastic Process (FFSP) source model
class FFSPSource:
    """Finite Fault Stochastic Process source model (Liu & Archuleta)"""
    
    def __init__(self,
                 id_sf_type: int, freq_min: float, freq_max: float,
                 fault_length: float, fault_width: float,
                 x_hypc: float, y_hypc: float, depth_hypc: float,
                 xref_hypc: float, yref_hypc: float,
                 magnitude: float, fc_main_1: float, fc_main_2: float,
                 rv_avg: float,
                 ratio_rise: float,
                 strike: float, dip: float, rake: float,
                 pdip_max: float, prake_max: float,
                 nsubx: int, nsuby: int,
                 nb_taper_trbl: List[int],
                 seeds: List[int],
                 id_ran1: int, id_ran2: int,
                 angle_north_to_x: float,
                 is_moment: int,
                 crust_model: CrustModel,
                 output_name: str = "FFSP_OUTPUT",
                 verbose: bool = True):
        """
        Initialize FFSP source model.
        
        Parameters
        ----------
        id_sf_type : int
            Slip function type
        freq_min, freq_max : float
            Frequency range (Hz)
        fault_length, fault_width : float
            Fault dimensions (km)
        x_hypc, y_hypc, depth_hypc : float
            Hypocenter position (km)
        xref_hypc, yref_hypc : float
            Reference position for hypocenter
        magnitude : float
            Moment magnitude
        fc_main_1, fc_main_2 : float
            Corner frequencies (Hz)
        rv_avg : float
            Average rupture velocity (km/s)
        ratio_rise : float
            Rise time ratio
        strike, dip, rake : float
            Fault geometry (degrees)
        pdip_max, prake_max : float
            Maximum perturbations (degrees)
        nsubx, nsuby : int
            Number of subfaults along strike and dip
        nb_taper_trbl : List[int]
            Taper zones [top, right, bottom, left]
        seeds : List[int]
            Random seeds [seed1, seed2, seed3]
        id_ran1, id_ran2 : int
            Realization range [start, end]
        angle_north_to_x : float
            Rotation angle (degrees)
        is_moment : int
            Result flag
        crust_model : CrustModel
            Velocity model
        output_name : str, optional
            Output file prefix
        verbose : bool, optional
            Print progress messages
        """
        
        if not isinstance(crust_model, CrustModel):
            raise TypeError("crust_model must be a CrustModel instance")

        # Store all parameters
        self.params = {
            'id_sf_type': id_sf_type, 'freq_min': freq_min, 'freq_max': freq_max,
            'fault_length': fault_length, 'fault_width': fault_width,
            'x_hypc': x_hypc, 'y_hypc': y_hypc, 'depth_hypc': depth_hypc,
            'xref_hypc': xref_hypc, 'yref_hypc': yref_hypc,
            'magnitude': magnitude, 'fc_main_1': fc_main_1, 'fc_main_2': fc_main_2,
            'rv_avg': rv_avg,
            'ratio_rise': ratio_rise,
            'strike': strike, 'dip': dip, 'rake': rake,
            'pdip_max': pdip_max, 'prake_max': prake_max,
            'nsubx': nsubx, 'nsuby': nsuby,
            'nb_taper_trbl': nb_taper_trbl,
            'seeds': seeds,
            'id_ran1': id_ran1, 'id_ran2': id_ran2,
            'angle_north_to_x': angle_north_to_x,
            'is_moment': is_moment,
            'output_name': output_name,
        }
        
        self.crust_model = crust_model
        self.output_name = output_name
        self.verbose = verbose
        
        # Results storage
        self.all_realizations = None
        self.best_realization = None
        self.source_stats = None
        self.subfaults = None

        # Subfault dimensions
        self.dx = fault_length / nsubx
        self.dy = fault_width / nsuby
        self.area = self.dx * self.dy
    
    def run(self) -> Dict:
        """
        Run FFSP to generate fault realizations using Fortran wrapper.
        Supports MPI parallelization with automatic data gathering.
        
        Returns
        -------
        Dict
            Best realization subfault data (only valid on rank 0 if using MPI)
        """
        
        # Detect MPI environment
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            use_mpi = comm.Get_size() > 1
            rank = comm.Get_rank() if use_mpi else 0
            nprocs = comm.Get_size() if use_mpi else 1
        except ImportError:
            use_mpi = False
            rank = 0
            nprocs = 1
        
        # Distribute realizations across MPI ranks
        total_models = self.params['id_ran2'] - self.params['id_ran1'] + 1
        
        if use_mpi:
            models_per_rank = total_models // nprocs
            remainder = total_models % nprocs
            
            if rank < remainder:
                start = rank * (models_per_rank + 1) + self.params['id_ran1']
                end = start + models_per_rank
            else:
                start = rank * models_per_rank + remainder + self.params['id_ran1']
                end = start + models_per_rank - 1
            
            my_n_models = end - start + 1
        else:
            start = self.params['id_ran1']
            end = self.params['id_ran2']
            my_n_models = total_models
        
        if self.verbose and rank == 0:
            print(f"\nRunning FFSP: {total_models} realizations on {nprocs} MPI ranks\n")
        
        # Import Fortran wrapper module
        try:
            from . import ffsp_core
        except ImportError:
            # Fallback: add ffsp directory to path
            import sys
            ffsp_dir = os.path.join(os.path.dirname(__file__), 'ffsp')
            sys.path.insert(0, ffsp_dir)
            import ffsp_core
        
        result = ffsp_core.ffsp_run_wrapper(
            self.params['id_sf_type'],
            self.params['freq_min'],
            self.params['freq_max'],
            self.params['fault_length'],
            self.params['fault_width'],
            self.params['x_hypc'],
            self.params['y_hypc'],
            self.params['depth_hypc'],
            self.params['xref_hypc'],
            self.params['yref_hypc'],
            self.params['magnitude'],
            self.params['fc_main_1'],
            self.params['fc_main_2'],
            self.params['rv_avg'],
            self.params['ratio_rise'],
            self.params['strike'],
            self.params['dip'],
            self.params['rake'],
            self.params['pdip_max'],
            self.params['prake_max'],
            self.params['nsubx'],
            self.params['nsuby'],
            np.array(self.params['nb_taper_trbl'], dtype=np.int32),
            np.array(self.params['seeds'], dtype=np.int32),
            start, end,
            self.params['angle_north_to_x'],
            self.params['is_moment'],
            self.crust_model.nlayers,
            self.crust_model.a.astype(np.float32),
            self.crust_model.b.astype(np.float32),
            self.crust_model.rho.astype(np.float32),
            self.crust_model.d.astype(np.float32),
            self.crust_model.qa.astype(np.float32),
            self.crust_model.qb.astype(np.float32),
        )
        
        # Unpack results from Fortran wrapper (27 values)
        (n_realizations, npts, x, y, z, slip, rupture_time, 
         rise_time, peak_time, strike, dip, rake,
         ave_tr, ave_tp, ave_vr, err_spectra, pdf,
         # Spectral data
         ntime_spec, nphf_spec, lnpt_spec,
         stf_time, stf,
         freq_spec, moment_rate, dcf,
         freq_center, logmean_synth, logmean_dcf) = result
        
        if use_mpi:
            if rank == 0:
                all_x = [x]
                all_y = [y]
                all_z = [z]
                all_slip = [slip]
                all_rupture_time = [rupture_time]
                all_rise_time = [rise_time]
                all_peak_time = [peak_time]
                all_strike = [strike]
                all_dip = [dip]
                all_rake = [rake]
                all_ave_tr = [ave_tr]
                all_ave_tp = [ave_tp]
                all_ave_vr = [ave_vr]
                all_err_spectra = [err_spectra]
                all_pdf = [pdf]
                
                for r in range(1, nprocs):
                    recv_data = comm.recv(source=r, tag=r)
                    
                    all_x.append(recv_data['x'])
                    all_y.append(recv_data['y'])
                    all_z.append(recv_data['z'])
                    all_slip.append(recv_data['slip'])
                    all_rupture_time.append(recv_data['rupture_time'])
                    all_rise_time.append(recv_data['rise_time'])
                    all_peak_time.append(recv_data['peak_time'])
                    all_strike.append(recv_data['strike'])
                    all_dip.append(recv_data['dip'])
                    all_rake.append(recv_data['rake'])
                    all_ave_tr.append(recv_data['ave_tr'])
                    all_ave_tp.append(recv_data['ave_tp'])
                    all_ave_vr.append(recv_data['ave_vr'])
                    all_err_spectra.append(recv_data['err_spectra'])
                    all_pdf.append(recv_data['pdf'])
                
                # Concatenate all data
                x = np.concatenate(all_x, axis=1)
                y = np.concatenate(all_y, axis=1)
                z = np.concatenate(all_z, axis=1)
                slip = np.concatenate(all_slip, axis=1)
                rupture_time = np.concatenate(all_rupture_time, axis=1)
                rise_time = np.concatenate(all_rise_time, axis=1)
                peak_time = np.concatenate(all_peak_time, axis=1)
                strike = np.concatenate(all_strike, axis=1)
                dip = np.concatenate(all_dip, axis=1)
                rake = np.concatenate(all_rake, axis=1)
                ave_tr = np.concatenate(all_ave_tr)
                ave_tp = np.concatenate(all_ave_tp)
                ave_vr = np.concatenate(all_ave_vr)
                err_spectra = np.concatenate(all_err_spectra)
                pdf = np.concatenate(all_pdf)
                
                n_realizations = total_models
                
            else:
                send_data = {
                    'x': x,
                    'y': y,
                    'z': z,
                    'slip': slip,
                    'rupture_time': rupture_time,
                    'rise_time': rise_time,
                    'peak_time': peak_time,
                    'strike': strike,
                    'dip': dip,
                    'rake': rake,
                    'ave_tr': ave_tr,
                    'ave_tp': ave_tp,
                    'ave_vr': ave_vr,
                    'err_spectra': err_spectra,
                    'pdf': pdf,
                }
                
                comm.send(send_data, dest=0, tag=rank)
                
                self.all_realizations = None
                self.source_stats = None
                self.best_realization = None
                self.subfaults = None
                self.active_realization = None
                
                return None
        
        # =====================================================================
        # Store results (only rank 0 in MPI mode, or single process)
        # =====================================================================
        
        # Store all realizations in compatible format
        self.all_realizations = {
            'n_realizations': n_realizations,
            'nseg': 1,
            'npts': npts,
            'x': x,
            'y': y,
            'z': z,
            'slip': slip,
            'rupture_time': rupture_time,
            'rise_time': rise_time,
            'peak_time': peak_time,
            'strike': strike,
            'dip': dip,
            'rake': rake,
        }
        
        # Store statistics (for plots)
        self.source_stats = {
            'source_score': {
                'n_realizations': n_realizations,
                'ave_tr': ave_tr,
                'ave_tp': ave_tp,
                'ave_vr': ave_vr,
                'err_spectra': err_spectra,
                'pdf': pdf,
            },
            # Spectral data (from best realization computed by rank 0 or single process)
            'spectrum': {
                'freq': freq_spec[:nphf_spec],
                'moment_rate_synth': moment_rate[:nphf_spec],
                'moment_rate_dcf': dcf[:nphf_spec],
            },
            'stf_time': {
                'time': stf_time[:ntime_spec],
                'stf': stf[:ntime_spec],
            },
            'spectrum_octave': {
                'freq_center': freq_center[:lnpt_spec],
                'logmean_synth': logmean_synth[:lnpt_spec],
                'logmean_dcf': logmean_dcf[:lnpt_spec],
            }
        }
        
        # Identify best realization (minimum PDF)
        best_idx = np.argmin(pdf)
        self.best_realization = {
            'nseg': 1,
            'npts': npts,
            'x': x[:, best_idx],
            'y': y[:, best_idx],
            'z': z[:, best_idx],
            'slip': slip[:, best_idx],
            'rupture_time': rupture_time[:, best_idx],
            'rise_time': rise_time[:, best_idx],
            'peak_time': peak_time[:, best_idx],
            'strike': strike[:, best_idx],
            'dip': dip[:, best_idx],
            'rake': rake[:, best_idx],
        }
        
        # Set best realization as active subfaults
        self.subfaults = self.best_realization
        self.active_realization = 'best'
        
        if self.verbose and (rank == 0 or not use_mpi):
            print(f"\nFFSP Complete: {n_realizations} realizations | Best: PDF={pdf[best_idx]:.4f}\n")
        
        return self.subfaults

    def get_realization(self, index: int) -> Dict:
        """
        Get specific realization by index (0-based).
        
        Parameters
        ----------
        index : int
            Realization index
            
        Returns
        -------
        Dict
            Realization data
        """
        
        if not hasattr(self, 'all_realizations') or not self.all_realizations:
            raise RuntimeError("No realizations available. Run FFSP first.")
        
        n = self.all_realizations['n_realizations']
        if not (0 <= index < n):
            raise IndexError(f"Index {index} out of range [0, {n-1}]")
        
        return {
            'nseg': self.all_realizations['nseg'],
            'npts': self.all_realizations['npts'],
            'x': self.all_realizations['x'][:, index],
            'y': self.all_realizations['y'][:, index],
            'z': self.all_realizations['z'][:, index],
            'slip': self.all_realizations['slip'][:, index],
            'rupture_time': self.all_realizations['rupture_time'][:, index],
            'rise_time': self.all_realizations['rise_time'][:, index],
            'peak_time': self.all_realizations['peak_time'][:, index],
            'strike': self.all_realizations['strike'][:, index],
            'dip': self.all_realizations['dip'][:, index],
            'rake': self.all_realizations['rake'][:, index],
        }

    def set_active_realization(self, index: int):
        """Set active realization for plotting."""
        self.subfaults = self.get_realization(index)
        self.active_realization = index

    def get_subfaults(self) -> Dict:
        """Get currently active subfault data."""
        if self.subfaults is None:
            raise RuntimeError("No active realization. Call set_active_realization() first.")
        return self.subfaults 
    
    # ============ FILE WRITING METHODS ============
    
    
    def write_hdf5(self, filename: str):
        """
        Write results to HDF5 file (modern, efficient format).
        Stores all realizations and metadata in a single compressed file.
        
        Parameters
        ----------
        filename : str
            Output HDF5 filename (will add .h5 if not present)
        """
        
        if self.all_realizations is None:
            raise RuntimeError("No realizations available. Run FFSP first.")
        
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 output. Install with: pip install h5py")
        
        if not filename.endswith('.h5'):
            filename += '.h5'
        
        print(f"Writing HDF5: {filename}")
        
        with h5py.File(filename, 'w') as f:
            grp_realizations = f.create_group('realizations')
            grp_best = f.create_group('best_realization')
            grp_stats = f.create_group('statistics')
            grp_params = f.create_group('parameters')
            
            for key, val in self.all_realizations.items():
                if isinstance(val, (int, float)):
                    grp_realizations.attrs[key] = val
                else:
                    grp_realizations.create_dataset(key, data=val, compression='gzip')
            
            if self.best_realization is not None:
                for key, val in self.best_realization.items():
                    if isinstance(val, (int, float)):
                        grp_best.attrs[key] = val
                    else:
                        grp_best.create_dataset(key, data=val, compression='gzip')
            
            if self.source_stats is not None:
                grp_score = grp_stats.create_group('source_score')
                for key, val in self.source_stats['source_score'].items():
                    if isinstance(val, (int, float)):
                        grp_score.attrs[key] = val
                    else:
                        grp_score.create_dataset(key, data=val, compression='gzip')
                
                if 'spectrum' in self.source_stats:
                    grp_spectrum = grp_stats.create_group('spectrum')
                    for key, val in self.source_stats['spectrum'].items():
                        grp_spectrum.create_dataset(key, data=val, compression='gzip')
                    
                    grp_stf = grp_stats.create_group('stf_time')
                    for key, val in self.source_stats['stf_time'].items():
                        grp_stf.create_dataset(key, data=val, compression='gzip')
                    
                    grp_octave = grp_stats.create_group('spectrum_octave')
                    for key, val in self.source_stats['spectrum_octave'].items():
                        grp_octave.create_dataset(key, data=val, compression='gzip')
            
            for key, val in self.params.items():
                if isinstance(val, (int, float, str)):
                    grp_params.attrs[key] = val
                elif isinstance(val, list):
                    grp_params.create_dataset(key, data=np.array(val))
            
            grp_params.attrs['dx'] = self.dx
            grp_params.attrs['dy'] = self.dy
            grp_params.attrs['area'] = self.area
            
            grp_crust = grp_params.create_group('crust_model')
            grp_crust.attrs['nlayers'] = self.crust_model.nlayers
            grp_crust.create_dataset('d', data=self.crust_model.d)
            grp_crust.create_dataset('a', data=self.crust_model.a)
            grp_crust.create_dataset('b', data=self.crust_model.b)
            grp_crust.create_dataset('rho', data=self.crust_model.rho)
            grp_crust.create_dataset('qa', data=self.crust_model.qa)
            grp_crust.create_dataset('qb', data=self.crust_model.qb)
        
        print(f"✓ HDF5 saved\n")
    
    def load_hdf5(self, filename: str):
        """
        Load results from HDF5 file into existing FFSPSource object.
        
        Parameters
        ----------
        filename : str
            Input HDF5 filename
        """
        
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 input. Install with: pip install h5py")
        
        if not filename.endswith('.h5'):
            filename += '.h5'
        
        print(f"Loading HDF5: {filename}")
        
        with h5py.File(filename, 'r') as f:
            # Load all realizations
            grp_realizations = f['realizations']
            self.all_realizations = {}
            for key in grp_realizations.keys():
                self.all_realizations[key] = grp_realizations[key][:]
            for key in grp_realizations.attrs.keys():
                self.all_realizations[key] = grp_realizations.attrs[key]
            
            # Load best realization
            grp_best = f['best_realization']
            self.best_realization = {}
            for key in grp_best.keys():
                self.best_realization[key] = grp_best[key][:]
            for key in grp_best.attrs.keys():
                self.best_realization[key] = grp_best.attrs[key]
            
            # Load statistics
            grp_stats = f['statistics']
            self.source_stats = {}
            
            grp_score = grp_stats['source_score']
            self.source_stats['source_score'] = {}
            for key in grp_score.keys():
                self.source_stats['source_score'][key] = grp_score[key][:]
            for key in grp_score.attrs.keys():
                self.source_stats['source_score'][key] = grp_score.attrs[key]
            
            if 'spectrum' in grp_stats:
                self.source_stats['spectrum'] = {}
                for key in grp_stats['spectrum'].keys():
                    self.source_stats['spectrum'][key] = grp_stats['spectrum'][key][:]
                
                self.source_stats['stf_time'] = {}
                for key in grp_stats['stf_time'].keys():
                    self.source_stats['stf_time'][key] = grp_stats['stf_time'][key][:]
                
                self.source_stats['spectrum_octave'] = {}
                for key in grp_stats['spectrum_octave'].keys():
                    self.source_stats['spectrum_octave'][key] = grp_stats['spectrum_octave'][key][:]
            
            # Load parameters
            grp_params = f['parameters']
            self.params = {}
            for key in grp_params.attrs.keys():
                self.params[key] = grp_params.attrs[key]
            for key in grp_params.keys():
                if key != 'crust_model':  # Skip crust_model group
                    self.params[key] = grp_params[key][:].tolist()
            
            # Reconstruct subfault geometry attributes
            self.dx = self.params['dx']
            self.dy = self.params['dy']
            self.area = self.params['area']
            self.output_name = self.params.get('output_name', 'FFSP_OUTPUT')
            self.verbose = True  # Default
            
            # Load crust model - reconstruct layer by layer
            grp_crust = grp_params['crust_model']
            from .crustmodel import CrustModel
            
            nlayers = grp_crust.attrs['nlayers']
            self.crust_model = CrustModel(nlayers)
            
            # Add each layer
            d = grp_crust['d'][:]
            a = grp_crust['a'][:]    # vp
            b = grp_crust['b'][:]    # vs
            rho = grp_crust['rho'][:]
            qa = grp_crust['qa'][:]
            qb = grp_crust['qb'][:]
            
            for i in range(nlayers):
                self.crust_model.add_layer(d[i], a[i], b[i], rho[i], qa[i], qb[i])
            
            # Set active realization
            self.subfaults = self.best_realization
            self.active_realization = 'best'
        
        print(f"✓ HDF5 loaded\n")
    
    @classmethod
    def from_hdf5(cls, filename: str):
        """
        Create FFSPSource object from HDF5 file (class method).
        This is the recommended way to load saved results.
        
        Parameters
        ----------
        filename : str
            Input HDF5 filename
            
        Returns
        -------
        FFSPSource
            Fully initialized FFSPSource object with all data loaded
            
        Examples
        --------
        >>> # Save results
        >>> source.run()
        >>> source.write_hdf5('results.h5')
        >>> 
        >>> # Load in new session
        >>> source = FFSPSource.from_hdf5('results.h5')
        >>> source.plot_spectral_comparison()
        """
        
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 input. Install with: pip install h5py")
        
        if not filename.endswith('.h5'):
            filename += '.h5'
        
        with h5py.File(filename, 'r') as f:
            # Load parameters first
            grp_params = f['parameters']
            params = {}
            for key in grp_params.attrs.keys():
                params[key] = grp_params.attrs[key]
            for key in grp_params.keys():
                if key != 'crust_model':
                    params[key] = grp_params[key][:].tolist()
            
            # Load crust model - reconstruct layer by layer
            grp_crust = grp_params['crust_model']
            from .crustmodel import CrustModel
            
            nlayers = grp_crust.attrs['nlayers']
            crust_model = CrustModel(nlayers)
            
            # Add each layer
            d = grp_crust['d'][:]
            a = grp_crust['a'][:]    # vp
            b = grp_crust['b'][:]    # vs
            rho = grp_crust['rho'][:]
            qa = grp_crust['qa'][:]
            qb = grp_crust['qb'][:]
            
            for i in range(nlayers):
                crust_model.add_layer(d[i], a[i], b[i], rho[i], qa[i], qb[i])
        
        # Create object using __init__ with loaded parameters
        obj = cls(
            id_sf_type=params['id_sf_type'],
            freq_min=params['freq_min'],
            freq_max=params['freq_max'],
            fault_length=params['fault_length'],
            fault_width=params['fault_width'],
            x_hypc=params['x_hypc'],
            y_hypc=params['y_hypc'],
            depth_hypc=params['depth_hypc'],
            xref_hypc=params['xref_hypc'],
            yref_hypc=params['yref_hypc'],
            magnitude=params['magnitude'],
            fc_main_1=params['fc_main_1'],
            fc_main_2=params['fc_main_2'],
            rv_avg=params['rv_avg'],
            ratio_rise=params['ratio_rise'],
            strike=params['strike'],
            dip=params['dip'],
            rake=params['rake'],
            pdip_max=params['pdip_max'],
            prake_max=params['prake_max'],
            nsubx=params['nsubx'],
            nsuby=params['nsuby'],
            nb_taper_trbl=params['nb_taper_trbl'],
            seeds=params['seeds'],
            id_ran1=params['id_ran1'],
            id_ran2=params['id_ran2'],
            angle_north_to_x=params['angle_north_to_x'],
            is_moment=params['is_moment'],
            crust_model=crust_model,
            output_name=params.get('output_name', 'FFSP_OUTPUT'),
            verbose=False  # Don't print during load
        )
        
        # Now load the data into the initialized object
        obj.load_hdf5(filename)
        
        return obj
    
    def write_ffsp_format(self, output_dir: str, output_name: str = None):
        if self.all_realizations is None:
            raise RuntimeError("No realizations available. Run FFSP first.")
        
        if output_name is None:
            output_name = self.output_name
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Writing FFSP: {output_dir}")
        
        n = self.all_realizations['n_realizations']
        npts = self.all_realizations['npts']
        
        for i in range(n):
            filename = os.path.join(output_dir, f"{output_name}.{i+1:03d}")
            with open(filename, 'w') as f:
                f.write(f"{self.all_realizations['nseg']} {npts}\n")
                for j in range(npts):
                    f.write(f"{self.all_realizations['x'][j, i]:15.6e} ")
                    f.write(f"{self.all_realizations['y'][j, i]:15.6e} ")
                    f.write(f"{self.all_realizations['z'][j, i]:15.6e} ")
                    f.write(f"{self.all_realizations['slip'][j, i]:15.6e} ")
                    f.write(f"{self.all_realizations['rupture_time'][j, i]:15.6e} ")
                    f.write(f"{self.all_realizations['rise_time'][j, i]:15.6e} ")
                    f.write(f"{self.all_realizations['peak_time'][j, i]:15.6e} ")
                    f.write(f"{self.all_realizations['strike'][j, i]:15.6e} ")
                    f.write(f"{self.all_realizations['dip'][j, i]:15.6e} ")
                    f.write(f"{self.all_realizations['rake'][j, i]:15.6e}\n")
        
        if self.best_realization is not None:
            filename = os.path.join(output_dir, f"{output_name}.bst")
            with open(filename, 'w') as f:
                f.write(f"{self.best_realization['nseg']} {self.best_realization['npts']}\n")
                for j in range(self.best_realization['npts']):
                    f.write(f"{self.best_realization['x'][j]:15.6e} ")
                    f.write(f"{self.best_realization['y'][j]:15.6e} ")
                    f.write(f"{self.best_realization['z'][j]:15.6e} ")
                    f.write(f"{self.best_realization['slip'][j]:15.6e} ")
                    f.write(f"{self.best_realization['rupture_time'][j]:15.6e} ")
                    f.write(f"{self.best_realization['rise_time'][j]:15.6e} ")
                    f.write(f"{self.best_realization['peak_time'][j]:15.6e} ")
                    f.write(f"{self.best_realization['strike'][j]:15.6e} ")
                    f.write(f"{self.best_realization['dip'][j]:15.6e} ")
                    f.write(f"{self.best_realization['rake'][j]:15.6e}\n")
        
        if self.source_stats is not None:
            filename = os.path.join(output_dir, "source_model.score")
            stats = self.source_stats['source_score']
            with open(filename, 'w') as f:
                f.write(f"{n}\n")
                f.write("Target: average Risetime= 0.0 average peaktime= 0.0\n")
                for i in range(n):
                    f.write(f"{output_name}.{i+1:03d}\n")
                    f.write(f"{stats['ave_tr'][i]:15.5e} ")
                    f.write(f"{stats['ave_tp'][i]:15.5e} ")
                    f.write(f"{stats['ave_vr'][i]:15.5e} ")
                    f.write(f"{stats['err_spectra'][i]:15.5e} ")
                    f.write(f"{stats['pdf'][i]:15.5e}\n")
        
        filename = os.path.join(output_dir, "source_model.list")
        with open(filename, 'w') as f:
            f.write(f"{self.params['id_sf_type']} ")
            f.write(f"{self.params['nsubx']} ")
            f.write(f"{self.params['nsuby']} ")
            f.write(f"{self.dx} ")
            f.write(f"{self.dy} ")
            f.write(f"{self.params['x_hypc']} ")
            f.write(f"{self.params['y_hypc']}\n")
            f.write(f"{self.params['xref_hypc']} ")
            f.write(f"{self.params['yref_hypc']} ")
            f.write(f"{self.params['angle_north_to_x']}\n")
            f.write(f"{output_name}.bst\n")
        
        filename = os.path.join(output_dir, "source_model.params")
        with open(filename, 'w') as f:
            for key, val in self.params.items():
                if isinstance(val, list):
                    f.write(f"{key} {' '.join(map(str, val))}\n")
                else:
                    f.write(f"{key} {val}\n")
        
        filename = os.path.join(output_dir, "velocity.vel")
        with open(filename, 'w') as f:
            f.write(f"{self.crust_model.nlayers}\n")
            for i in range(self.crust_model.nlayers):
                f.write(f"{self.crust_model.d[i]:.6e} ")
                f.write(f"{self.crust_model.a[i]:.6e} ")
                f.write(f"{self.crust_model.b[i]:.6e} ")
                f.write(f"{self.crust_model.rho[i]:.6e} ")
                f.write(f"{self.crust_model.qa[i]:.6e} ")
                f.write(f"{self.crust_model.qb[i]:.6e}\n")
        
        if self.source_stats and 'spectrum' in self.source_stats:
            filename = os.path.join(output_dir, "calsvf.dat")
            spectrum = self.source_stats['spectrum']
            with open(filename, 'w') as f:
                f.write(f"{len(spectrum['freq'])}\n")
                for i in range(len(spectrum['freq'])):
                    f.write(f"{spectrum['freq'][i]:15.6e} ")
                    f.write(f"{spectrum['moment_rate_synth'][i]:15.6e} ")
                    f.write(f"{spectrum['moment_rate_dcf'][i]:15.6e}\n")
            
            filename = os.path.join(output_dir, "calsvf_tim.dat")
            stf = self.source_stats['stf_time']
            with open(filename, 'w') as f:
                f.write(f"{len(stf['time'])}\n")
                for i in range(len(stf['time'])):
                    f.write(f"{stf['time'][i]:15.6e} ")
                    f.write(f"{stf['stf'][i]:15.6e}\n")
            
            filename = os.path.join(output_dir, "logsvf.dat")
            octave = self.source_stats['spectrum_octave']
            with open(filename, 'w') as f:
                f.write(f"{len(octave['freq_center'])}\n")
                for i in range(len(octave['freq_center'])):
                    f.write(f"{octave['freq_center'][i]:15.6e} ")
                    f.write(f"{octave['logmean_synth'][i]:15.6e} ")
                    f.write(f"{octave['logmean_dcf'][i]:15.6e}\n")
        
        print(f"✓ FFSP saved\n")



    def load_ffsp_format(self, input_dir: str, output_name: str = "FFSP_OUTPUT"):
            print(f"Loading FFSP: {input_dir}")
            
            params_file = os.path.join(input_dir, "source_model.params")
            params = {}
            with open(params_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    key = parts[0]
                    if key in ['nb_taper_trbl', 'seeds']:
                        params[key] = [int(x) for x in parts[1:]]
                    elif key in ['id_sf_type', 'nsubx', 'nsuby', 'id_ran1', 'id_ran2', 'is_moment']:
                        params[key] = int(parts[1])
                    elif key == 'output_name':
                        params[key] = parts[1]
                    else:
                        params[key] = float(parts[1])
            
            self.params = params
            self.dx = params['dx']
            self.dy = params['dy']
            self.area = params['area']
            self.output_name = params.get('output_name', 'FFSP_OUTPUT')
            self.verbose = True
            
            vel_file = os.path.join(input_dir, "velocity.vel")
            from .crustmodel import CrustModel
            with open(vel_file, 'r') as f:
                nlayers = int(f.readline().strip())
                self.crust_model = CrustModel(nlayers)
                for i in range(nlayers):
                    values = f.readline().split()
                    self.crust_model.add_layer(float(values[0]), float(values[1]), float(values[2]),
                                              float(values[3]), float(values[4]), float(values[5]))
            
            score_file = os.path.join(input_dir, "source_model.score")
            with open(score_file, 'r') as f:
                n_realizations = int(f.readline().strip())
                f.readline()
                ave_tr, ave_tp, ave_vr, err_spectra, pdf = [], [], [], [], []
                for i in range(n_realizations):
                    f.readline()
                    values = f.readline().split()
                    ave_tr.append(float(values[0]))
                    ave_tp.append(float(values[1]))
                    ave_vr.append(float(values[2]))
                    err_spectra.append(float(values[3]))
                    pdf.append(float(values[4]))
            
            npts = int(params['nsubx']) * int(params['nsuby'])
            x = np.zeros((npts, n_realizations))
            y = np.zeros((npts, n_realizations))
            z = np.zeros((npts, n_realizations))
            slip = np.zeros((npts, n_realizations))
            rupture_time = np.zeros((npts, n_realizations))
            rise_time = np.zeros((npts, n_realizations))
            peak_time = np.zeros((npts, n_realizations))
            strike = np.zeros((npts, n_realizations))
            dip = np.zeros((npts, n_realizations))
            rake = np.zeros((npts, n_realizations))
            
            for i in range(n_realizations):
                filename = os.path.join(input_dir, f"{output_name}.{i+1:03d}")
                with open(filename, 'r') as f:
                    header = f.readline().split()
                    nseg = int(header[0])
                    for j in range(npts):
                        values = f.readline().split()
                        x[j, i] = float(values[0])
                        y[j, i] = float(values[1])
                        z[j, i] = float(values[2])
                        slip[j, i] = float(values[3])
                        rupture_time[j, i] = float(values[4])
                        rise_time[j, i] = float(values[5])
                        peak_time[j, i] = float(values[6])
                        strike[j, i] = float(values[7])
                        dip[j, i] = float(values[8])
                        rake[j, i] = float(values[9])
            
            self.all_realizations = {
                'n_realizations': n_realizations, 'nseg': nseg, 'npts': npts,
                'x': x, 'y': y, 'z': z, 'slip': slip, 'rupture_time': rupture_time,
                'rise_time': rise_time, 'peak_time': peak_time, 'strike': strike,
                'dip': dip, 'rake': rake,
            }
            
            best_file = os.path.join(input_dir, f"{output_name}.bst")
            best_x, best_y, best_z = np.zeros(npts), np.zeros(npts), np.zeros(npts)
            best_slip, best_rupture_time, best_rise_time = np.zeros(npts), np.zeros(npts), np.zeros(npts)
            best_peak_time, best_strike, best_dip, best_rake = np.zeros(npts), np.zeros(npts), np.zeros(npts), np.zeros(npts)
            
            with open(best_file, 'r') as f:
                f.readline()
                for j in range(npts):
                    values = f.readline().split()
                    best_x[j], best_y[j], best_z[j] = float(values[0]), float(values[1]), float(values[2])
                    best_slip[j], best_rupture_time[j], best_rise_time[j] = float(values[3]), float(values[4]), float(values[5])
                    best_peak_time[j], best_strike[j], best_dip[j], best_rake[j] = float(values[6]), float(values[7]), float(values[8]), float(values[9])
            
            self.best_realization = {
                'nseg': nseg, 'npts': npts, 'x': best_x, 'y': best_y, 'z': best_z,
                'slip': best_slip, 'rupture_time': best_rupture_time, 'rise_time': best_rise_time,
                'peak_time': best_peak_time, 'strike': best_strike, 'dip': best_dip, 'rake': best_rake,
            }
            
            self.source_stats = {
                'source_score': {
                    'n_realizations': n_realizations,
                    'ave_tr': np.array(ave_tr), 'ave_tp': np.array(ave_tp), 'ave_vr': np.array(ave_vr),
                    'err_spectra': np.array(err_spectra), 'pdf': np.array(pdf),
                }
            }
            
            calsvf_file = os.path.join(input_dir, "calsvf.dat")
            if os.path.exists(calsvf_file):
                with open(calsvf_file, 'r') as f:
                    nphf_spec = int(f.readline().strip())
                    freq_spec, moment_rate, dcf = np.zeros(nphf_spec), np.zeros(nphf_spec), np.zeros(nphf_spec)
                    for i in range(nphf_spec):
                        values = f.readline().split()
                        freq_spec[i], moment_rate[i], dcf[i] = float(values[0]), float(values[1]), float(values[2])
                self.source_stats['spectrum'] = {'freq': freq_spec, 'moment_rate_synth': moment_rate, 'moment_rate_dcf': dcf}
                
                with open(os.path.join(input_dir, "calsvf_tim.dat"), 'r') as f:
                    ntime_spec = int(f.readline().strip())
                    time, stf = np.zeros(ntime_spec), np.zeros(ntime_spec)
                    for i in range(ntime_spec):
                        values = f.readline().split()
                        time[i], stf[i] = float(values[0]), float(values[1])
                self.source_stats['stf_time'] = {'time': time, 'stf': stf}
                
                with open(os.path.join(input_dir, "logsvf.dat"), 'r') as f:
                    lnpt_spec = int(f.readline().strip())
                    freq_center, logmean_synth, logmean_dcf = np.zeros(lnpt_spec), np.zeros(lnpt_spec), np.zeros(lnpt_spec)
                    for i in range(lnpt_spec):
                        values = f.readline().split()
                        freq_center[i], logmean_synth[i], logmean_dcf[i] = float(values[0]), float(values[1]), float(values[2])
                self.source_stats['spectrum_octave'] = {'freq_center': freq_center, 'logmean_synth': logmean_synth, 'logmean_dcf': logmean_dcf}
            
            self.subfaults = self.best_realization
            self.active_realization = 'best'
            
            print(f"✓ FFSP loaded\n")

    @classmethod
    def from_ffsp_format(cls, input_dir: str, output_name: str = "FFSP_OUTPUT"):
        params_file = os.path.join(input_dir, "source_model.params")
        params = {}
        with open(params_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                key = parts[0]
                if key in ['nb_taper_trbl', 'seeds']:
                    params[key] = [int(x) for x in parts[1:]]
                elif key in ['id_sf_type', 'nsubx', 'nsuby', 'id_ran1', 'id_ran2', 'is_moment']:
                    params[key] = int(parts[1])
                elif key == 'output_name':
                    params[key] = parts[1]
                else:
                    params[key] = float(parts[1])
        
        vel_file = os.path.join(input_dir, "velocity.vel")
        from .crustmodel import CrustModel
        with open(vel_file, 'r') as f:
            nlayers = int(f.readline().strip())
            crust_model = CrustModel(nlayers)
            for i in range(nlayers):
                values = f.readline().split()
                crust_model.add_layer(float(values[0]), float(values[1]), float(values[2]),
                                     float(values[3]), float(values[4]), float(values[5]))
        
        obj = cls(
            id_sf_type=params['id_sf_type'], freq_min=params['freq_min'], freq_max=params['freq_max'],
            fault_length=params['fault_length'], fault_width=params['fault_width'],
            x_hypc=params['x_hypc'], y_hypc=params['y_hypc'], depth_hypc=params['depth_hypc'],
            xref_hypc=params['xref_hypc'], yref_hypc=params['yref_hypc'],
            magnitude=params['magnitude'], fc_main_1=params['fc_main_1'], fc_main_2=params['fc_main_2'],
            rv_avg=params['rv_avg'], ratio_rise=params['ratio_rise'],
            strike=params['strike'], dip=params['dip'], rake=params['rake'],
            pdip_max=params['pdip_max'], prake_max=params['prake_max'],
            nsubx=params['nsubx'], nsuby=params['nsuby'],
            nb_taper_trbl=params['nb_taper_trbl'], seeds=params['seeds'],
            id_ran1=params['id_ran1'], id_ran2=params['id_ran2'],
            angle_north_to_x=params['angle_north_to_x'], is_moment=params['is_moment'],
            crust_model=crust_model, output_name=params.get('output_name', 'FFSP_OUTPUT'),
            verbose=False
        )
        
        obj.load_ffsp_format(input_dir, output_name)
        return obj



    # ============ PLOTTING METHODS ============
    
    def plot_histogram(self, field='slip', bins=50, figsize=(7, 5)):
        """Plot histogram of field across all realizations."""
        valid_fields = ['x', 'y', 'z', 'slip', 'rupture_time', 'rise_time', 
                       'peak_time', 'strike', 'dip', 'rake']
        if field not in valid_fields:
            raise ValueError(f"field must be one of {valid_fields}, got '{field}'")
            
        plt.figure(figsize=figsize)
        
        # Plot all realizations
        for i in range(self.all_realizations['n_realizations']):
            var = self.all_realizations[field][:, i]
            plt.hist(var, bins=bins, alpha=0.4, label=f'Rlz{i+1}')
        
        # Plot best realization
        var_best = self.best_realization[field]
        plt.hist(var_best, bins=bins, histtype='step', color='red', 
                linewidth=2, label='Best')
        
        # Labels
        field_labels = {
            'x': 'North (m)', 'y': 'East (m)', 'z': 'Depth (m)',
            'slip': 'Slip (m)', 'rupture_time': 'Rupture Time (s)',
            'rise_time': 'Rise Time (s)', 'peak_time': 'Peak Time (s)',
            'strike': 'Strike (°)', 'dip': 'Dip (°)', 'rake': 'Rake (°)'
        }
        
        plt.xlabel(field_labels[field])
        plt.ylabel('Frequency')
        plt.title(f'{field_labels[field]} - Multiple Realizations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_spacial_distribution(self, figsize=(10, 8), field='rise_time', 
                                      cmap='coolwarm', show_contours=True, 
                                      contour_field='rupture_time', show_hypocenter=True,
                                      save_fig=False, model_name='model'):
        """Plot spatial distribution of subfault parameters."""
        
        nx = int(self.params['nsubx'])
        ny = int(self.params['nsuby'])
        lx = self.params['fault_length']
        ly = self.params['fault_width']
        dx = self.dx
        dy = self.dy
        cxp = self.params['x_hypc']
        cyp = self.params['y_hypc']
        
        # Validate fields
        valid_fields = ['slip', 'rupture_time', 'rise_time', 'peak_time', 
                       'strike', 'dip', 'rake']
        if field not in valid_fields:
            raise ValueError(f"field must be one of {valid_fields}")
        if contour_field not in valid_fields:
            raise ValueError(f"contour_field must be one of {valid_fields}")
        # Prepare data
        field_data = np.transpose(self.subfaults[field].reshape(nx, ny))
        contour_data = np.transpose(self.subfaults[contour_field].reshape(nx, ny))
        x = np.linspace(-lx/2, lx/2, nx)
        y = np.linspace(-ly/2, ly/2, ny)
        X, Y = np.meshgrid(x, y)
        # Labels
        field_labels = {
            'slip': 'Slip [m]', 'rupture_time': 'Rupture Time [s]',
            'rise_time': 'Rise Time [s]', 'peak_time': 'Peak Time [s]',
            'strike': 'Strike [°]', 'dip': 'Dip [°]', 'rake': 'Rake [°]',
        }
        # Plot
        plt.figure(figsize=figsize)
        plt.imshow(field_data[::-1], cmap=cmap, 
                  extent=(-lx/2-dx/2, lx/2+dx/2, -ly/2-dy/2, ly/2+dy/2), 
                  interpolation='nearest')
        plt.colorbar(label=field_labels[field], shrink=ly/lx)
        # Contours
        if show_contours:
            contours = plt.contour(X, Y, contour_data, 8, colors='blue', linewidths=1.5)
            plt.clabel(contours, fontsize=10, fmt='%2.1f', inline=1)
            
        # Hypocenter
        if show_hypocenter:
            plt.scatter(cxp-lx/2, cyp-ly/2, c='red', s=300, marker='*', 
                       edgecolors='white', linewidth=2, label='Hypocenter', zorder=10)
            plt.legend(loc='upper right')
        plt.xlabel('Along Strike [km]')
        plt.ylabel('Down Dip [km]')
        
        if show_contours:
            plt.title(f'{field_labels[field]} Distribution (contours: {field_labels[contour_field]})', 
                     fontsize=14, fontweight='bold')
        else:
            plt.title(f'{field_labels[field]} Distribution', fontsize=14, fontweight='bold')
        
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig:
            plt.savefig(f'{model_name}_spatial_distribution.svg', 
                        format='svg', 
                        dpi=600, 
                        bbox_inches='tight', 
                        transparent=True,
                        facecolor='none')
        
        plt.show()

    def plot_quality_metrics(self, figsize=(14, 5)):
        """Plot PDF and Spectral Error side by side."""
        if self.source_stats is None:
            print("No source statistics available. Run simulation first.")
            return
            
        stats = self.source_stats['source_score']
        n = stats['n_realizations']
        idx = np.arange(1, n + 1)
        best_idx = np.argmin(stats['pdf'])
        
        plt.figure(figsize=figsize)
        
        # Subplot 1: PDF
        plt.subplot(1, 2, 1)
        colors = ['tab:green' if i == best_idx else 'tab:blue' for i in range(n)]
        plt.bar(idx, stats['pdf'], color=colors, edgecolor='black', alpha=0.7)
        plt.axhline(stats['pdf'][best_idx], color='tab:red', ls='--', lw=2, 
                   label=f'Best = {stats["pdf"][best_idx]:.3f}')
        plt.xlabel('Realization', fontsize=12)
        plt.ylabel('PDF', fontsize=12)
        plt.title(f'PDF Quality Metric (Best: #{best_idx+1})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Subplot 2: Spectral Error
        plt.subplot(1, 2, 2)
        colors = ['tab:green' if i == best_idx else 'tab:orange' for i in range(n)]
        plt.bar(idx, stats['err_spectra'], color=colors, edgecolor='black', alpha=0.7)
        plt.axhline(stats['err_spectra'][best_idx], color='tab:red', ls='--', lw=2, 
                   label=f'Best = {stats["err_spectra"][best_idx]:.4f}')
        plt.xlabel('Realization', fontsize=12)
        plt.ylabel('Spectral Error (RMS)', fontsize=12)
        plt.title(f'Spectral Error (Best: #{best_idx+1})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_temporal_metrics(self, figsize=(15, 5)):
        """Plot Rise Time, Peak Time, and Rupture Velocity."""
        if self.source_stats is None:
            print("No source statistics available. Run simulation first.")
            return

        stats = self.source_stats['source_score']
        n = stats['n_realizations']
        idx = np.arange(1, n + 1)
        best_idx = np.argmin(stats['pdf'])

        plt.figure(figsize=figsize)
        
        # Subplot 1: Rise Time
        plt.subplot(1, 3, 1)
        plt.plot(idx, stats['ave_tr'], 'o-', color='tab:purple', lw=2, markersize=6)
        plt.plot(best_idx+1, stats['ave_tr'][best_idx], 's', color='tab:green', 
                markersize=15, label=f'Best = {stats["ave_tr"][best_idx]:.3f} s', zorder=10)
        plt.xlabel('Realization', fontsize=12)
        plt.ylabel('Average Rise Time (s)', fontsize=12)
        plt.title(f'Rise Time (Best: #{best_idx+1})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Subplot 2: Peak Time
        plt.subplot(1, 3, 2)
        plt.plot(idx, stats['ave_tp'], 'o-', color='tab:orange', lw=2, markersize=6)
        plt.plot(best_idx+1, stats['ave_tp'][best_idx], 's', color='tab:green', 
                markersize=15, label=f'Best = {stats["ave_tp"][best_idx]:.3f} s', zorder=10)
        plt.xlabel('Realization', fontsize=12)
        plt.ylabel('Average Peak Time (s)', fontsize=12)
        plt.title(f'Peak Time (Best: #{best_idx+1})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Subplot 3: Rupture Velocity
        plt.subplot(1, 3, 3)
        plt.plot(idx, stats['ave_vr'], 'o-', color='tab:cyan', lw=2, markersize=6)
        plt.plot(best_idx+1, stats['ave_vr'][best_idx], 's', color='tab:green', 
                markersize=15, label=f'Best = {stats["ave_vr"][best_idx]:.3f} km/s', zorder=10)
        plt.xlabel('Realization', fontsize=12)
        plt.ylabel('Average Rupture Velocity (km/s)', fontsize=12)
        plt.title(f'Rupture Velocity (Best: #{best_idx+1})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_spectral_comparison(self, figsize=(14, 6)):
        """Plot Moment Rate Spectrum and Octave-Averaged Spectrum side by side."""
        if self.source_stats is None:
            print("No source statistics available. Run simulation first.")
            return
        
        spectrum = self.source_stats['spectrum']
        octave = self.source_stats['spectrum_octave']
        plt.figure(figsize=figsize)
        
        # Subplot 1: Full Spectrum
        plt.subplot(1, 2, 1)
        plt.loglog(spectrum['freq'], spectrum['moment_rate_synth'], color='tab:blue', lw=2.5, label='Synthetic Model')
        plt.loglog(spectrum['freq'], spectrum['moment_rate_dcf'], color='tab:red', lw=2.5, ls='--', label='DCF Target')
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Moment Rate Spectrum', fontsize=12)
        plt.title('Moment Rate Spectrum', fontsize=13, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, which='both', alpha=0.3)
        
        # Subplot 2: Octave-Averaged
        plt.subplot(1, 2, 2)
        plt.semilogx(octave['freq_center'], octave['logmean_synth'], 'o-', color='tab:blue', lw=2.5, markersize=8, label='Synthetic (log-mean)')
        plt.semilogx(octave['freq_center'], octave['logmean_dcf'], 's--', color='tab:red', lw=2.5, markersize=8, label='DCF Target (log-mean)')
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Log Mean Amplitude', fontsize=12)
        plt.title('Octave-Averaged Spectrum', fontsize=13, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_source_time_function(self, figsize=(10, 6),xlim=None):
        """Plot Source Time Function (STF)."""
        if self.source_stats is None:
            print("No source statistics available. Run simulation first.")
            return
        
        stf = self.source_stats['stf_time']
        plt.figure(figsize=figsize)
        plt.plot(stf['time'], stf['stf'], color='black', lw=1.5, label='STF')
        plt.fill_between(stf['time'], 0, stf['stf'], alpha=0.3, color='tab:cyan')
        max_idx = np.argmax(stf['stf'])
        plt.plot(stf['time'][max_idx], stf['stf'][max_idx], 'o', color='tab:red', markersize=12, label=f'Peak at t={stf["time"][max_idx]:.2f} s')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Moment Rate', fontsize=12)
        plt.title('Source Time Function (STF)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        if xlim is not None:
            plt.xlim(xlim)
        plt.tight_layout()
        plt.show()