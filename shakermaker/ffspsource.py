import os
import tempfile
import shutil
from typing import Optional, Dict, List
import numpy as np
from .crustmodel import CrustModel
from .ffsp import write_ffsp_inp, write_velocity_file, run_ffsp, parse_all_realizations, parse_best_realization, parse_statistical_results
import matplotlib.pyplot as plt
from .ffsp.ffsp_mpi_runner import run_ffsp_mpi
        
# Finite Fault Stochastic Process (FFSP) source model
class FFSPSource:
    """Finite Fault Stochastic Process source model (Liu & Archuleta)"""
    
    def __init__(self,
                 id_sf_type: int, freq_min: float, freq_max: float,
                 fault_length: float,fault_width: float,
                 x_hypc: float, y_hypc: float, depth_hypc: float,
                 xref_hypc: float,yref_hypc: float,
                 magnitude: float,  fc_main_1: float, fc_main_2: float,
                 rv_avg: float,
                 ratio_rise: float,
                 strike: float, dip: float, rake: float,
                 pdip_max: float,  prake_max: float,
                 nsubx: int,  nsuby: int,
                 nb_taper_trbl: List[int],
                 seeds: List[int],
                 id_ran1: int,  id_ran2: int,
                 angle_north_to_x: float,
                 is_moment: int,
                 crust_model: CrustModel,
                 output_name: str = "FFSP_OUTPUT",
                 work_dir: Optional[str] = None,
                 cleanup: bool = True,
                 verbose: bool = True):
        
        if not isinstance(crust_model, CrustModel):
            raise TypeError("crust_model must be a CrustModel instance")

        self.params = {
            'id_sf_type': id_sf_type,'freq_min': freq_min,'freq_max': freq_max,
            'fault_length': fault_length,'fault_width': fault_width,
            'x_hypc': x_hypc,'y_hypc': y_hypc, 'depth_hypc': depth_hypc,
            'xref_hypc': xref_hypc,'yref_hypc': yref_hypc,
            'magnitude': magnitude, 'fc_main_1': fc_main_1, 'fc_main_2': fc_main_2,
            'rv_avg': rv_avg,
            'ratio_rise': ratio_rise,
            'strike': strike, 'dip': dip, 'rake': rake,
            'pdip_max': pdip_max, 'prake_max': prake_max,
            'nsubx': nsubx,  'nsuby': nsuby,
            'nb_taper_trbl': nb_taper_trbl,
            'seeds': seeds,
            'id_ran1': id_ran1,  'id_ran2': id_ran2,
            'velocity_file': 'velocity.vel',
            'angle_north_to_x': angle_north_to_x,
            'is_moment': is_moment,
            'output_name': output_name,    }
        
        self.crust_model = crust_model
        self.output_name = output_name
        self.work_dir = work_dir
        self.cleanup = cleanup
        self.verbose = verbose
        
        self.all_realizations = None
        self.best_realization = None
        self.source_stats = None
        self.subfaults = None
        self._temp_dir = None
    
    def run(self) -> Dict:
        """Run FFSP to generate fault realizations"""
        
        # Configurar directorio de trabajo
        if self.work_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix='ffsp_')
            work_dir = self._temp_dir
        else:
            work_dir = self.work_dir
            os.makedirs(work_dir, exist_ok=True)
            self._cleanup_old_outputs(work_dir)
        
        if not self.cleanup:
            print(f"--- Working directory: {work_dir}")
        
        try:
            # DETECTAR MPI PRIMERO
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                use_mpi = comm.Get_size() > 1
                rank = comm.Get_rank() if use_mpi else 0
            except ImportError:
                use_mpi = False
                rank = 0
            
            # ESCRIBIR ARCHIVOS: Solo si NO hay MPI
            if not use_mpi:
                write_ffsp_inp(self.params, os.path.join(work_dir, 'ffsp.inp'))
                write_velocity_file(self.crust_model, os.path.join(work_dir, 'velocity.vel'))
            
            if self.verbose and rank == 0:
                print("Running FFSP...")
            
            # EJECUTAR FFSP
            if use_mpi:
                # MPI paralelo
                from .ffsp.ffsp_mpi_runner import run_ffsp_mpi
                run_ffsp_mpi(self.params, self.crust_model, work_dir, verbose=self.verbose)
                
                # Parse resultados
                if rank == 0:
                    # Rank 0 usa directorio consolidado
                    consolidated_dir = os.path.join(work_dir, 'consolidated')
                    self._parse_output(consolidated_dir)
                else:
                    # Otros ranks: dummy parse 
                    self.realizations = []
                    self.best_realization = None
                    self.subfaults = None
                    self.all_realizations = {}
                    self.stats = {}
            else:
                # Serial
                run_ffsp(work_dir, verbose=self.verbose)
                self._parse_output(work_dir)
            
            return self.subfaults
            
        finally:
            # Cleanup solo si es temporal
            if self.cleanup and self._temp_dir is not None:
                shutil.rmtree(self._temp_dir)
                if self.verbose:
                    print("\nCleaned up temporary files")


    # Remove old FFSP output files from work directory
    def _cleanup_old_outputs(self, work_dir: str):
        """Remove old FFSP output files from work directory"""
        if not os.path.exists(work_dir):
            return
        
        for item in os.listdir(work_dir):
            if item.startswith(f"{self.output_name}."):
                try:
                    path = os.path.join(work_dir, item)
                    if os.path.isfile(path):
                        os.remove(path)
                except:
                    pass
        
        # Limpiar subdirectorios de MPI previos
        for item in os.listdir(work_dir):
            if item.startswith('rank_') or item == 'consolidated':
                try:
                    path = os.path.join(work_dir, item)
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                except:
                    pass

    def _parse_output(self, work_dir: str):
        """Parse FFSP output files"""
        
        # Parse all realizations
        self.all_realizations = parse_all_realizations(self.output_name, work_dir )
        
        # Parse best realization
        self.best_realization = parse_best_realization(self.output_name, work_dir )
        
        # Parse statistical results
        self.source_stats = parse_statistical_results(work_dir)
        
        # Set best as active subfaults
        if self.best_realization is not None:
            self.subfaults = self.best_realization
            self.active_realization = 'best'
        
        # Print summary
        if self.verbose and self.all_realizations:
            n = self.all_realizations['n_realizations']
            print(f"\nFound {n} realization(s) - Active: {self.active_realization}")
            if self.best_realization:
                slip_min = self.best_realization['slip'].min()
                slip_max = self.best_realization['slip'].max()
                print(f"  Slip: {slip_min:.3f} - {slip_max:.3f} m")

    # Get specific realization by index (0-based)
    def get_realization(self, index: int) -> Dict:
        """Get specific realization by index (0-based)"""
        
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

    # Set active realization for plotting
    def set_active_realization(self, index: int):
        """Set active realization for plotting"""
        self.subfaults = self.get_realization(index)
        self.active_realization = index

    # Get currently active subfault data
    def get_subfaults(self) -> Dict:
        """Get currently active subfault data"""
        if self.subfaults is None:
            raise RuntimeError("No active realization. Call set_active_realization() first.")
        return self.subfaults 
    
# PLOTS (revisar si la dejamos aqui)

    # Plot histogram of field across all realizations
    def plot_histogram(self, field='slip', bins=50, figsize=(7, 5)):
        valid_fields = ['x', 'y', 'z', 'slip', 'rupture_time', 'rise_time', 'peak_time', 'strike', 'dip', 'rake']
        if field not in valid_fields:
            raise ValueError(f"field must be one of {valid_fields}, got '{field}'")    
        plt.figure(figsize=figsize)    
        # Plot all realizations
        for i in range(self.all_realizations['n_realizations']):
            var = self.all_realizations[field][:, i]
            plt.hist(var, bins=bins, alpha=0.4, label=f'Rlz{i+1}')    
        # Plot best realization
        var_best = self.best_realization[field]
        plt.hist(var_best, bins=bins, histtype='step', color='red', linewidth=2, label='Best')
        
        # Labels
        field_labels = {
            'x': 'North (m)',
            'y': 'East (m)',
            'z': 'Depth (m)',
            'slip': 'Slip (m)',
            'rupture_time': 'Rupture Time (s)',
            'rise_time': 'Rise Time (s)',
            'peak_time': 'Peak Time (s)',
            'strike': 'Strike (°)',
            'dip': 'Dip (°)',
            'rake': 'Rake (°)'}
        
        plt.xlabel(field_labels[field])
        plt.ylabel('Frequency')
        plt.title(f'{field_labels[field]} - Multiple Realizations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Plot spatial distribution of subfault parameters
    def plot_spacial_distribution(self, figsize=(10, 8) , field='rise_time', cmap='coolwarm', show_contours=True, contour_field='rupture_time', show_hypocenter=True):

        nx = self.params['nsubx']
        ny = self.params['nsuby']
        lx = self.params['fault_length']
        ly = self.params['fault_width']
        dx = lx / nx
        dy = ly / ny
        cxp = self.params['x_hypc']
        cyp = self.params['y_hypc']
        
        # Validate fields
        valid_fields = ['slip', 'rupture_time', 'rise_time', 'peak_time', 'strike', 'dip', 'rake']
        if field not in valid_fields:
            raise ValueError(f"field must be one of {valid_fields}, got '{field}'")
        if contour_field not in valid_fields:
            raise ValueError(f"contour_field must be one of {valid_fields}, got '{contour_field}'")

        # Prepare data
        field_data = np.transpose(self.subfaults[field].reshape(nx, ny))
        contour_data = np.transpose(self.subfaults[contour_field].reshape(nx, ny))
        x = np.linspace(-lx/2, lx/2, nx)
        y = np.linspace(0, ly, ny)
        X, Y = np.meshgrid(x, y)

        # Labels
        field_labels = {
            'slip': 'Slip [m]',
            'rupture_time': 'Rupture Time [s]',
            'rise_time': 'Rise Time [s]',
            'peak_time': 'Peak Time [s]',
            'strike': 'Strike [°]',
            'dip': 'Dip [°]',
            'rake': 'Rake [°]',        
        }

        # Plot
        plt.figure(figsize=figsize)
        plt.imshow(field_data[::-1], cmap=cmap, extent=(-lx/2-dx/2, lx/2+dx/2, -dy/2, ly+dy/2), interpolation='nearest')
        plt.colorbar(label=field_labels[field], shrink=ly/lx)

        # Contours
        if show_contours:
            contours = plt.contour(X, Y, contour_data, 8, colors='blue', linewidths=1.5)
            plt.clabel(contours, fontsize=10, fmt='%2.1f', inline=1)                
        # Hypocenter
        if show_hypocenter:
            plt.scatter(cxp-lx/2, cyp, c='red', s=300, marker='*', edgecolors='white', linewidth=2, label='Hypocenter', zorder=10)
            plt.legend(loc='upper right')

        plt.xlabel('Along Strike [km]')
        plt.ylabel('Down Dip [km]')
        
        # Title
        if show_contours:
            plt.title(f'{field_labels[field]} Distribution (contours: {field_labels[contour_field]})', fontsize=14, fontweight='bold')
        else:
            plt.title(f'{field_labels[field]} Distribution', fontsize=14, fontweight='bold')
        
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()

    # PPlot PDF and Spectral Error side by side
    def plot_quality_metrics(self , figsize: tuple = (14, 5)):
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
        plt.axhline(stats['pdf'][best_idx], color='tab:red', ls='--', lw=2, label=f'Best = {stats["pdf"][best_idx]:.3f}')
        plt.xlabel('Realization', fontsize=12)
        plt.ylabel('PDF', fontsize=12)
        plt.title(f'PDF Quality Metric (Best: #{best_idx+1})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        # Subplot 2: Spectral Error
        plt.subplot(1, 2, 2)
        colors = ['tab:green' if i == best_idx else 'tab:orange' for i in range(n)]
        plt.bar(idx, stats['err_spectra'], color=colors, edgecolor='black', alpha=0.7)
        plt.axhline(stats['err_spectra'][best_idx], color='tab:red', ls='--', lw=2, label=f'Best = {stats["err_spectra"][best_idx]:.4f}')
        plt.xlabel('Realization', fontsize=12)
        plt.ylabel('Spectral Error (RMS)', fontsize=12)
        plt.title(f'Spectral Error (Best: #{best_idx+1})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Plot Rise Time, Peak Time, and Rupture Velocity
    def plot_temporal_metrics(self, figsize: tuple = (15, 5)):
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
        plt.plot(best_idx+1, stats['ave_tr'][best_idx], 's', color='tab:green', markersize=15, label=f'Best = {stats["ave_tr"][best_idx]:.3f} s', zorder=10)
        plt.xlabel('Realization', fontsize=12)
        plt.ylabel('Average Rise Time (s)', fontsize=12)
        plt.title(f'Rise Time (Best: #{best_idx+1})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        # Subplot 2: Peak Time
        plt.subplot(1, 3, 2)
        plt.plot(idx, stats['ave_tp'], 'o-', color='tab:orange', lw=2, markersize=6)
        plt.plot(best_idx+1, stats['ave_tp'][best_idx], 's', color='tab:green', markersize=15, label=f'Best = {stats["ave_tp"][best_idx]:.3f} s', zorder=10)
        plt.xlabel('Realization', fontsize=12)
        plt.ylabel('Average Peak Time (s)', fontsize=12)
        plt.title(f'Peak Time (Best: #{best_idx+1})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        # Subplot 3: Rupture Velocity
        plt.subplot(1, 3, 3)
        plt.plot(idx, stats['ave_vr'], 'o-', color='tab:cyan', lw=2, markersize=6)
        plt.plot(best_idx+1, stats['ave_vr'][best_idx], 's', color='tab:green', markersize=15, label=f'Best = {stats["ave_vr"][best_idx]:.3f} km/s', zorder=10)
        plt.xlabel('Realization', fontsize=12)
        plt.ylabel('Average Rupture Velocity (km/s)', fontsize=12)
        plt.title(f'Rupture Velocity (Best: #{best_idx+1})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Plot Moment Rate Spectrum and Octave-Averaged Spectrum side by side
    def plot_spectral_comparison(self, figsize: tuple = (14, 6)):       
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
    
    # Plot Source Time Function (STF)
    def plot_source_time_function(self, figsize: tuple = (10, 6)):
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
        plt.tight_layout()
        plt.show()