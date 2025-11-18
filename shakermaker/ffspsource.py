import os
import tempfile
import shutil
from typing import Optional, Dict, List
import numpy as np
from .crustmodel import CrustModel
from .ffsp import (
    write_ffsp_inp, 
    write_velocity_file, 
    run_ffsp,
    parse_all_realizations,
    parse_best_realization,
)

# Finite Fault Stochastic Process (FFSP) source model
class FFSPSource:
    """    
    Finite Fault Stochastic Process (FFSP) source model
    Generates stochastic slip distributions using FFSP code by Liu & Archuleta.
    """
    
    def __init__(self,
                 id_sf_type: int,
                 freq_min: float,
                 freq_max: float,
                 fault_length: float,
                 fault_width: float,
                 x_hypc: float,
                 y_hypc: float,
                 depth_hypc: float,
                 xref_hypc: float,
                 yref_hypc: float,
                 magnitude: float,
                 fc_main_1: float,
                 fc_main_2: float,
                 rv_avg: float,
                 ratio_rise: float,
                 strike: float,
                 dip: float,
                 rake: float,
                 pdip_max: float,
                 prake_max: float,
                 nsubx: int,
                 nsuby: int,
                 nb_taper_trbl: List[int],
                 seeds: List[int],
                 id_ran1: int,
                 id_ran2: int,
                 angle_north_to_x: float,
                 is_moment: int,
                 crust_model: CrustModel,
                 output_name: str = "FFSP_OUTPUT",
                 work_dir: Optional[str] = None,
                 cleanup: bool = True,
                 verbose: bool = True):
        
        if not isinstance(crust_model, CrustModel):
            raise TypeError("crust_model must be a CrustModel instance")
        
        if len(nb_taper_trbl) != 4:
            raise ValueError("nb_taper_trbl must have exactly 4 elements")
        if len(seeds) != 3:
            raise ValueError("seeds must have exactly 3 elements")
        
        self.params = {
            'id_sf_type': id_sf_type,
            'freq_min': freq_min,
            'freq_max': freq_max,
            'fault_length': fault_length,
            'fault_width': fault_width,
            'x_hypc': x_hypc,
            'y_hypc': y_hypc,
            'depth_hypc': depth_hypc,
            'xref_hypc': xref_hypc,
            'yref_hypc': yref_hypc,
            'magnitude': magnitude,
            'fc_main_1': fc_main_1,
            'fc_main_2': fc_main_2,
            'rv_avg': rv_avg,
            'ratio_rise': ratio_rise,
            'strike': strike,
            'dip': dip,
            'rake': rake,
            'pdip_max': pdip_max,
            'prake_max': prake_max,
            'nsubx': nsubx,
            'nsuby': nsuby,
            'nb_taper_trbl': nb_taper_trbl,
            'seeds': seeds,
            'id_ran1': id_ran1,
            'id_ran2': id_ran2,
            'velocity_file': 'velocity.vel',
            'angle_north_to_x': angle_north_to_x,
            'is_moment': is_moment,
            'output_name': output_name,
        }
        
        self.crust_model = crust_model
        self.output_name = output_name
        self.work_dir = work_dir
        self.cleanup = cleanup
        self.verbose = verbose
        
        self.all_realizations: Optional[Dict] = None
        self.best_realization: Optional[Dict] = None
        self.subfaults: Optional[Dict] = None
        self._temp_dir: Optional[str] = None
    
    # Execute FFSP simulation and return best/first realization
    def run(self) -> Dict:           
        if self.work_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix='ffsp_')
            work_dir = self._temp_dir
        else:
            work_dir = self.work_dir
            os.makedirs(work_dir, exist_ok=True)
            self._cleanup_old_outputs(work_dir)

        if self.verbose:
            print(f"Working directory: {work_dir}")
        
        try:
            inp_file = os.path.join(work_dir, 'ffsp.inp')
            vel_file = os.path.join(work_dir, 'velocity.vel')
            
            if self.verbose:
                print("Generating input files...")
            
            write_ffsp_inp(self.params, inp_file)
            write_velocity_file(self.crust_model, vel_file)
            
            if self.verbose:
                print("Running FFSP...")
            
            run_ffsp(work_dir, verbose=self.verbose)
            
            self.all_realizations = parse_all_realizations(self.output_name, work_dir)
            
            if not self.all_realizations:
                raise RuntimeError(f"No FFSP output files found in {work_dir}")
            
            self.best_realization = parse_best_realization(self.output_name, work_dir)
            
            if self.best_realization:
                self.subfaults = self.best_realization
                active_source = "best"
            else:
                self.subfaults = self.get_realization(0)
                active_source = "first"
            
            if self.verbose:
                n = self.all_realizations['n_realizations']
                print(f"\nFound {n} realization(s)")
                print(f"Subfaults: {self.all_realizations['npts']}")
                print(f"Active: {active_source}")
                print(f"  Slip: {self.subfaults['slip'].min():.3f} - {self.subfaults['slip'].max():.3f} m")
                print(f"  Rupture time: {self.subfaults['rupture_time'].min():.2f} - {self.subfaults['rupture_time'].max():.2f} s")
            
            return self.subfaults
            
        finally:
            if self.cleanup and self._temp_dir is not None:
                shutil.rmtree(self._temp_dir)
                if self.verbose:
                    print("\nCleaned up temporary files")

    # Remove old FFSP output files from work directory
    def _cleanup_old_outputs(self, work_dir: str):
        if not os.path.exists(work_dir):
            return        
        patterns = [f"{self.output_name}.bst", f"{self.output_name}.*" ]
        removed = []
        for item in os.listdir(work_dir):
            # Remove .bst file
            if item == f"{self.output_name}.bst":
                os.remove(os.path.join(work_dir, item))
                removed.append(item)
            # Remove .XXX files
            elif item.startswith(f"{self.output_name}."):
                suffix = item[len(self.output_name)+1:]
                if len(suffix) == 3 and suffix.isdigit():
                    os.remove(os.path.join(work_dir, item))
                    removed.append(item)
        
        if removed and self.verbose:
            print(f"Cleaned {len(removed)} old output file(s)")

    # Get specific realization by index (0-based)
    def get_realization(self, index: int) -> Dict:        
        
        if self.all_realizations is None:
            raise RuntimeError("Must call run() first")
        
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
            'unknown': self.all_realizations['unknown'][:, index],
            'strike': self.all_realizations['strike'][:, index],
            'dip': self.all_realizations['dip'][:, index],
            'rake': self.all_realizations['rake'][:, index],
        }
    

    # Set active realization for plotting
    def set_active_realization(self, index: int):
        self.subfaults = self.get_realization(index)
    # Get currently active subfault data
    def get_subfaults(self) -> Dict:
        if self.subfaults is None:
            raise RuntimeError("Must call run() first")
        return self.subfaults
    # Plot slip distribution of active realization
    def plot_slip_distribution(self, figsize=(10, 8), cmap='coolwarm'):

        if self.subfaults is None:
            raise RuntimeError("Must call run() first")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")
        
        nx = self.params['nsubx']
        ny = self.params['nsuby']
        lx = self.params['fault_length']
        ly = self.params['fault_width']
        dx = lx / nx
        dy = ly / ny
        cxp = self.params['x_hypc']
        cyp = self.params['y_hypc']
        
        slip = np.transpose(self.subfaults['slip'].reshape(nx, ny))
        rptm = np.transpose(self.subfaults['rupture_time'].reshape(nx, ny))
        rstm = np.transpose(self.subfaults['rise_time'].reshape(nx, ny))
        
        x = np.linspace(-lx/2, lx/2, nx)
        y = np.linspace(0, ly, ny)
        X, Y = np.meshgrid(x, y)
        
        plt.figure(figsize=figsize)
        plt.imshow(rstm[::-1], cmap=cmap, extent=(-lx/2-dx/2, lx/2+dx/2, -dy/2, ly+dy/2),interpolation='nearest')
        plt.colorbar(label='Rise Time [s]', shrink=ly/lx)
        contours=plt.contour(X, Y, rptm, 8, colors='blue')
        plt.clabel(contours, fontsize=12, fmt='%2.1f', inline=1)
        plt.scatter(cxp-lx/2, cyp, c='red', s=300, marker='*',  edgecolors='white', linewidth=2)
        plt.xlabel('Along Strike [km]')
        plt.ylabel('Down Dip [km]')
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()