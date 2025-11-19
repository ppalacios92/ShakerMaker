import os
import numpy as np
from typing import List, Dict, Optional

# Write FFSP input file
def write_ffsp_inp(params: dict, filename: str = 'ffsp.inp') -> None:
    
    with open(filename, 'w') as f:
        f.write(f"{params['id_sf_type']} {params['freq_min']} {params['freq_max']}\n")                                                      #l1
        f.write(f"{params['fault_length']} {params['fault_width']}\n")                                                                      #l2                               
        f.write(f"{params['x_hypc']} {params['y_hypc']} {params['depth_hypc']}\n")                                                          #l3             
        f.write(f"{params['xref_hypc']} {params['yref_hypc']}\n")                                                                           #l4                       
        f.write(f"{params['magnitude']} {params['fc_main_1']} {params['fc_main_2']} {params['rv_avg']}\n")                                  #l5         
        f.write(f"{params['ratio_rise']}\n")                                                                                                #l6                                
        f.write(f"{params['strike']} {params['dip']} {params['rake']}\n")                                                                   #l7             
        f.write(f"{params['pdip_max']} {params['prake_max']}\n")                                                                            #l8          
        f.write(f"{params['nsubx']} {params['nsuby']}\n")                                                                                   #l9               
        f.write(f"{params['nb_taper_trbl'][0]} {params['nb_taper_trbl'][1]} {params['nb_taper_trbl'][2]} {params['nb_taper_trbl'][3]}\n")   #l10
        f.write(f"{params['seeds'][0]} {params['seeds'][1]} {params['seeds'][2]}\n")                                                        #l11               
        f.write(f"{params['id_ran1']} {params['id_ran2']}\n")                                                                               #l12         
        f.write(f"{params['velocity_file']}\n")                                                                                             #l13        
        f.write(f"{params['angle_north_to_x']}\n")                                                                                          #l14                                                
        f.write(f"{params['is_moment']}\n")                                                                                                 #l15                
        f.write(f"{params['output_name']}\n")                                                                                               #l16        

# Write velocity model file for FFSP
def write_velocity_file(crust_model, filename: str = 'velocity.vel') -> None:
    with open(filename, 'w') as f:
        f.write(f"{crust_model.nlayers} 2.0\n") # 2.0 es freq_ref pero nunca se lo uso, consultar
        # Buscar freq_ref en todo el código: grep -n "freq_ref" *.f90 *.f /  grep -B 5 -A 5 "freq_ref" *.f90
        for i in range(crust_model.nlayers):
            f.write(f"{crust_model.a[i]:.5f}   {crust_model.b[i]:.6f}   "
                   f"{crust_model.rho[i]:.4f}  {crust_model.d[i]:.5f}   "
                   f"{crust_model.qa[i]:.5f}   {crust_model.qb[i]:.5f}\n")

# Find FFSP output files matching base_name.XXX pattern
def find_ffsp_output_files(base_name: str, parent_dir: str) -> List[str]:

    if not os.path.exists(parent_dir):
        return []    
    output_files = []
    for item in os.listdir(parent_dir):
        if item.startswith(base_name + "."):
            suffix = item[len(base_name)+1:]
            if len(suffix) == 3 and suffix.isdigit():
                full_path = os.path.join(parent_dir, item)
                if os.path.isfile(full_path):
                    output_files.append(full_path)
    
    return sorted(output_files)

# Parse all FFSP realizations base_name.XXX pattern and return as structured dict
def parse_all_realizations(base_name: str, work_dir: str) -> Dict[str, np.ndarray]:

    if not os.path.exists(work_dir):
        return {}
    # Find all base_name.XXX files
    output_files = []
    for item in os.listdir(work_dir):
        if item.startswith(base_name + "."):
            suffix = item[len(base_name)+1:]
            if len(suffix) == 3 and suffix.isdigit():
                full_path = os.path.join(work_dir, item)
                if os.path.isfile(full_path):
                    output_files.append(full_path)
    
    output_files = sorted(output_files)
    if not output_files:
        return {}    
    # Parse first file to get dimensions
    with open(output_files[0]) as fid:
        header = fid.readline().split()
        nseg = int(header[0])
        npts = int(header[1])    
    n_realizations = len(output_files)
    
    # IFormat output
    result = {
        'n_realizations': n_realizations,
        'nseg': nseg,
        'npts': npts,
        'x': np.zeros((npts, n_realizations)),
        'y': np.zeros((npts, n_realizations)),
        'z': np.zeros((npts, n_realizations)),
        'slip': np.zeros((npts, n_realizations)),
        'rupture_time': np.zeros((npts, n_realizations)),
        'rise_time': np.zeros((npts, n_realizations)),
        'peak_time': np.zeros((npts, n_realizations)),
        'strike': np.zeros((npts, n_realizations)),
        'dip': np.zeros((npts, n_realizations)),
        'rake': np.zeros((npts, n_realizations))
    }
    # Parse all files
    for i, filepath in enumerate(output_files):
        data = np.loadtxt(filepath, skiprows=1)
        result['x'][:, i] = data[:, 0]
        result['y'][:, i] = data[:, 1]
        result['z'][:, i] = data[:, 2]
        result['slip'][:, i] = data[:, 3]
        result['rupture_time'][:, i] = data[:, 4]
        result['rise_time'][:, i] = data[:, 5]
        result['peak_time'][:, i] = data[:, 6]
        result['strike'][:, i] = data[:, 7]
        result['dip'][:, i] = data[:, 8]
        result['rake'][:, i] = data[:, 9]
    
    return result

# Parse best realization selected by FFSP (base_name.bst)
def parse_best_realization(base_name: str, work_dir: str) -> Optional[Dict[str, np.ndarray]]:

    bst_file = os.path.join(work_dir, f"{base_name}.bst")
    if not os.path.exists(bst_file):
        return None
    with open(bst_file) as fid:
        header = fid.readline().split()
        nseg = int(header[0])
        npts = int(header[1])
    data = np.loadtxt(bst_file, skiprows=1)
    return {
        'nseg': nseg,
        'npts': npts,
        'x': data[:, 0],
        'y': data[:, 1],
        'z': data[:, 2],
        'slip': data[:, 3],
        'rupture_time': data[:, 4],
        'rise_time': data[:, 5],
        'peak_time': data[:, 6],
        'strike': data[:, 7],
        'dip': data[:, 8],
        'rake': data[:, 9]
    }

# Parse all statistical output files from FFSP_DCF_v2
def parse_statistical_results(work_dir: str) -> Optional[Dict]:
    results = {}
    
    # 1. Parse source_model.score
    source_file = os.path.join(work_dir, "source_model.score")
    if not os.path.exists(source_file):
        return None    
    with open(source_file) as f:
        n_realizations = int(f.readline().strip())
        f.readline()          
        names, ave_tr, ave_tp, ave_vr, err_spectra, pdf = [], [], [], [], [], []        
        for _ in range(n_realizations):
            name = f.readline().strip()
            values = list(map(float, f.readline().split()))
            names.append(name)
            ave_tr.append(values[0])
            ave_tp.append(values[1])
            ave_vr.append(values[2])
            err_spectra.append(values[3])
            pdf.append(values[4])    
    results['source_score'] = {
            'n_realizations': n_realizations,
            'names': names,
            'ave_tr': np.array(ave_tr),
            'ave_tp': np.array(ave_tp),
            'ave_vr': np.array(ave_vr),
            'err_spectra': np.array(err_spectra),
            'pdf': np.array(pdf)       }    

    # 2. Parse source_model.list
    list_file = os.path.join(work_dir, "source_model.list")
    if not os.path.exists(list_file):
        return None
    with open(list_file) as f:
        line1 = list(map(float, f.readline().split()))
        line2 = list(map(float, f.readline().split()))
        best_file = f.readline().strip()
    
    results['source_list'] = {
        'id': int(line1[0]),
        'nsubx': int(line1[1]),
        'nsuby': int(line1[2]),
        'dx': line1[3],
        'dy': line1[4],
        'x_hypc': line1[5],
        'y_hypc': line1[6],
        'xref_hypc': line2[0],
        'yref_hypc': line2[1],
        'angle_north': line2[2],
        'best_file': best_file    }
    
    # 3. Parse calsvf.dat (frequency domain spectrum)
    calsvf_file = os.path.join(work_dir, "calsvf.dat")
    if not os.path.exists(calsvf_file):
        return None
    data = np.loadtxt(calsvf_file, skiprows=1)
    
    results['spectrum'] = {
        'freq': data[:, 0],
        'moment_rate_synth': data[:, 1],
        'moment_rate_dcf': data[:, 2]    }
    
    # 4. Parse calsvf_tim.dat (time domain STF)
    stf_file = os.path.join(work_dir, "calsvf_tim.dat")
    if not os.path.exists(stf_file):
        return None
    data = np.loadtxt(stf_file, skiprows=1)
    
    results['stf_time'] = {
        'time': data[:, 0],
        'stf': data[:, 1]    }
    
    # 5. Parse logsvf.dat (octave-averaged spectrum)
    logsvf_file = os.path.join(work_dir, "logsvf.dat")
    if not os.path.exists(logsvf_file):
        return None
    data = np.loadtxt(logsvf_file, skiprows=1)
    
    results['spectrum_octave'] = {
        'freq_center': data[:, 0],
        'logmean_synth': data[:, 1],
        'logmean_dcf': data[:, 2]    }
    
    return results