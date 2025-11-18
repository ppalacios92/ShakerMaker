import os
import subprocess
from typing import Tuple


# Locate FFSP executable
def find_ffsp_executable() -> str:

    ffsp_dir = os.path.dirname(os.path.abspath(__file__))
    ffsp_exec = os.path.join(ffsp_dir, 'ffsp_dcf_v2') # ffsp_dcf_v2 name in the makefile
    if not os.path.exists(ffsp_exec):
        raise FileNotFoundError(f"FFSP not found: {ffsp_exec}\nRun 'make' in {ffsp_dir}")
    if not os.access(ffsp_exec, os.X_OK):
        raise PermissionError(f"Not executable: {ffsp_exec}\nRun: chmod +x {ffsp_exec}") # Problemas de lectura
    return ffsp_exec

# Execute FFSP in working directory
def run_ffsp(work_dir: str, verbose: bool = True) -> Tuple[int, str, str]:    

    ffsp_exec = find_ffsp_executable()    
    if not os.path.exists(os.path.join(work_dir, 'ffsp.inp')):
        raise FileNotFoundError(f"Missing: {work_dir}/ffsp.inp")
    
    result = subprocess.run([ffsp_exec], cwd=work_dir, capture_output=True, text=True)
    
    if verbose and result.stdout:
        print(result.stdout)    
    if result.returncode != 0:
        raise RuntimeError(f"FFSP failed (code {result.returncode}): {result.stderr}")
    
    return result.returncode, result.stdout, result.stderr