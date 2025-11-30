import os
import shutil
import time
from mpi4py import MPI
from .ffsp_io import write_velocity_file, write_ffsp_inp
from .ffsp_runner import run_ffsp

def run_ffsp_mpi(params, crust_model, work_dir, verbose=False):
    """Run FFSP in parallel using MPI"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Distribuir modelos
    total_models = params['id_ran2'] - params['id_ran1'] + 1
    models_per_rank = total_models // size
    remainder = total_models % size
    
    if rank < remainder:
        start = rank * (models_per_rank + 1) + params['id_ran1']
        end = start + models_per_rank
    else:
        start = rank * models_per_rank + remainder + params['id_ran1']
        end = start + models_per_rank - 1
    
    if rank == 0 and verbose:
        print(f"MPI: {total_models} models across {size} processes")
    
    # Setup rank directory - SERIAL
    for r in range(size):
        if rank == r:
            rank_work_dir = os.path.join(work_dir, f'rank_{rank:04d}')
            os.makedirs(rank_work_dir, exist_ok=True)
        # Esperar a que termine
        comm.Barrier()  
    
    rank_work_dir = os.path.join(work_dir, f'rank_{rank:04d}')
    
    # Write config files
    params_rank = params.copy()
    params_rank['id_ran1'] = start
    params_rank['id_ran2'] = end
    params_rank['velocity_file'] = 'velocity.vel'
    
    # Escribir archivos - SERIAL
    for r in range(size):
        if rank == r:
            write_velocity_file(crust_model, os.path.join(rank_work_dir, 'velocity.vel'))
            write_ffsp_inp(params_rank, os.path.join(rank_work_dir, 'ffsp.inp'))
        # Esperar a que termine
        comm.Barrier() 
    
    # Small delay for filesystem
    time.sleep(0.1)
    comm.Barrier()

    # Run FFSP
    run_ffsp(rank_work_dir, verbose=(rank == 0 and verbose))
    
    # Consolidate (rank 0 only)
    comm.Barrier()
    if rank == 0:
        consolidate_results(work_dir, size, total_models)
    comm.Barrier()

def consolidate_results(work_dir, size, total_models):
    """Consolidate all results"""
    consolidated_dir = os.path.join(work_dir, 'consolidated')
    os.makedirs(consolidated_dir, exist_ok=True)
    
    print("Consolidating results...")
    
    # Copy ALL models with sequential numbering
    model_num = 1
    for r in range(size):
        rank_dir = os.path.join(work_dir, f'rank_{r:04d}')
        if not os.path.exists(rank_dir):
            continue
            
        # Copy FFSP_OUTPUT.XXX
        for f in sorted(os.listdir(rank_dir)):
            if f.startswith('FFSP_OUTPUT.') and f[-3:].isdigit():
                shutil.copy2(
                    os.path.join(rank_dir, f),
                    os.path.join(consolidated_dir, f'FFSP_OUTPUT.{model_num:03d}')
                )
                model_num += 1
        
        # Copy source_model.XXX as-is
        for f in os.listdir(rank_dir):
            if f.startswith('source_model.') and f[-3:].isdigit():
                shutil.copy2(
                    os.path.join(rank_dir, f),
                    os.path.join(consolidated_dir, f)
                )
    
    # Find best model
    all_scores = []
    for r in range(size):
        score_file = os.path.join(work_dir, f'rank_{r:04d}', 'source_model.score')
        if not os.path.exists(score_file):
            continue
        
        with open(score_file) as f:
            lines = f.readlines()[2:]  # Skip header
        
        for i in range(0, len(lines)-1, 2):
            values = list(map(float, lines[i+1].split()))
            all_scores.append({
                'file': lines[i].strip(),
                'pdf': values[-1],
                'rank': r,
                'values': values
            })
    
    best = min(all_scores, key=lambda x: x['pdf'])
    print(f"Best: {best['file']} (PDF={best['pdf']:.6f})")
    
    # Copy best .bst
    shutil.copy2(
        os.path.join(work_dir, f"rank_{best['rank']:04d}", 'FFSP_OUTPUT.bst'),
        os.path.join(consolidated_dir, 'FFSP_OUTPUT.bst')
    )
    
    # Write consolidated score
    with open(os.path.join(consolidated_dir, 'source_model.score'), 'w') as f:
        f.write(f"{total_models}\n")
        f.write("Target: average Risetime= 0.0 average peaktime= 0.0\n")
        for s in sorted(all_scores, key=lambda x: x['file']):
            f.write(f"{s['file']}\n")
            f.write(' '.join([f"{v:15.5e}" for v in s['values']]) + '\n')
    
    # Copy list file
    src_list = os.path.join(work_dir, 'rank_0000', 'source_model.list')
    if os.path.exists(src_list):
        with open(src_list) as f:
            lines = f.readlines()[:3]
        with open(os.path.join(consolidated_dir, 'source_model.list'), 'w') as f:
            f.writelines(lines)
            f.write('FFSP_OUTPUT.bst\n')
    
    # Copy stats from best rank
    for fname in ['calsvf.dat', 'calsvf_tim.dat', 'logsvf.dat']:
        src = os.path.join(work_dir, f"rank_{best['rank']:04d}", fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(consolidated_dir, fname))
    
    print(f"Consolidated {model_num-1} models")