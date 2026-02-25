"""
shakermaker_op.py - Optimized ShakerMaker pipeline with precomputed pair-to-slot mapping.

Based on JAA clean implementation. New methods are suffixed with _op to distinguish
from original JAA/PXP methods.

Key improvement over JAA/PXP:
  Stage 0 (gen_pairs_op): builds full pair_to_slot[i_station * nsrc + i_psource] = k
  Stage 1 (compute_gf_op): computes Green's functions for each unique slot k (unchanged logic)
  Stage 2 (run_op):        O(1) lookup via pair_to_slot instead of O(N) linear search

Backward compatible: if pair_to_slot absent in HDF5, falls back to legacy linear search.
"""

import copy
import os
import numpy as np
import logging
import traceback
import h5py
from time import perf_counter

from shakermaker.crustmodel import CrustModel
from shakermaker.faultsource import FaultSource
from shakermaker.stationlist import StationList
from shakermaker.stationlistwriter import StationListWriter
from shakermaker import core

try:
    from mpi4py import MPI
    use_mpi = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
except Exception:
    rank = 0
    nprocs = 1
    use_mpi = False


class ShakerMaker:
    """
    Optimized ShakerMaker pipeline.

    Drop-in replacement for ShakerMaker with three-stage workflow:
      - gen_pairs_op:   Stage 0 - build geometry mapping and unique GF slots
      - compute_gf_op:  Stage 1 - compute Green's functions for unique slots
      - run_op:         Stage 2 - convolve GFs with STF and write output

    Or use run_fast_faster_op(stage='all') to run all stages sequentially.

    :param crust: Crustal model
    :type crust: CrustModel
    :param source: Fault source (FaultSource with PointSources, including FFSP-derived)
    :type source: FaultSource
    :param receivers: Station list (DRMBox, SurfaceGrid, StationList)
    :type receivers: StationList
    """

    def __init__(self, crust, source, receivers):
        assert isinstance(crust, CrustModel), \
            "crust must be an instance of shakermaker.CrustModel"
        assert isinstance(source, FaultSource), \
            "source must be an instance of shakermaker.FaultSource"
        assert isinstance(receivers, StationList), \
            "receivers must be an instance of shakermaker.StationList"

        self._crust = crust
        self._source = source
        self._receivers = receivers
        self._mpi_rank = rank
        self._mpi_nprocs = nprocs
        self._logger = logging.getLogger(__name__)

    # =========================================================================
    # Stage 0: Build geometry mapping
    # =========================================================================

    def gen_pairs_op(self,
                     h5_database_name,
                     delta_h=0.04,
                     delta_v_rec=0.002,
                     delta_v_src=0.2,
                     npairs_max=200000,
                     showProgress=True):
        """
        Stage 0: Identify unique (dh, z_src, z_rec) geometry slots and build
        the full pair_to_slot mapping.

        Saves to h5_database_name.h5:
          - pairs_to_compute[k, 2]  : representative (i_station, i_psource) for slot k
          - dh_of_pairs[k]          : horizontal distance for slot k
          - dv_of_pairs[k]          : vertical distance for slot k
          - zrec_of_pairs[k]        : receiver depth for slot k
          - zsrc_of_pairs[k]        : source depth for slot k
          - pair_to_slot[nsta*nsrc] : mapping (i_station*nsrc + i_psource) -> k  [NEW]
          - delta_h, delta_v_rec, delta_v_src (scalars)
          - nstations, nsources (scalars for validation)

        Only rank 0 runs this stage (geometry computation is fast, serial numpy).

        :param h5_database_name: Output HDF5 path (without .h5 extension)
        :param delta_h: Horizontal distance tolerance (km)
        :param delta_v_rec: Receiver depth tolerance (km)
        :param delta_v_src: Source depth tolerance (km)
        :param npairs_max: Max unique slots (pre-allocated, raise if exceeded)
        :param showProgress: Print progress every 50k pairs
        """
        if rank != 0:
            # Only rank 0 builds the mapping; other ranks wait
            if use_mpi and nprocs > 1:
                comm.Barrier()
            return

        nsources   = self._source.nsources
        nstations  = self._receivers.nstations
        npairs_total = nstations * nsources

        if rank == 0:
            print(f"\n[Stage 0] Building pair mapping: {nstations} stations x {nsources} sources = {npairs_total} pairs")
            print(f"[Stage 0] Tolerances: delta_h={delta_h}, delta_v_rec={delta_v_rec}, delta_v_src={delta_v_src}")
            print(f"[Stage 0] Max unique slots: {npairs_max}")

        t0_start = perf_counter()

        # Pre-allocate slot arrays
        pairs_to_compute = np.empty((npairs_max, 2), dtype=np.int32)
        dh_of_pairs      = np.empty(npairs_max, dtype=np.float64)
        dv_of_pairs      = np.empty(npairs_max, dtype=np.float64)
        zrec_of_pairs    = np.empty(npairs_max, dtype=np.float64)
        zsrc_of_pairs    = np.empty(npairs_max, dtype=np.float64)

        # Full mapping: for every (i_station, i_psource) -> slot index k
        pair_to_slot = np.full(npairs_total, -1, dtype=np.int32)

        n_slots = 0  # number of unique slots found so far

        for i_station, station in enumerate(self._receivers):
            z_rec = station.x[2]
            for i_psource, psource in enumerate(self._source):
                z_src = psource.x[2]

                d  = station.x - psource.x
                dh = np.sqrt(d[0]**2 + d[1]**2)
                dv = abs(d[2])

                flat_idx = i_station * nsources + i_psource

                if n_slots == 0:
                    # First pair always becomes slot 0
                    k = 0
                    pairs_to_compute[0] = [i_station, i_psource]
                    dh_of_pairs[0]   = dh
                    dv_of_pairs[0]   = dv
                    zrec_of_pairs[0] = z_rec
                    zsrc_of_pairs[0] = z_src
                    n_slots = 1
                else:
                    # Vectorized tolerance check against all existing slots
                    not_covered = (
                        (np.abs(dh   - dh_of_pairs[:n_slots])   > delta_h)   |
                        (np.abs(z_src - zsrc_of_pairs[:n_slots]) > delta_v_src) |
                        (np.abs(z_rec - zrec_of_pairs[:n_slots]) > delta_v_rec)
                    )

                    if np.all(not_covered):
                        # New unique geometry -> new slot
                        if n_slots >= npairs_max:
                            raise RuntimeError(
                                f"[Stage 0] Exceeded npairs_max={npairs_max}. "
                                f"Increase npairs_max or coarsen tolerances."
                            )
                        k = n_slots
                        pairs_to_compute[k] = [i_station, i_psource]
                        dh_of_pairs[k]   = dh
                        dv_of_pairs[k]   = dv
                        zrec_of_pairs[k] = z_rec
                        zsrc_of_pairs[k] = z_src
                        n_slots += 1
                    else:
                        # Covered: find the closest matching slot
                        # covered slots have not_covered == False
                        covered_mask = ~not_covered
                        covered_indices = np.where(covered_mask)[0]
                        # Among covered slots, pick the one with smallest L1 distance
                        dist = (np.abs(dh    - dh_of_pairs[covered_indices]) +
                                np.abs(z_src - zsrc_of_pairs[covered_indices]) +
                                np.abs(z_rec - zrec_of_pairs[covered_indices]))
                        k = covered_indices[np.argmin(dist)]

                pair_to_slot[flat_idx] = k

                if showProgress and flat_idx % 50000 == 0 and flat_idx > 0:
                    elapsed = perf_counter() - t0_start
                    eta = elapsed / flat_idx * (npairs_total - flat_idx)
                    print(f"[Stage 0] {flat_idx}/{npairs_total} pairs | {n_slots} slots | "
                          f"elapsed={elapsed:.1f}s ETA={eta:.1f}s")

        # Trim to actual number of slots
        pairs_to_compute = pairs_to_compute[:n_slots]
        dh_of_pairs      = dh_of_pairs[:n_slots]
        dv_of_pairs      = dv_of_pairs[:n_slots]
        zrec_of_pairs    = zrec_of_pairs[:n_slots]
        zsrc_of_pairs    = zsrc_of_pairs[:n_slots]

        elapsed = perf_counter() - t0_start
        reduction_pct = (1 - n_slots / npairs_total) * 100
        print(f"\n[Stage 0] Done in {elapsed:.1f}s")
        print(f"[Stage 0] Unique slots: {n_slots} / {npairs_total} ({reduction_pct:.1f}% reduction)")

        # Validate: every pair must have a valid slot
        assert np.all(pair_to_slot >= 0), \
            "[Stage 0] BUG: some pairs have no assigned slot (pair_to_slot contains -1)"
        assert np.all(pair_to_slot < n_slots), \
            "[Stage 0] BUG: pair_to_slot contains out-of-range slot index"

        # Save to HDF5
        with h5py.File(h5_database_name + '.h5', 'w') as hf:
            hf.create_dataset("pairs_to_compute", data=pairs_to_compute)
            hf.create_dataset("dh_of_pairs",      data=dh_of_pairs)
            hf.create_dataset("dv_of_pairs",      data=dv_of_pairs)
            hf.create_dataset("zrec_of_pairs",    data=zrec_of_pairs)
            hf.create_dataset("zsrc_of_pairs",    data=zsrc_of_pairs)
            hf.create_dataset("pair_to_slot",     data=pair_to_slot)
            hf.create_dataset("delta_h",          data=delta_h)
            hf.create_dataset("delta_v_rec",      data=delta_v_rec)
            hf.create_dataset("delta_v_src",      data=delta_v_src)
            hf.create_dataset("nstations",        data=nstations)
            hf.create_dataset("nsources",         data=nsources)

        print(f"[Stage 0] Saved to {h5_database_name}.h5")

        if use_mpi and nprocs > 1:
            comm.Barrier()

    # =========================================================================
    # Stage 1: Compute Green's functions for unique slots
    # =========================================================================

    def compute_gf_op(self,
                      h5_database_name,
                      dt=0.05,
                      nfft=4096,
                      tb=1000,
                      smth=1,
                      sigma=2,
                      taper=0.9,
                      wc1=1,
                      wc2=2,
                      pmin=0,
                      pmax=1,
                      dk=0.3,
                      nx=1,
                      kc=15.0,
                      verbose=False,
                      debugMPI=False,
                      showProgress=True):
        """
        Stage 1: Compute FK Green's function kernel (tdata) for each unique slot k.

        Reads h5_database_name.h5 (produced by gen_pairs_op).
        Writes tdata_dict group into the same HDF5 file.

        MPI: rank 0 coordinates; worker ranks compute and send tdata to rank 0.

        :param h5_database_name: HDF5 path (without .h5 extension)
        :param dt: Time step (s)
        :param nfft: Number of FFT points
        :param tb: Samples before first arrival
        :param smth: Output densification factor
        :param sigma: Damping factor
        :param taper: Low-pass taper (0-1)
        :param wc1, wc2: Filter corner frequencies
        :param pmin, pmax: Phase velocity bounds
        :param dk: Wavenumber sampling interval
        :param nx: Number of distance ranges
        :param kc: Max wavenumber
        :param verbose: Verbose output
        :param debugMPI: Write MPI debug files
        :param showProgress: Print ETA on rank 0
        """
        title = f"[Stage 1] compute_gf_op: {dt=} {nfft=} {dk=} {tb=}"

        if rank == 0:
            print(f"\n{title}")
            hfile = h5py.File(h5_database_name + '.h5', 'r+')
        else:
            hfile = h5py.File(h5_database_name + '.h5', 'r')

        pairs_to_compute = hfile["/pairs_to_compute"][:]
        npairs = len(pairs_to_compute)

        if rank == 0:
            print(f"[Stage 1] Computing GF for {npairs} unique slots")
            # Clear and recreate tdata_dict group
            if "tdata_dict" in hfile:
                print("[Stage 1] Found existing tdata_dict, overwriting.")
                del hfile["tdata_dict"]
            tdata_group = hfile.create_group("tdata_dict")

        if debugMPI:
            fid_debug = open(f"rank_{rank}_stage1.debuginfo", "w")
            def printMPI(*args):
                fid_debug.write(" ".join(str(a) for a in args) + "\n")
        else:
            fid_debug = open(os.devnull, "w")
            printMPI = lambda *args: None

        # MPI distribution: rank 0 coordinates, workers compute
        if nprocs == 1 or rank == 0:
            next_pair = rank
            skip_pairs = 1
        else:
            next_pair = rank - 1
            skip_pairs = nprocs - 1

        perf_time_begin = perf_counter()
        perf_time_core  = np.zeros(1, dtype=np.float64)
        perf_time_send  = np.zeros(1, dtype=np.float64)
        perf_time_recv  = np.zeros(1, dtype=np.float64)

        tstart_pair = perf_counter()

        for ipair, (i_station, i_psource) in enumerate(pairs_to_compute):
            station = self._receivers.get_station_by_id(int(i_station))
            psource = self._source.get_source_by_id(int(i_psource))

            if ipair == next_pair:
                if nprocs == 1 or (rank > 0 and nprocs > 1):
                    # Compute crust model for this pair (only when needed)
                    aux_crust = copy.deepcopy(self._crust)
                    aux_crust.split_at_depth(psource.x[2])
                    aux_crust.split_at_depth(station.x[2])

                    t1 = perf_counter()
                    tdata, z, e, n, t0 = self._call_core_op(
                        dt, nfft, tb, nx, sigma, smth, wc1, wc2,
                        pmin, pmax, dk, kc, taper, aux_crust, psource, station, verbose)
                    perf_time_core += perf_counter() - t1

                    nt = len(z)
                    t0_arr = np.array([t0], dtype=np.float64)
                    tdata_c = np.empty((nt, 9), dtype=np.float64)
                    for comp in range(9):
                        tdata_c[:, comp] = tdata[0, comp, :]

                    if rank > 0:
                        t1 = perf_counter()
                        comm.Send(np.array([nt], dtype=np.int32), dest=0, tag=3 * ipair)
                        comm.Send(t0_arr,  dest=0, tag=3 * ipair + 1)
                        comm.Send(tdata_c, dest=0, tag=3 * ipair + 2)
                        perf_time_send += perf_counter() - t1
                        next_pair += skip_pairs

                if rank == 0:
                    if nprocs > 1:
                        remote = ipair % (nprocs - 1) + 1
                        t1 = perf_counter()
                        ant = np.empty(1, dtype=np.int32)
                        t0_arr = np.empty(1, dtype=np.float64)
                        comm.Recv(ant,    source=remote, tag=3 * ipair)
                        comm.Recv(t0_arr, source=remote, tag=3 * ipair + 1)
                        nt = ant[0]
                        tdata_c = np.empty((nt, 9), dtype=np.float64)
                        comm.Recv(tdata_c, source=remote, tag=3 * ipair + 2)
                        perf_time_recv += perf_counter() - t1

                    tdata_group[f"{ipair}_t0"]    = t0_arr[0]
                    tdata_group[f"{ipair}_tdata"]  = tdata_c

                    if showProgress:
                        elapsed = perf_counter() - tstart_pair
                        eta = elapsed / (ipair + 1) * (npairs - ipair - 1)
                        hh, rem = divmod(int(eta), 3600)
                        mm, ss = divmod(rem, 60)
                        print(f"[Stage 1] {ipair+1}/{npairs} ETA {hh:02d}:{mm:02d}:{ss:02d}")

                    next_pair += 1

        fid_debug.close()
        hfile.close()

        perf_time_total = perf_counter() - perf_time_begin
        if rank == 0:
            print(f"\n[Stage 1] Done. Total time: {perf_time_total:.1f}s")

        if use_mpi and nprocs > 1:
            # Gather and report perf stats
            all_max_core = np.array([-np.inf])
            all_min_core = np.array([ np.inf])
            comm.Reduce(perf_time_core, all_max_core, op=MPI.MAX, root=0)
            comm.Reduce(perf_time_core, all_min_core, op=MPI.MIN, root=0)
            if rank == 0:
                print(f"[Stage 1] core time: max={all_max_core[0]:.2f}s min={all_min_core[0]:.2f}s")

    # =========================================================================
    # Stage 2: Convolve GFs with STF and write output
    # =========================================================================

    def run_op(self,
               h5_database_name,
               dt=0.05,
               nfft=4096,
               tb=1000,
               smth=1,
               sigma=2,
               taper=0.9,
               wc1=1,
               wc2=2,
               pmin=0,
               pmax=1,
               dk=0.3,
               nx=1,
               kc=15.0,
               writer=None,
               writer_mode='progressive',
               verbose=False,
               debugMPI=False,
               tmin=0.,
               tmax=100,
               showProgress=True,
               allow_out_of_bounds=False):
        """
        Stage 2: For each (station, source) pair, look up tdata from DB,
        call _call_core_fast_op, convolve with STF, accumulate station response.

        Uses pair_to_slot for O(1) lookup if present in HDF5.
        Falls back to legacy linear search for backward compatibility.

        :param h5_database_name: HDF5 path (without .h5)
        :param writer: StationListWriter instance (DRM, HDF5, etc.)
        :param writer_mode: 'progressive' or 'legacy'
        :param allow_out_of_bounds: If True, use closest slot even outside tolerance
        """
        title = f"[Stage 2] run_op: {dt=} {nfft=} {tmin=} {tmax=}"

        if rank == 0:
            print(f"\n{title}")

        if rank > 0:
            hfile = h5py.File(h5_database_name + '.h5', 'r')
        else:
            hfile = h5py.File(h5_database_name + '.h5', 'r+')

        # Load mapping arrays
        dh_of_pairs   = hfile["/dh_of_pairs"][:]
        zrec_of_pairs = hfile["/zrec_of_pairs"][:]
        zsrc_of_pairs = hfile["/zsrc_of_pairs"][:]

        # Detect if optimized mapping is available
        use_mapping = "pair_to_slot" in hfile
        if use_mapping:
            pair_to_slot = hfile["/pair_to_slot"][:]
            nsources_db  = int(hfile["/nsources"][()])
            nstations_db = int(hfile["/nstations"][()])
            if rank == 0:
                print(f"[Stage 2] pair_to_slot found -> O(1) lookup enabled")
            # Validate dimensions match current source/receiver setup
            assert nsources_db == self._source.nsources, \
                f"[Stage 2] HDF5 nsources={nsources_db} != current {self._source.nsources}"
            assert nstations_db == self._receivers.nstations, \
                f"[Stage 2] HDF5 nstations={nstations_db} != current {self._receivers.nstations}"
        else:
            if rank == 0:
                print(f"[Stage 2] pair_to_slot not found -> legacy linear search (backward compat)")
            # Also load tolerances for legacy search
            delta_h     = float(hfile["/delta_h"][()])
            delta_v_rec = float(hfile["/delta_v_rec"][()])
            delta_v_src = float(hfile["/delta_v_src"][()])

        if rank > 0:
            writer = None

        if writer and rank == 0:
            assert isinstance(writer, StationListWriter), \
                "writer must be a StationListWriter instance"
            writer.initialize(self._receivers, 2 * nfft,
                              tmin=tmin, tmax=tmax, dt=dt,
                              writer_mode=writer_mode)
            writer.write_metadata(self._receivers.metadata)

        if debugMPI:
            fid_debug = open(f"rank_{rank}_stage2.debuginfo", "w")
            def printMPI(*args):
                fid_debug.write(" ".join(str(a) for a in args) + "\n")
        else:
            fid_debug = open(os.devnull, "w")
            printMPI = lambda *args: None

        nsources  = self._source.nsources
        nstations = self._receivers.nstations

        next_station  = rank
        skip_stations = nprocs

        perf_time_begin = perf_counter()
        perf_time_core  = np.zeros(1, dtype=np.float64)
        perf_time_conv  = np.zeros(1, dtype=np.float64)
        perf_time_add   = np.zeros(1, dtype=np.float64)
        perf_time_send  = np.zeros(1, dtype=np.float64)
        perf_time_recv  = np.zeros(1, dtype=np.float64)

        npairs_skip = 0
        n_my_stations = 0

        # ------------------------------------------------------------------ #
        # Stage 2a: Each rank processes its assigned stations                  #
        # ------------------------------------------------------------------ #
        for i_station, station in enumerate(self._receivers):
            if i_station != next_station:
                continue

            tstart_station = perf_counter()

            for i_psource, psource in enumerate(self._source):

                # --- Find slot k ---
                if use_mapping:
                    k = int(pair_to_slot[i_station * nsources + i_psource])
                else:
                    # Legacy linear search (JAA/PXP backward compat)
                    d = station.x - psource.x
                    dh    = np.sqrt(d[0]**2 + d[1]**2)
                    z_src = psource.x[2]
                    z_rec = station.x[2]

                    min_dist = np.inf
                    k = -1
                    for i in range(len(dh_of_pairs)):
                        in_tol = (abs(dh    - dh_of_pairs[i])   < delta_h and
                                  abs(z_src - zsrc_of_pairs[i]) < delta_v_src and
                                  abs(z_rec - zrec_of_pairs[i]) < delta_v_rec)
                        if in_tol or allow_out_of_bounds:
                            dist = (abs(dh    - dh_of_pairs[i]) +
                                    abs(z_src - zsrc_of_pairs[i]) +
                                    abs(z_rec - zrec_of_pairs[i]))
                            if dist < min_dist:
                                min_dist = dist
                                k = i

                    if k == -1:
                        print(f"[Stage 2] No match: {i_station=} {i_psource=} - SKIPPING")
                        npairs_skip += 1
                        if npairs_skip > 500:
                            print(f"[Stage 2] Rank {rank}: too many skipped pairs, aborting!")
                            if use_mpi and nprocs > 1:
                                comm.Abort()
                        continue

                # --- Load tdata from HDF5 ---
                tdata = hfile[f"/tdata_dict/{k}_tdata"][:]
                t0    = float(hfile[f"/tdata_dict/{k}_t0"][()])

                # --- Compute rotated GF (fast: reuses precomputed tdata) ---
                aux_crust = copy.deepcopy(self._crust)
                aux_crust.split_at_depth(psource.x[2])
                aux_crust.split_at_depth(station.x[2])

                t1 = perf_counter()
                z, e, n, t0_out = self._call_core_fast_op(
                    tdata, dt, nfft, tb, nx, sigma, smth, wc1, wc2,
                    pmin, pmax, dk, kc, taper, aux_crust, psource, station, verbose)
                perf_time_core += perf_counter() - t1

                # --- Convolve with source time function ---
                t1 = perf_counter()
                t_arr = np.arange(0, len(z) * dt, dt) + psource.tt + t0_out
                psource.stf.dt = dt
                z_stf = psource.stf.convolve(z, t_arr)
                e_stf = psource.stf.convolve(e, t_arr)
                n_stf = psource.stf.convolve(n, t_arr)
                perf_time_conv += perf_counter() - t1

                # --- Accumulate station response ---
                t1 = perf_counter()
                try:
                    station.add_to_response(z_stf, e_stf, n_stf, t_arr, tmin, tmax)
                except Exception:
                    traceback.print_exc()
                    if use_mpi and nprocs > 1:
                        comm.Abort()

                # --- Optionally store Green's function in station ---
                station.add_greens_function(z, e, n, t_arr, tdata, t0_out, i_psource)

                perf_time_add += perf_counter() - t1

                if showProgress and rank == 0 and i_psource % 500 == 0:
                    pct = i_psource / nsources * 100
                    elapsed = perf_counter() - tstart_station
                    eta = elapsed / (i_psource + 1) * (nsources - i_psource - 1)
                    print(f"[Stage 2] sta {i_station} src {i_psource}/{nsources} "
                          f"({pct:.1f}%) ETA={eta:.1f}s")

            # Station complete
            n_my_stations += 1

            if showProgress:
                elapsed = perf_counter() - tstart_station
                pct = i_station / nstations * 100
                print(f"[Stage 2] Rank {rank} station {i_station}/{nstations} "
                      f"({pct:.1f}%) done in {elapsed:.1f}s")

            next_station += skip_stations

        hfile.close()

        # ------------------------------------------------------------------ #
        # Stage 2b: Gather results to rank 0 and write output                 #
        # ------------------------------------------------------------------ #
        if use_mpi and nprocs > 1:
            # Worker ranks send their station data to rank 0
            if rank > 0:
                my_station = rank
                while my_station < nstations:
                    station = self._receivers.get_station_by_id(my_station)
                    z, e, n, t = station.get_response()

                    t1 = perf_counter()
                    comm.Send(np.array([len(z)], dtype=np.int32), dest=0, tag=2 * my_station)
                    data = np.column_stack([z, e, n, t])
                    comm.Send(data, dest=0, tag=2 * my_station + 1)
                    perf_time_send += perf_counter() - t1

                    my_station += nprocs

            # Rank 0 receives from all workers then writes
            if rank == 0:
                print("[Stage 2] Rank 0 gathering results...")

                for remote_rank in range(1, nprocs):
                    remote_station = remote_rank
                    while remote_station < nstations:
                        station = self._receivers.get_station_by_id(remote_station)

                        t1 = perf_counter()
                        ant = np.empty(1, dtype=np.int32)
                        comm.Recv(ant, source=remote_rank, tag=2 * remote_station)
                        nt = ant[0]
                        data = np.empty((nt, 4), dtype=np.float64)
                        comm.Recv(data, source=remote_rank, tag=2 * remote_station + 1)
                        perf_time_recv += perf_counter() - t1

                        z, e, n, t = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
                        station.add_to_response(z, e, n, t, tmin, tmax)

                        if writer:
                            writer.write_station(station, remote_station)

                        remote_station += nprocs

                # Write rank 0's own stations
                my_station = 0
                while my_station < nstations:
                    station = self._receivers.get_station_by_id(my_station)
                    if writer:
                        writer.write_station(station, my_station)
                    my_station += nprocs

                if writer:
                    writer.close()

                perf_time_total = perf_counter() - perf_time_begin
                print(f"\n[Stage 2] Done. Total time: {perf_time_total:.1f}s")

        else:
            # Single process: write directly
            if writer:
                for i_station, station in enumerate(self._receivers):
                    writer.write_station(station, i_station)
                writer.close()

            perf_time_total = perf_counter() - perf_time_begin
            print(f"\n[Stage 2] Done. Total time: {perf_time_total:.1f}s")

        fid_debug.close()

    # =========================================================================
    # Unified entry point: run all stages
    # =========================================================================

    def run_fast_faster_op(self,
                           stage='all',
                           h5_database_name=None,
                           # Stage 0 params
                           delta_h=0.04,
                           delta_v_rec=0.002,
                           delta_v_src=0.2,
                           npairs_max=200000,
                           # Core params (stages 1 & 2)
                           dt=0.05,
                           nfft=4096,
                           tb=1000,
                           smth=1,
                           sigma=2,
                           taper=0.9,
                           wc1=1,
                           wc2=2,
                           pmin=0,
                           pmax=1,
                           dk=0.3,
                           nx=1,
                           kc=15.0,
                           # Stage 2 params
                           writer=None,
                           writer_mode='progressive',
                           tmin=0.,
                           tmax=100,
                           allow_out_of_bounds=False,
                           # General
                           verbose=False,
                           debugMPI=False,
                           showProgress=True):
        """
        Unified pipeline entry point. Runs Stage 0, 1, and/or 2.

        :param stage: Which stage(s) to run: 0, 1, 2, or 'all'
        :param h5_database_name: HDF5 database path (without .h5 extension)
        :param delta_h: Horizontal distance tolerance for geometry grouping (km)
        :param delta_v_rec: Receiver depth tolerance (km)
        :param delta_v_src: Source depth tolerance (km)
        :param npairs_max: Max unique geometry slots in Stage 0
        :param dt: Time step (s)
        :param nfft: FFT size
        :param tb: Samples before first arrival
        :param smth: Output densification
        :param sigma: Damping
        :param taper: Low-pass taper
        :param wc1, wc2: Filter corner frequencies
        :param pmin, pmax: Phase velocity bounds
        :param dk: Wavenumber interval
        :param nx: Number of distance ranges
        :param kc: Max wavenumber
        :param writer: StationListWriter for output (Stage 2)
        :param writer_mode: 'progressive' or 'legacy' (Stage 2)
        :param tmin, tmax: Time window (s) for Stage 2
        :param allow_out_of_bounds: Use closest slot even outside tolerance
        :param verbose: Verbose Fortran core output
        :param debugMPI: Write MPI debug files
        :param showProgress: Print ETA progress
        """
        assert h5_database_name is not None, \
            "run_fast_faster_op: h5_database_name is required"
        assert stage in (0, 1, 2, 'all'), \
            "run_fast_faster_op: stage must be 0, 1, 2, or 'all'"

        run_s0 = stage in (0, 'all')
        run_s1 = stage in (1, 'all')
        run_s2 = stage in (2, 'all')

        if run_s0:
            self.gen_pairs_op(
                h5_database_name=h5_database_name,
                delta_h=delta_h,
                delta_v_rec=delta_v_rec,
                delta_v_src=delta_v_src,
                npairs_max=npairs_max,
                showProgress=showProgress)

        if run_s1:
            self.compute_gf_op(
                h5_database_name=h5_database_name,
                dt=dt, nfft=nfft, tb=tb, smth=smth,
                sigma=sigma, taper=taper, wc1=wc1, wc2=wc2,
                pmin=pmin, pmax=pmax, dk=dk, nx=nx, kc=kc,
                verbose=verbose, debugMPI=debugMPI,
                showProgress=showProgress)

        if run_s2:
            self.run_op(
                h5_database_name=h5_database_name,
                dt=dt, nfft=nfft, tb=tb, smth=smth,
                sigma=sigma, taper=taper, wc1=wc1, wc2=wc2,
                pmin=pmin, pmax=pmax, dk=dk, nx=nx, kc=kc,
                writer=writer, writer_mode=writer_mode,
                tmin=tmin, tmax=tmax,
                allow_out_of_bounds=allow_out_of_bounds,
                verbose=verbose, debugMPI=debugMPI,
                showProgress=showProgress)

    # =========================================================================
    # Internal: Fortran core wrappers (identical to JAA)
    # =========================================================================

    def _call_core_op(self, dt, nfft, tb, nx, sigma, smth, wc1, wc2,
                      pmin, pmax, dk, kc, taper, crust, psource, station, verbose=False):
        """Call core.subgreen to compute full FK kernel tdata."""
        mb  = crust.nlayers
        src = crust.get_layer(psource.x[2]) + 1
        rcv = crust.get_layer(station.x[2]) + 1

        stype = 2
        updn  = 0

        pf = psource.angles[0]
        df = psource.angles[1]
        lf = psource.angles[2]
        sx = psource.x[0]; sy = psource.x[1]
        rx = station.x[0]; ry = station.x[1]
        x  = np.sqrt((sx - rx)**2 + (sy - ry)**2)

        tdata, z, e, n, t0 = core.subgreen(
            mb, src, rcv, stype, updn,
            crust.d, crust.a, crust.b, crust.rho, crust.qa, crust.qb,
            dt, nfft, tb, nx, sigma, smth, wc1, wc2,
            pmin, pmax, dk, kc, taper, x, pf, df, lf, sx, sy, rx, ry)

        return tdata, z, e, n, t0

    def _call_core_fast_op(self, tdata, dt, nfft, tb, nx, sigma, smth, wc1, wc2,
                           pmin, pmax, dk, kc, taper, crust, psource, station, verbose=False):
        """Call core.subgreen2 using precomputed tdata kernel."""
        mb  = crust.nlayers
        src = crust.get_layer(psource.x[2]) + 1
        rcv = crust.get_layer(station.x[2]) + 1

        stype = 2
        updn  = 0

        pf = psource.angles[0]
        df = psource.angles[1]
        lf = psource.angles[2]
        sx = psource.x[0]; sy = psource.x[1]
        rx = station.x[0]; ry = station.x[1]
        x  = np.sqrt((sx - rx)**2 + (sy - ry)**2)

        # Reshape tdata to format expected by subgreen2
        tdata_ = tdata.T
        tdata_ = tdata_.reshape((1, tdata_.shape[0], tdata_.shape[1]))

        z, e, n, t0 = core.subgreen2(
            mb, src, rcv, stype, updn,
            crust.d, crust.a, crust.b, crust.rho, crust.qa, crust.qb,
            dt, nfft, tb, nx, sigma, smth, wc1, wc2,
            pmin, pmax, dk, kc, taper, x, pf, df, lf, tdata_, sx, sy, rx, ry)

        return z, e, n, t0

    # =========================================================================
    # Properties (mirror original ShakerMaker)
    # =========================================================================

    @property
    def mpi_rank(self):
        return self._mpi_rank

    @property
    def mpi_nprocs(self):
        return self._mpi_nprocs
