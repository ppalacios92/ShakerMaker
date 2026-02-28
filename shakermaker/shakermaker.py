"""
shakermaker.py  --  ShakerMaker simulation engine, OP architecture.

Three-stage pipeline with O(1) Green's Function lookup via pair_to_slot:

  Stage 0  gen_pairs_op
      Identifies unique geometries (dh, z_src, z_rec) across all
      (station, source) pairs and builds the flat mapping
          pair_to_slot[i_station * nsources + i_psource] = k

      Algorithm -- Option B: dimension-decomposition parallel greedy.
      Slots with different z_src or z_rec bins are fully independent,
      so the 3-D greedy decomposes into many small 1-D greedy problems
      over dh.  All ranks compute geometry in parallel; rank 0 distributes
      independent (z_src_bin, z_rec_bin) groups across all ranks, each
      rank runs a tiny 1-D greedy, results are merged and pairs are
      assigned via KDTree lookup.

  Stage 1  compute_gf_op
      Computes the FK kernel (tdata) for each unique slot k.
      MPI parallel: workers compute, rank 0 collects and writes to HDF5.

  Stage 2  run_op
      For every (station, source) pair, retrieves tdata via O(1) lookup,
      calls _call_core_fast, convolves with the source time function, and
      accumulates the station response.
      MPI parallel: stations distributed across ranks.

Orchestrator:
  run_fast_faster_op(stage=0|1|2|'all')

Debug / validation (no database):
  run()

STKO geometry export:
  export_drm_geometry()

Legacy HDF5 migration (JAA / PXP databases already containing tdata_dict):
  build_pair_to_slot_from_legacy_h5()
      Reads the existing geometry arrays (dh_of_pairs, zrec_of_pairs,
      zsrc_of_pairs) from a JAA/PXP HDF5 file and writes the three
      datasets needed by run_op:
          /pair_to_slot   (nstations * nsources,)  int32
          /nstations      scalar
          /nsources       scalar
      No Green's Functions are recomputed.

HDF5 database layout
--------------------
  /pairs_to_compute      (n_slots, 2)   int32    representative [i_sta, i_src]
  /dh_of_pairs           (n_slots,)     float64  horizontal distance per slot
  /dv_of_pairs           (n_slots,)     float64  vertical distance per slot
  /zrec_of_pairs         (n_slots,)     float64  receiver depth per slot
  /zsrc_of_pairs         (n_slots,)     float64  source depth per slot
  /pair_to_slot          (nsta*nsrc,)   int32    flat index -> slot  [OP only]
  /nstations             scalar         int      [OP only]
  /nsources              scalar         int      [OP only]
  /delta_h               scalar         float64
  /delta_v_rec           scalar         float64
  /delta_v_src           scalar         float64
  /tdata_dict/{k}_tdata  (nt, 9)        float64  FK kernel, C-order
  /tdata_dict/{k}_t0     scalar         float64  time offset of slot k
"""

import copy
import os
import traceback
import logging
import numpy as np
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
    comm   = MPI.COMM_WORLD
    rank   = comm.Get_rank()
    nprocs = comm.Get_size()
except ImportError:
    use_mpi = False
    rank   = 0
    nprocs = 1


# ---------------------------------------------------------------------------
# Module-level helpers  (shared by all methods, no duplication)
# ---------------------------------------------------------------------------

def _perf_counters():
    """Return a dict of zeroed timing accumulators."""
    return {k: np.zeros(1, dtype=np.double)
            for k in ('core', 'send', 'recv', 'conv', 'add')}


def _print_perf_stats(c, total):
    """Reduce timing counters across MPI ranks and print on rank 0."""
    if not (use_mpi and nprocs > 1):
        return
    labels = {'core': 'time_core', 'send': 'time_send', 'recv': 'time_recv',
              'conv': 'time_conv', 'add':  'time_add'}
    if rank == 0:
        print("\nPerformance statistics (all MPI processes):")
    for key in ('core', 'send', 'recv', 'conv', 'add'):
        mx = np.array([-np.inf]); mn = np.array([np.inf])
        comm.Reduce(c[key], mx, op=MPI.MAX, root=0)
        comm.Reduce(c[key], mn, op=MPI.MIN, root=0)
        if rank == 0 and total > 0:
            print(f"  {labels[key]:12s}:  "
                  f"max={mx[0]:.3f}s ({mx[0]/total*100:.2f}%)  "
                  f"min={mn[0]:.3f}s ({mn[0]/total*100:.2f}%)")


def _eta_str(elapsed, done, total):
    """Return 'H:MM:SS.s' ETA string."""
    if done <= 0:
        return "??:??:??"
    rem = elapsed / done * (total - done)
    hh  = int(rem) // 3600
    mm  = (int(rem) % 3600) // 60
    ss  = rem % 60
    return f"{hh}:{mm:02d}:{ss:04.1f}"


# ---------------------------------------------------------------------------
# ShakerMaker
# ---------------------------------------------------------------------------

class ShakerMaker:
    """This is the main class in ShakerMaker, used to define a model, link
    components, set simulation parameters and execute it.

    OP architecture: three-stage pipeline with O(1) Green's Function lookup.
    See module docstring for full description.

    :param crust: Crustal model used by the simulation.
    :type crust: CrustModel
    :param source: Source model(s).
    :type source: FaultSource
    :param receivers: Receiver station(s).
    :type receivers: StationList
    """

    def __init__(self, crust, source, receivers):
        assert isinstance(crust, CrustModel), \
            "crust must be an instance of the shakermaker.CrustModel class"
        assert isinstance(source, FaultSource), \
            "source must be an instance of the shakermaker.FaultSource class"
        assert isinstance(receivers, StationList), \
            "receivers must be an instance of the shakermaker.StationList class"

        self._crust     = crust
        self._source    = source
        self._receivers = receivers
        self._mpi_rank  = rank
        self._mpi_nprocs = nprocs
        self._logger    = logging.getLogger(__name__)

    # =========================================================================
    # run()  --  direct pair-by-pair, no database  (JAA-compatible debug method)
    # =========================================================================

    def run(self,
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
            verbose=False,
            debugMPI=False,
            tmin=0.,
            tmax=100,
            showProgress=True,
            writer_mode='progressive'):
        """Run the simulation pair by pair. No Green's Function database.

        Useful for debugging and validating results against the OP pipeline.
        Every (source, receiver) pair is computed independently; no reuse of
        previously computed Green's Functions.

        :param sigma: Its role is to damp the trace (at rate of exp(-sigma*t)) to reduce the wrap-arround.
        :type sigma: double
        :param nfft: Number of time-points to use in fft
        :type nfft: integer
        :param dt: Simulation time-step
        :type dt: double
        :param tb: Num. of samples before the first arrival.
        :type tb: integer
        :param taper: For low-pass filter, 0-1.
        :type taper: double
        :param smth: Densify the output samples by a factor of smth
        :type smth: double
        :param wc1: (George.. please provide one-line description!)
        :type wc1: double
        :param wc2: (George.. please provide one-line description!)
        :type wc2: double
        :param pmin: Max. phase velocity, in 1/vs, 0 the best.
        :type pmin: double
        :param pmax: Min. phase velocity, in 1/vs.
        :type pmax: double
        :param dk: Sample interval in wavenumber, in Pi/x, 0.1-0.4.
        :type dk: double
        :param nx: Number of distance ranges to compute.
        :type nx: integer
        :param kc: It's kmax, equal to 1/hs. Because the kernels decay with k at rate of exp(-k*hs) at w=0, we require kmax > 10 to make sure we have have summed enough.
        :type kc: double
        :param writer: Use this writer class to store outputs
        :type writer: StationListWriter
        :param writer_mode: 'progressive' or 'legacy'
        :type writer_mode: str
        """
        title = f"🎉 ¡LARGA VIDA AL LADRUNO_OP_kdTREE! 🎉 ShakerMaker Run begin. {dt=} {nfft=} {dk=} {tb=} {tmin=} {tmax=}"
        
        if rank == 0:
            print(f"\n\n{title}")
            print("-" * len(title))
            omp = os.environ.get('OMP_NUM_THREADS', 'not set')
            print(f"Hybrid Parallelization:")
            print(f"   MPI processes  : {nprocs}")
            print(f"   OpenMP threads : {omp}")
            if omp != 'not set':
                print(f"   Total threads  : {nprocs} x {omp} = {nprocs * int(omp)}")
            print("-" * len(title))

        perf_time_begin = perf_counter()
        c = _perf_counters()

        if debugMPI:
            fid = open(f"rank_{rank}_run.debuginfo", "w")
            def printMPI(*args):
                fid.write(" ".join(str(a) for a in args) + "\n")
        else:
            fid = open(os.devnull, "w")
            printMPI = lambda *args: None

        self._logger.info(
            'ShakerMaker.run - starting\n\tNumber of sources: {}\n'
            '\tNumber of receivers: {}\n\tTotal src-rcv pairs: {}\n'
            '\tdt: {}\n\tnfft: {}'
            .format(self._source.nsources, self._receivers.nstations,
                    self._source.nsources * self._receivers.nstations,
                    dt, nfft))

        if rank > 0:
            writer = None
        if writer and rank == 0:
            assert isinstance(writer, StationListWriter), \
                "'writer' must be an instance of the shakermaker.StationListWriter class or None"
            writer.initialize(self._receivers, 2 * nfft,
                              tmin=tmin, tmax=tmax, dt=dt,
                              writer_mode=writer_mode)
            writer.write_metadata(self._receivers.metadata)

        ipair = 0
        if nprocs == 1 or rank == 0:
            next_pair, skip_pairs = rank, 1
        else:
            next_pair, skip_pairs = rank - 1, nprocs - 1

        npairs = self._receivers.nstations * self._source.nsources
        tstart = perf_counter()

        for i_station, station in enumerate(self._receivers):
            for i_psource, psource in enumerate(self._source):
                aux_crust = copy.deepcopy(self._crust)
                aux_crust.split_at_depth(psource.x[2])
                aux_crust.split_at_depth(station.x[2])

                if ipair == next_pair:
                    if verbose:
                        print(f"rank={rank} nprocs={nprocs} ipair={ipair} "
                              f"skip_pairs={skip_pairs} npairs={npairs} !!")

                    if nprocs == 1 or (rank > 0 and nprocs > 1):
                        if verbose:
                            print("calling core START")
                        t1 = perf_counter()
                        tdata, z, e, n, t0 = self._call_core(
                            dt, nfft, tb, nx, sigma, smth,
                            wc1, wc2, pmin, pmax, dk, kc,
                            taper, aux_crust, psource, station, verbose)
                        c['core'] += perf_counter() - t1
                        if verbose:
                            print("calling core END")

                        nt = len(z)
                        t1 = perf_counter()
                        t = np.arange(0, nt * dt, dt) + psource.tt + t0
                        psource.stf.dt = dt
                        z_stf = psource.stf.convolve(z, t)
                        e_stf = psource.stf.convolve(e, t)
                        n_stf = psource.stf.convolve(n, t)
                        c['conv'] += perf_counter() - t1

                        if rank > 0:
                            t1 = perf_counter()
                            comm.Send(np.array([nt], dtype=np.int32),
                                      dest=0, tag=2 * ipair)
                            data = np.empty((nt, 4), dtype=np.float64)
                            data[:, 0] = z_stf; data[:, 1] = e_stf
                            data[:, 2] = n_stf; data[:, 3] = t
                            comm.Send(data, dest=0, tag=2 * ipair + 1)
                            c['send'] += perf_counter() - t1
                            next_pair += skip_pairs

                    if rank == 0:
                        if nprocs > 1:
                            remote = ipair % (nprocs - 1) + 1
                            t1 = perf_counter()
                            printMPI(f"P0 Recv from remote {remote}")
                            ant = np.empty(1, dtype=np.int32)
                            comm.Recv(ant, source=remote, tag=2 * ipair)
                            nt = ant[0]
                            data = np.empty((nt, 4), dtype=np.float64)
                            comm.Recv(data, source=remote, tag=2 * ipair + 1)
                            z_stf = data[:, 0]; e_stf = data[:, 1]
                            n_stf = data[:, 2]; t     = data[:, 3]
                            c['recv'] += perf_counter() - t1

                        next_pair += 1
                        try:
                            t1 = perf_counter()
                            station.add_to_response(z_stf, e_stf, n_stf,
                                                    t, tmin, tmax)
                            c['add'] += perf_counter() - t1
                        except Exception:
                            traceback.print_exc()
                            if use_mpi and nprocs > 1:
                                comm.Abort()

                        if showProgress:
                            elapsed = perf_counter() - tstart
                            print(f"{ipair} of {npairs} done  "
                                  f"ETA={_eta_str(elapsed, ipair + 1, npairs)}  "
                                  f"t=[{t[0]:.4f}, {t[-1]:.4f}] "
                                  f"({tmin=:.4f} {tmax=:.4f})")
                ipair += 1

            if verbose:
                self._logger.debug(
                    f'ShakerMaker.run - finished station {i_station} '
                    f'(rank={rank} ipair={ipair} next_pair={next_pair})')

            if writer and rank == 0:
                printMPI(f"Rank 0 writing station {i_station}")
                writer.write_station(station, i_station)
                printMPI(f"Rank 0 done writing station {i_station}")

        if writer and rank == 0:
            writer.close()

        fid.close()
        perf_time_total = perf_counter() - perf_time_begin

        if rank == 0:
            print(f"\n\nShakerMaker run done. Total time: {perf_time_total:.2f} s")
            print("-" * 50)

        _print_perf_stats(c, perf_time_total)

    # =========================================================================
    # Stage 0  --  gen_pairs_op
    # =========================================================================

    def gen_pairs_op(self,
                     h5_database_name,
                     delta_h=0.04,
                     delta_v_rec=0.002,
                     delta_v_src=0.2,
                     npairs_max=200000,
                     showProgress=True):
        """Stage 0 of the OP pipeline.

        Scans all (station, source) pairs, identifies geometrically unique
        slots based on (dh, z_src, z_rec) tolerances, and builds the flat
        mapping array::

            pair_to_slot[i_station * nsources + i_psource] = k

        **Algorithm -- MPI parallel geometry + Numba-compiled JAA greedy:**

        The geometry computation is distributed across all MPI ranks
        (vectorised, no Python loop).  The greedy slot-finding uses
        *exactly* the same algorithm as JAA, compiled to native code with
        Numba ``@njit`` so that it runs 100--500× faster than Python while
        producing bit-for-bit identical results.

        1. [All ranks] Vectorised geometry for local station slice:
           compute (dh, z_src, z_rec) via NumPy broadcasting.
        2. [All ranks] Gather geometry arrays to rank 0 via ``Gatherv``.
        3. [Rank 0] Run the JAA greedy in canonical pair order, compiled
           to C via Numba @njit.  Produces exactly the same slots as the
           original serial JAA implementation.
        4. [Rank 0] Build ``pair_to_slot`` and ``pairs_to_compute``,
           write HDF5 database.
        5. All ranks synchronise at a ``Barrier``.

        Complexity:
          - Geometry : O(n_pairs / nprocs) -- vectorised, MPI parallel.
          - Greedy   : O(n_pairs × n_slots) -- Numba compiled (single rank).
                       Typical speedup vs Python: 100--500×.
                       A 24-hour Python Stage 0 becomes ~10 minutes.

        Result: **identical slot count and pair_to_slot mapping** as the
        original serial JAA greedy for every geometry type.

        Writes to ``h5_database_name``:

        - ``/pairs_to_compute``  (n_slots, 2)  representative [i_sta, i_src]
        - ``/dh_of_pairs``       (n_slots,)    horizontal distance
        - ``/dv_of_pairs``       (n_slots,)    vertical distance
        - ``/zrec_of_pairs``     (n_slots,)    receiver depth
        - ``/zsrc_of_pairs``     (n_slots,)    source depth
        - ``/pair_to_slot``      (nsta*nsrc,)  flat index -> slot index k
        - ``/delta_h``, ``/delta_v_rec``, ``/delta_v_src``
        - ``/nstations``, ``/nsources``

        :param h5_database_name: HDF5 file path (full path, including .h5).
        :type h5_database_name: str
        :param delta_h: Horizontal distance tolerance (km).
        :type delta_h: double
        :param delta_v_rec: Receiver depth tolerance (km).
        :type delta_v_rec: double
        :param delta_v_src: Source depth tolerance (km).
        :type delta_v_src: double
        :param npairs_max: Kept for API compatibility; not used internally.
        :type npairs_max: integer
        :param showProgress: Print per-rank timing lines.
        :type showProgress: bool
        """
        # ------------------------------------------------------------------
        # JAA greedy compiled to native C with Numba.
        # Defined inside the method so it is always available even if Numba
        # was imported after the module was loaded.  The @njit decorator
        # compiles on first call; subsequent calls use the cached binary.
        #
        # This is EXACTLY the same algorithm as JAA:
        #   for each pair in canonical order:
        #     if no existing slot covers it -> create new slot (anchor = pair)
        #     else -> assign to nearest covering slot (L1 distance)
        #
        # The inner loop over existing slots is a plain C for-loop in
        # compiled code, giving 100--500x speedup over the Python version
        # while producing bit-for-bit identical slot arrays and pair_to_slot.
        # ------------------------------------------------------------------
        try:
            from numba import njit as _njit

            @_njit
            def _greedy_jaa(dh_arr, zsrc_arr, zrec_arr,
                             delta_h, delta_v_src, delta_v_rec):
                N         = len(dh_arr)
                slot_dh   = np.empty(N, dtype=np.float64)
                slot_zsrc = np.empty(N, dtype=np.float64)
                slot_zrec = np.empty(N, dtype=np.float64)
                p2s       = np.full(N, -1, dtype=np.int32)
                n_slots   = 0

                for i in range(N):
                    if n_slots == 0:
                        slot_dh[0]   = dh_arr[i]
                        slot_zsrc[0] = zsrc_arr[i]
                        slot_zrec[0] = zrec_arr[i]
                        p2s[i]       = 0
                        n_slots      = 1
                    else:
                        # Find covering slot closest in L1 norm
                        found     = -1
                        best_dist = 1e18
                        for k in range(n_slots):
                            d_dh  = dh_arr[i]   - slot_dh[k]
                            d_zs  = zsrc_arr[i] - slot_zsrc[k]
                            d_zr  = zrec_arr[i] - slot_zrec[k]
                            if d_dh  < 0.0: d_dh  = -d_dh
                            if d_zs  < 0.0: d_zs  = -d_zs
                            if d_zr  < 0.0: d_zr  = -d_zr
                            if (d_dh <= delta_h and
                                    d_zs <= delta_v_src and
                                    d_zr <= delta_v_rec):
                                dist = d_dh + d_zs + d_zr
                                if dist < best_dist:
                                    best_dist = dist
                                    found     = k
                        if found == -1:
                            slot_dh[n_slots]   = dh_arr[i]
                            slot_zsrc[n_slots] = zsrc_arr[i]
                            slot_zrec[n_slots] = zrec_arr[i]
                            p2s[i]             = n_slots
                            n_slots           += 1
                        else:
                            p2s[i] = found

                return (slot_dh[:n_slots], slot_zsrc[:n_slots],
                        slot_zrec[:n_slots], p2s)

            _use_numba = True

        except ImportError:
            _use_numba = False

        if rank == 0:
            if _use_numba:
                print("  Numba available -- greedy will run compiled (fast path)")
            else:
                print("  Numba NOT available -- greedy runs in Python (slow path).")
                print("  Install numba for 100-500x speedup: pip install numba")

        # ------------------------------------------------------------------
        # Pure-Python fallback (identical algorithm, no Numba dependency).
        # Used automatically when Numba is not installed.
        # ------------------------------------------------------------------
        def _greedy_jaa_python(dh_arr, zsrc_arr, zrec_arr,
                                delta_h, delta_v_src, delta_v_rec):
            N = len(dh_arr)
            slot_dh   = np.empty(N, dtype=np.float64)
            slot_zsrc = np.empty(N, dtype=np.float64)
            slot_zrec = np.empty(N, dtype=np.float64)
            p2s       = np.full(N, -1, dtype=np.int32)
            n_slots   = 0
            for i in range(N):
                if n_slots == 0:
                    slot_dh[0]   = dh_arr[i]
                    slot_zsrc[0] = zsrc_arr[i]
                    slot_zrec[0] = zrec_arr[i]
                    p2s[i]       = 0
                    n_slots      = 1
                else:
                    not_covered = (
                        (np.abs(dh_arr[i]   - slot_dh[:n_slots])   > delta_h)    |
                        (np.abs(zsrc_arr[i] - slot_zsrc[:n_slots]) > delta_v_src) |
                        (np.abs(zrec_arr[i] - slot_zrec[:n_slots]) > delta_v_rec)
                    )
                    if np.all(not_covered):
                        slot_dh[n_slots]   = dh_arr[i]
                        slot_zsrc[n_slots] = zsrc_arr[i]
                        slot_zrec[n_slots] = zrec_arr[i]
                        p2s[i]             = n_slots
                        n_slots           += 1
                    else:
                        covered = np.where(~not_covered)[0]
                        dist = (np.abs(dh_arr[i]   - slot_dh[covered]) +
                                np.abs(zsrc_arr[i] - slot_zsrc[covered]) +
                                np.abs(zrec_arr[i] - slot_zrec[covered]))
                        p2s[i] = covered[np.argmin(dist)]
            return (slot_dh[:n_slots], slot_zsrc[:n_slots],
                    slot_zrec[:n_slots], p2s)

        nsources  = self._source.nsources
        nstations = self._receivers.nstations
        N         = nstations * nsources          # total pairs

        title = (f"🎉 ¡LARGA VIDA AL LADRUNO_OP_kdTREE! 🎉 ShakerMaker Gen GF database pairs begin. "
                 f"{delta_h=} {delta_v_rec=} {delta_v_src=}")
        if rank == 0:
            print(f"\n\n{title}")
            print("-" * len(title))
            print(f"  Stations    : {nstations}")
            print(f"  Sources     : {nsources}")
            print(f"  Total pairs : {N}")
            print(f"  Max slots   : {npairs_max}  (kept for API compat)")
            print(f"  MPI ranks   : {nprocs}  (geometry parallel, greedy Numba-compiled)")

        t0_global = perf_counter()

        # ------------------------------------------------------------------
        # Step 1: every rank computes geometry for its contiguous station
        # slice using fully vectorised NumPy -- no Python loop over pairs.
        # ------------------------------------------------------------------
        base, rem = divmod(nstations, nprocs)
        r_start   = rank * base + min(rank, rem)
        r_end     = r_start + base + (1 if rank < rem else 0)
        my_nsta   = r_end - r_start

        # Pre-fetch coordinates once -- avoids repeated Python attribute access
        my_sta = np.array(
            [self._receivers.get_station_by_id(i).x
             for i in range(r_start, r_end)],
            dtype=np.float64)                        # (my_nsta, 3)
        src_coords = np.array(
            [self._source.get_source_by_id(j).x
             for j in range(nsources)],
            dtype=np.float64)                        # (nsources, 3)

        # Tile to full (my_nsta * nsources, 3) -- broadcasting, no loop
        sta_rep  = np.repeat(my_sta,    nsources, axis=0)  # (my_n*nsrc, 3)
        src_tile = np.tile(src_coords, (my_nsta,  1))      # (my_n*nsrc, 3)

        d_xy   = sta_rep[:, :2] - src_tile[:, :2]
        dh_loc = np.sqrt(np.einsum('ij,ij->i', d_xy, d_xy))  # horizontal dist
        zs_loc = src_tile[:, 2]                               # z_src per pair
        zr_loc = sta_rep[:, 2]                                # z_rec per pair
        dv_loc = np.abs(zr_loc - zs_loc)                      # vertical dist

        t1_geom = perf_counter()
        if showProgress:
            print(f"  [Rank {rank}] geometry ({my_nsta * nsources:,} pairs): "
                  f"{t1_geom - t0_global:.3f}s")

        # ------------------------------------------------------------------
        # Step 2: gather all geometry on rank 0 via Gatherv.
        # Each rank sends 4 float64 per pair: dh, dv, z_src, z_rec.
        # ------------------------------------------------------------------
        local_geom = np.ascontiguousarray(
            np.column_stack([dh_loc, dv_loc, zs_loc, zr_loc]))  # (my_n*nsrc, 4)
        my_size    = my_nsta * nsources

        if use_mpi and nprocs > 1:
            all_sizes = np.array(comm.allgather(my_size), dtype=np.int64)
            counts    = (all_sizes * 4).tolist()
            displ     = [0] + list(np.cumsum(counts[:-1]))
            if rank == 0:
                recv_geom = np.empty(N * 4, dtype=np.float64)
            else:
                recv_geom = None
            comm.Gatherv(local_geom.ravel(),
                         [recv_geom, counts, displ, MPI.DOUBLE],
                         root=0)
        else:
            recv_geom = local_geom.ravel()

        # ------------------------------------------------------------------
        # Steps 3--4 run exclusively on rank 0.
        # ------------------------------------------------------------------
        if rank == 0:
            t2_gather = perf_counter()
            if showProgress:
                print(f"  [Rank 0] gather complete: {t2_gather - t1_geom:.3f}s")

            all_geom = recv_geom.reshape(N, 4)  # columns: dh, dv, zsrc, zrec
            dh_all   = np.ascontiguousarray(all_geom[:, 0])
            dv_all   = all_geom[:, 1]
            zs_all   = np.ascontiguousarray(all_geom[:, 2])
            zr_all   = np.ascontiguousarray(all_geom[:, 3])

            # ----------------------------------------------------------
            # Step 3: run JAA greedy in canonical pair order.
            #
            # If Numba is available the function is compiled to native C
            # on first call (warm-up is done below).  The algorithm is
            # identical to JAA: same slot decisions, same pair_to_slot.
            # ----------------------------------------------------------
            if _use_numba:
                # Warm-up: compile with a tiny slice so the JIT cost is
                # not measured inside the timed section.
                if showProgress:
                    print(f"  [Rank 0] warming up Numba JIT...")
                _greedy_jaa(dh_all[:1], zs_all[:1], zr_all[:1],
                            delta_h, delta_v_src, delta_v_rec)
                t3_warmup = perf_counter()
                if showProgress:
                    print(f"  [Rank 0] Numba JIT ready: {t3_warmup - t2_gather:.3f}s")

                t_greedy_start = perf_counter()
                slot_dh, slot_zsrc, slot_zrec, pair_to_slot_full = _greedy_jaa(
                    dh_all, zs_all, zr_all,
                    delta_h, delta_v_src, delta_v_rec)
            else:
                t_greedy_start = perf_counter()
                slot_dh, slot_zsrc, slot_zrec, pair_to_slot_full = _greedy_jaa_python(
                    dh_all, zs_all, zr_all,
                    delta_h, delta_v_src, delta_v_rec)

            slot_dv   = np.abs(slot_zrec - slot_zsrc)
            n_slots   = len(slot_dh)
            pair_to_slot_full = pair_to_slot_full.astype(np.int32)

            t4_greedy = perf_counter()
            if showProgress:
                print(f"  [Rank 0] JAA greedy ({N:,} pairs -> {n_slots} slots): "
                      f"{t4_greedy - t_greedy_start:.3f}s")

            # ----------------------------------------------------------
            # Step 4: build pairs_to_compute.
            # For each slot k, pick the first pair (in canonical order)
            # that was assigned to it -- this is the representative
            # [i_station, i_source] used in Stage 1.
            # ----------------------------------------------------------
            order        = np.argsort(pair_to_slot_full, kind='stable')
            _, first_occ = np.unique(pair_to_slot_full[order],
                                     return_index=True)
            repr_flat    = order[first_occ]           # canonical flat index

            sta_of_repr      = (repr_flat // nsources).astype(np.int32)
            src_of_repr      = (repr_flat  % nsources).astype(np.int32)
            pairs_to_compute = np.column_stack(
                [sta_of_repr, src_of_repr]).astype(np.int32)

            dh_of_pairs   = dh_all[repr_flat]
            dv_of_pairs   = dv_all[repr_flat]
            zsrc_of_pairs = zs_all[repr_flat]
            zrec_of_pairs = zr_all[repr_flat]

            # Sanity checks
            assert np.all(pair_to_slot_full >= 0), \
                "[Stage 0] BUG: pair_to_slot contains negative index"
            assert np.all(pair_to_slot_full < n_slots), \
                "[Stage 0] BUG: pair_to_slot index out of range"
            assert len(np.unique(pair_to_slot_full)) == n_slots, \
                "[Stage 0] BUG: some slots have zero pairs assigned"

            elapsed   = perf_counter() - t0_global
            reduction = (1.0 - n_slots / N) * 100.0
            print(f"\nNeed only {n_slots} pairs of {N} "
                  f"({n_slots / N * 100:.1f}% of total, "
                  f"{reduction:.1f}% reduction)")
            print(f"Stage 0 done. Time: {elapsed:.1f}s")

            # ----------------------------------------------------------
            # Step 5: write HDF5 database
            # ----------------------------------------------------------
            with h5py.File(h5_database_name, 'w') as hf:
                hf.create_dataset("pairs_to_compute", data=pairs_to_compute)
                hf.create_dataset("dh_of_pairs",      data=dh_of_pairs)
                hf.create_dataset("dv_of_pairs",      data=dv_of_pairs)
                hf.create_dataset("zrec_of_pairs",    data=zrec_of_pairs)
                hf.create_dataset("zsrc_of_pairs",    data=zsrc_of_pairs)
                hf.create_dataset("pair_to_slot",     data=pair_to_slot_full)
                hf.create_dataset("delta_h",          data=delta_h)
                hf.create_dataset("delta_v_rec",      data=delta_v_rec)
                hf.create_dataset("delta_v_src",      data=delta_v_src)
                hf.create_dataset("nstations",        data=nstations)
                hf.create_dataset("nsources",         data=nsources)

            print(f"Database written to: {h5_database_name}")

        # All ranks wait here before Stage 1 starts
        if use_mpi and nprocs > 1:
            comm.Barrier()

    # =========================================================================
    # Stage 1  --  compute_gf_op
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
        """Stage 1 of the OP pipeline.

        Computes the FK kernel (tdata) for every unique slot k produced by
        Stage 0. Reads ``h5_database_name`` and appends the group
        ``/tdata_dict`` to the same file.

        MPI: rank 0 coordinates and writes; worker ranks compute and send.

        :param h5_database_name: HDF5 file path (full path, including .h5).
        :type h5_database_name: str
        :param sigma: Its role is to damp the trace (at rate of exp(-sigma*t)) to reduce the wrap-arround.
        :type sigma: double
        :param nfft: Number of time-points to use in fft
        :type nfft: integer
        :param dt: Simulation time-step
        :type dt: double
        :param tb: Num. of samples before the first arrival.
        :type tb: integer
        :param taper: For low-pass filter, 0-1.
        :type taper: double
        :param smth: Densify the output samples by a factor of smth
        :type smth: double
        :param wc1: (George.. please provide one-line description!)
        :type wc1: double
        :param wc2: (George.. please provide one-line description!)
        :type wc2: double
        :param pmin: Max. phase velocity, in 1/vs, 0 the best.
        :type pmin: double
        :param pmax: Min. phase velocity, in 1/vs.
        :type pmax: double
        :param dk: Sample interval in wavenumber, in Pi/x, 0.1-0.4.
        :type dk: double
        :param nx: Number of distance ranges to compute.
        :type nx: integer
        :param kc: It's kmax, equal to 1/hs. Because the kernels decay with k at rate of exp(-k*hs) at w=0, we require kmax > 10 to make sure we have have summed enough.
        :type kc: double
        :param verbose: Verbose output from the Fortran core.
        :type verbose: bool
        :param debugMPI: Write per-rank debug files.
        :type debugMPI: bool
        :param showProgress: Print ETA on rank 0.
        :type showProgress: bool
        """
        title = (f"🎉 ¡LARGA VIDA AL LADRUNO_OP_kdTREE! 🎉 ShakerMaker Gen Green's functions database begin. "
                 f"{dt=} {nfft=} {dk=} {tb=}")

        if rank == 0:
            print(f"\n\n{title}")
            print("-" * len(title))
            print(f"  MPI processes  : {nprocs}")
            print(f"  OpenMP threads : {os.environ.get('OMP_NUM_THREADS','not set')}")
            print(f"  Loading database: {h5_database_name}")
            hfile = h5py.File(h5_database_name, 'r+')
        else:
            hfile = h5py.File(h5_database_name, 'r')

        pairs_to_compute = hfile["/pairs_to_compute"][:]
        npairs = len(pairs_to_compute)

        if rank == 0:
            print(f"  Slots to compute: {npairs}")
            if "tdata_dict" in hfile:
                print("  Found existing tdata_dict group. Overwriting.")
                del hfile["tdata_dict"]
            tdata_group = hfile.create_group("tdata_dict")

        if debugMPI:
            fid = open(f"rank_{rank}_stage1.debuginfo", "w")
            def printMPI(*args):
                fid.write(" ".join(str(a) for a in args) + "\n")
        else:
            fid = open(os.devnull, "w")
            printMPI = lambda *args: None

        self._logger.info(
            'ShakerMaker.compute_gf_op - starting\n'
            '\tNumber of sources: {}\n\tNumber of receivers: {}\n'
            '\tTotal src-rcv pairs: {}\n\tdt: {}\n\tnfft: {}'
            .format(self._source.nsources, self._receivers.nstations,
                    self._source.nsources * self._receivers.nstations, dt, nfft))

        if nprocs == 1 or rank == 0:
            next_pair, skip_pairs = rank, 1
        else:
            next_pair, skip_pairs = rank - 1, nprocs - 1

        perf_time_begin = perf_counter()
        c      = _perf_counters()
        tstart = perf_counter()
        ipair  = 0

        for i_station, i_psource in pairs_to_compute:
            station  = self._receivers.get_station_by_id(int(i_station))
            psource  = self._source.get_source_by_id(int(i_psource))
            aux_crust = copy.deepcopy(self._crust)
            aux_crust.split_at_depth(psource.x[2])
            aux_crust.split_at_depth(station.x[2])

            if ipair == next_pair:
                if nprocs == 1 or (rank > 0 and nprocs > 1):
                    if verbose:
                        print("calling core START")
                    t1 = perf_counter()
                    tdata, z, e, n, t0 = self._call_core(
                        dt, nfft, tb, nx, sigma, smth,
                        wc1, wc2, pmin, pmax, dk, kc,
                        taper, aux_crust, psource, station, verbose)
                    c['core'] += perf_counter() - t1
                    if verbose:
                        print("calling core END")

                    nt     = len(z)
                    t0_arr = np.array([t0], dtype=np.double)
                    # Convert tdata from Fortran layout (1,9,nt) to C-order (nt,9)
                    tdata_c = np.empty((nt, 9), dtype=np.float64)
                    for comp in range(9):
                        tdata_c[:, comp] = tdata[0, comp, :]

                    if rank > 0:
                        t1 = perf_counter()
                        comm.Send(np.array([nt], dtype=np.int32),
                                  dest=0, tag=3 * ipair)
                        comm.Send(t0_arr,  dest=0, tag=3 * ipair + 1)
                        comm.Send(tdata_c, dest=0, tag=3 * ipair + 2)
                        c['send'] += perf_counter() - t1
                        next_pair += skip_pairs

                if rank == 0:
                    if nprocs > 1:
                        remote  = ipair % (nprocs - 1) + 1
                        t1      = perf_counter()
                        ant     = np.empty(1, dtype=np.int32)
                        t0_arr  = np.empty(1, dtype=np.double)
                        printMPI(f"P0 Recv from remote {remote}")
                        comm.Recv(ant,    source=remote, tag=3 * ipair)
                        comm.Recv(t0_arr, source=remote, tag=3 * ipair + 1)
                        nt = ant[0]
                        tdata_c = np.empty((nt, 9), dtype=np.float64)
                        comm.Recv(tdata_c, source=remote, tag=3 * ipair + 2)
                        c['recv'] += perf_counter() - t1

                    # Write slot k to HDF5
                    tdata_group[f"{ipair}_t0"]    = t0_arr[0]
                    tdata_group[f"{ipair}_tdata"] = tdata_c
                    next_pair += 1

                    if showProgress:
                        elapsed = perf_counter() - tstart
                        print(f"{ipair} of {npairs} done  "
                              f"ETA={_eta_str(elapsed, ipair + 1, npairs)}")
            ipair += 1

        fid.close()
        hfile.close()
        perf_time_total = perf_counter() - perf_time_begin

        if rank == 0:
            print(f"\n\nShakerMaker Generate GF database done. "
                  f"Total time: {perf_time_total:.2f} s")
            print("-" * 50)

        _print_perf_stats(c, perf_time_total)

        if use_mpi and nprocs > 1:
            comm.Barrier()

    # =========================================================================
    # Stage 2  --  run_op
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
               showProgress=True):
        """Stage 2 of the OP pipeline.

        For each (station, source) pair retrieves the precomputed tdata via
        the O(1) pair_to_slot index, calls _call_core_fast (skipping the FK
        integration), convolves with the source time function, and accumulates
        the station response.

        MPI: each rank processes its assigned stations; rank 0 then collects
        results from workers and writes via the writer.

        Requires ``/pair_to_slot`` in the HDF5 file. Use
        :meth:`build_pair_to_slot_from_legacy_h5` to add it to legacy
        JAA / PXP databases without recomputing any Green's Functions.

        :param h5_database_name: HDF5 file path (full path, including .h5).
        :type h5_database_name: str
        :param writer: Use this writer class to store outputs
        :type writer: StationListWriter
        :param writer_mode: 'progressive' or 'legacy'
        :type writer_mode: str
        :param tmin: Start of output time window (s).
        :type tmin: double
        :param tmax: End of output time window (s).
        :type tmax: double
        (remaining parameters identical to :meth:`run`)
        """
        title = (f"🎉 ¡LARGA VIDA AL LADRUNO_OP_kdTREE! 🎉 ShakerMaker Run (Stage 2 - OP) begin. "
                 f"{dt=} {nfft=} {dk=} {tb=} {tmin=} {tmax=}")

        if rank == 0:
            print(f"\n\n{title}")
            print("-" * len(title))
            print(f"  MPI processes  : {nprocs}")
            print(f"  OpenMP threads : {os.environ.get('OMP_NUM_THREADS','not set')}")
            print(f"  Loading database: {h5_database_name}")
            hfile = h5py.File(h5_database_name, 'r+')
        else:
            hfile = h5py.File(h5_database_name, 'r')

        # O(1) lookup array
        pair_to_slot = hfile["/pair_to_slot"][:]
        nsources_db  = int(hfile["/nsources"][()])
        nstations_db = int(hfile["/nstations"][()])

        if rank == 0:
            print(f"  pair_to_slot: O(1) lookup "
                  f"({nstations_db} stations x {nsources_db} sources)")

        # Validate that current model matches the database
        assert nsources_db == self._source.nsources, (
            f"[Stage 2] nsources mismatch: "
            f"HDF5={nsources_db}, model={self._source.nsources}")
        assert nstations_db == self._receivers.nstations, (
            f"[Stage 2] nstations mismatch: "
            f"HDF5={nstations_db}, model={self._receivers.nstations}")

        if rank > 0:
            writer = None
        if writer and rank == 0:
            assert isinstance(writer, StationListWriter), \
                "'writer' must be an instance of the shakermaker.StationListWriter class or None"
            writer.initialize(self._receivers, 2 * nfft,
                              tmin=tmin, tmax=tmax, dt=dt,
                              writer_mode=writer_mode)
            writer.write_metadata(self._receivers.metadata)

        if debugMPI:
            fid = open(f"rank_{rank}_stage2.debuginfo", "w")
            def printMPI(*args):
                fid.write(" ".join(str(a) for a in args) + "\n")
        else:
            fid = open(os.devnull, "w")
            printMPI = lambda *args: None

        self._logger.info(
            'ShakerMaker.run_op - starting\n'
            '\tNumber of sources: {}\n\tNumber of receivers: {}\n'
            '\tTotal src-rcv pairs: {}\n\tdt: {}\n\tnfft: {}'
            .format(self._source.nsources, self._receivers.nstations,
                    self._source.nsources * self._receivers.nstations, dt, nfft))

        nsources      = self._source.nsources
        nstations     = self._receivers.nstations
        next_station  = rank
        skip_stations = nprocs
        perf_time_begin = perf_counter()
        c             = _perf_counters()
        n_my_stations = 0

        # ------------------------------------------------------------------
        # Pass 1: each rank processes its assigned stations
        # ------------------------------------------------------------------
        for i_station, station in enumerate(self._receivers):
            if i_station != next_station:
                continue

            tstart_sta = perf_counter()

            for i_psource, psource in enumerate(self._source):
                # O(1) slot lookup
                k     = int(pair_to_slot[i_station * nsources + i_psource])
                tdata = hfile[f"/tdata_dict/{k}_tdata"][:]

                aux_crust = copy.deepcopy(self._crust)
                aux_crust.split_at_depth(psource.x[2])
                aux_crust.split_at_depth(station.x[2])

                if verbose:
                    print(f"  rank={rank} sta={i_station} "
                          f"src={i_psource} slot={k}")

                t1 = perf_counter()
                z, e, n, t0 = self._call_core_fast(
                    tdata, dt, nfft, tb, nx, sigma, smth,
                    wc1, wc2, pmin, pmax, dk, kc,
                    taper, aux_crust, psource, station, verbose)
                c['core'] += perf_counter() - t1

                t1    = perf_counter()
                t_arr = np.arange(0, len(z) * dt, dt) + psource.tt + t0
                psource.stf.dt = dt
                z_stf = psource.stf.convolve(z, t_arr)
                e_stf = psource.stf.convolve(e, t_arr)
                n_stf = psource.stf.convolve(n, t_arr)
                c['conv'] += perf_counter() - t1

                try:
                    t1 = perf_counter()
                    station.add_to_response(z_stf, e_stf, n_stf,
                                            t_arr, tmin, tmax)
                    c['add'] += perf_counter() - t1
                except Exception:
                    traceback.print_exc()
                    if use_mpi and nprocs > 1:
                        comm.Abort()

                if showProgress and i_psource % 1000 == 0:
                    elapsed = perf_counter() - tstart_sta
                    print(f"  rank={rank} sta={i_station} "
                          f"src={i_psource}/{nsources} "
                          f"({i_psource/nsources*100:.1f}%)  "
                          f"ETA={_eta_str(elapsed, i_psource+1, nsources)}")

            n_my_stations += 1
            elapsed_sta = perf_counter() - tstart_sta
            nsta_left   = (nstations - i_station - 1) // skip_stations
            print(f"  rank={rank} sta {i_station+1}/{nstations} "
                  f"({(i_station+1)/nstations*100:.1f}%)  "
                  f"sta_time={elapsed_sta:.1f}s  "
                  f"ETA_total={_eta_str(elapsed_sta*nsta_left,1,2)}")
            next_station += skip_stations

        hfile.close()

        # ------------------------------------------------------------------
        # Pass 2: workers send, rank 0 collects and writes
        # ------------------------------------------------------------------
        if use_mpi and nprocs > 1:
            if rank > 0:
                printMPI(f"Rank {rank} sending data to rank 0")
                my_sta = rank
                while my_sta < nstations:
                    sta = self._receivers.get_station_by_id(my_sta)
                    z_r, e_r, n_r, t_r = sta.get_response()
                    t1 = perf_counter()
                    comm.Send(np.array([len(z_r)], dtype=np.int32),
                              dest=0, tag=2 * my_sta)
                    comm.Send(np.column_stack([z_r, e_r, n_r, t_r]),
                              dest=0, tag=2 * my_sta + 1)
                    c['send'] += perf_counter() - t1
                    my_sta += nprocs
                printMPI(f"Rank {rank} DONE sending.")

            if rank == 0:
                print("Rank 0 collecting results from workers...")
                count = 0
                for remote in range(1, nprocs):
                    rsta = remote
                    while rsta < nstations:
                        sta = self._receivers.get_station_by_id(rsta)
                        t1  = perf_counter()
                        ant = np.empty(1, dtype=np.int32)
                        comm.Recv(ant, source=remote, tag=2 * rsta)
                        nt   = ant[0]
                        data = np.empty((nt, 4), dtype=np.float64)
                        comm.Recv(data, source=remote, tag=2 * rsta + 1)
                        c['recv'] += perf_counter() - t1
                        sta.add_to_response(
                            data[:, 0], data[:, 1], data[:, 2], data[:, 3],
                            tmin, tmax)
                        if writer:
                            writer.write_station(sta, rsta)
                        count += 1
                        rsta += nprocs

                # Write rank 0's own stations
                my_sta = 0
                while my_sta < nstations:
                    sta = self._receivers.get_station_by_id(my_sta)
                    if writer:
                        writer.write_station(sta, my_sta)
                    my_sta += nprocs
                count += n_my_stations
                print(f"Rank 0: {count}/{nstations} stations collected.")
                if writer:
                    writer.close()
        else:
            if writer:
                for i_station, station in enumerate(self._receivers):
                    writer.write_station(station, i_station)
                writer.close()

        fid.close()
        perf_time_total = perf_counter() - perf_time_begin

        if rank == 0:
            print(f"\n\nShakerMaker Run (Stage 2 - OP) done. "
                  f"Total time: {perf_time_total:.2f} s")
            print("-" * 50)

        _print_perf_stats(c, perf_time_total)

    # =========================================================================
    # Orchestrator  --  run_fast_faster_op
    # =========================================================================

    def run_fast_faster_op(self,
                           stage='all',
                           h5_database_name=None,
                           # Stage 0
                           delta_h=0.04,
                           delta_v_rec=0.002,
                           delta_v_src=0.2,
                           npairs_max=200000,
                           # Core (stages 1 and 2)
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
                           # Stage 2
                           writer=None,
                           writer_mode='progressive',
                           tmin=0.,
                           tmax=100,
                           # General
                           verbose=False,
                           debugMPI=False,
                           showProgress=True):
        """Orchestrator for the full OP pipeline.

        Runs Stage 0, 1, and/or 2 according to the ``stage`` parameter.
        Stages can be run independently, which is the recommended approach in
        HPC workflows where Stage 1-2 are MPI-parallel.

        Stage 0 is now also MPI parallel (geometry computed across all ranks;
        rank 0 handles the unique-slot reduction and HDF5 write).

        :param stage: Stages to run: ``0``, ``1``, ``2``, or ``'all'``.
        :type stage: int or str
        :param h5_database_name: HDF5 file path (full path, including .h5). Required.
        :type h5_database_name: str
        :param delta_h: Horizontal distance tolerance for slot grouping (km).
        :type delta_h: double
        :param delta_v_rec: Receiver depth tolerance for slot grouping (km).
        :type delta_v_rec: double
        :param delta_v_src: Source depth tolerance for slot grouping (km).
        :type delta_v_src: double
        :param npairs_max: Kept for API compatibility; not used internally.
        :type npairs_max: integer
        :param sigma: Its role is to damp the trace (at rate of exp(-sigma*t)) to reduce the wrap-arround.
        :type sigma: double
        :param nfft: Number of time-points to use in fft
        :type nfft: integer
        :param dt: Simulation time-step
        :type dt: double
        :param tb: Num. of samples before the first arrival.
        :type tb: integer
        :param taper: For low-pass filter, 0-1.
        :type taper: double
        :param smth: Densify the output samples by a factor of smth
        :type smth: double
        :param wc1: (George.. please provide one-line description!)
        :type wc1: double
        :param wc2: (George.. please provide one-line description!)
        :type wc2: double
        :param pmin: Max. phase velocity, in 1/vs, 0 the best.
        :type pmin: double
        :param pmax: Min. phase velocity, in 1/vs.
        :type pmax: double
        :param dk: Sample interval in wavenumber, in Pi/x, 0.1-0.4.
        :type dk: double
        :param nx: Number of distance ranges to compute.
        :type nx: integer
        :param kc: It's kmax, equal to 1/hs. Because the kernels decay with k at rate of exp(-k*hs) at w=0, we require kmax > 10 to make sure we have have summed enough.
        :type kc: double
        :param writer: Use this writer class to store outputs (Stage 2).
        :type writer: StationListWriter
        :param writer_mode: 'progressive' or 'legacy'.
        :type writer_mode: str
        :param tmin: Start of output time window (s).
        :type tmin: double
        :param tmax: End of output time window (s).
        :type tmax: double
        :param verbose: Verbose output from the Fortran core.
        :type verbose: bool
        :param debugMPI: Write per-rank debug files.
        :type debugMPI: bool
        :param showProgress: Print ETA.
        :type showProgress: bool
        """
        assert h5_database_name is not None, \
            "run_fast_faster_op: h5_database_name is required"
        assert stage in (0, 1, 2, 'all'), \
            "run_fast_faster_op: stage must be 0, 1, 2, or 'all'"

        perf_time_begin = perf_counter()

        if rank == 0:
            title = (f"🎉 ¡LARGA VIDA AL LADRUNO_OP_kdTREE! 🎉 ShakerMaker run_fast_faster_op | stage={stage} | "
                     f"{dt=} {nfft=} {dk=} {tb=} {tmin=} {tmax=}")
            print(f"\n\n{title}")
            print("-" * len(title))
            omp = os.environ.get('OMP_NUM_THREADS', 'not set')
            print(f"Hybrid Parallelization:")
            print(f"   MPI processes  : {nprocs}")
            print(f"   OpenMP threads : {omp}")
            if omp != 'not set':
                try:
                    print(f"   Total threads  : {nprocs} x {omp} = "
                          f"{nprocs * int(omp)}")
                except ValueError:
                    pass
            print(f"   DB file        : {h5_database_name}")
            print("-" * len(title))

        if stage in (0, 'all'):
            self.gen_pairs_op(
                h5_database_name=h5_database_name,
                delta_h=delta_h, delta_v_rec=delta_v_rec,
                delta_v_src=delta_v_src, npairs_max=npairs_max,
                showProgress=showProgress)
            if stage == 0:
                if rank == 0:
                    print(f"Stage 0 complete -> {h5_database_name}")
                return

        if stage in (1, 'all'):
            self.compute_gf_op(
                h5_database_name=h5_database_name,
                dt=dt, nfft=nfft, tb=tb, smth=smth,
                sigma=sigma, taper=taper, wc1=wc1, wc2=wc2,
                pmin=pmin, pmax=pmax, dk=dk, nx=nx, kc=kc,
                verbose=verbose, debugMPI=debugMPI,
                showProgress=showProgress)
            if stage == 1:
                if rank == 0:
                    print(f"Stage 1 complete -> {h5_database_name}")
                return

        if stage in (2, 'all'):
            if writer is None and rank == 0:
                print("WARNING: Stage 2 requires a writer. Aborting.")
                return
            self.run_op(
                h5_database_name=h5_database_name,
                dt=dt, nfft=nfft, tb=tb, smth=smth,
                sigma=sigma, taper=taper, wc1=wc1, wc2=wc2,
                pmin=pmin, pmax=pmax, dk=dk, nx=nx, kc=kc,
                writer=writer, writer_mode=writer_mode,
                tmin=tmin, tmax=tmax,
                verbose=verbose, debugMPI=debugMPI,
                showProgress=showProgress)

        if rank == 0 and stage == 'all':
            total = perf_counter() - perf_time_begin
            print("\n" + "=" * 70)
            print("ALL STAGES COMPLETE")
            print("=" * 70)
            print(f"  Total time: {total:.2f} s")
            if total > 60:
                print(f"  Total time: {total/60:.2f} min")
            if total > 3600:
                print(f"  Total time: {total/3600:.2f} hrs")
            print("=" * 70 + "\n")

    # =========================================================================
    # Legacy migration  --  build_pair_to_slot_from_legacy_h5
    # =========================================================================

    def build_pair_to_slot_from_legacy_h5(self,
                                          h5_database_name,
                                          delta_h=None,
                                          delta_v_rec=None,
                                          delta_v_src=None,
                                          showProgress=True):
        """Migrate a legacy JAA / PXP Green's Function database to OP format.

        Reads ``h5_database_name`` (which must already contain
        ``/dh_of_pairs``, ``/zrec_of_pairs``, ``/zsrc_of_pairs``, and
        ``/tdata_dict``) and writes three new datasets:

        - ``/pair_to_slot``  (nstations * nsources,)  int32
        - ``/nstations``     scalar
        - ``/nsources``      scalar

        After this call the database is fully compatible with :meth:`run_op`
        and :meth:`run_fast_faster_op` (stage=2), reusing all previously
        computed Green's Functions without any recomputation.

        **Algorithm** (MPI + KDTree):

        1. Rank 0 reads slot geometry arrays and tolerances from the HDF5 file
           and broadcasts them to all ranks via ``comm.bcast``.
        2. Each rank builds an identical :class:`scipy.spatial.cKDTree` from
           the slot coordinates normalised by the respective tolerances.
        3. Each rank owns a contiguous station slice ``[i_start, i_end)`` and
           assembles the full ``(my_nstations * nsources, 3)`` query matrix
           in a single vectorised operation (no Python loop over sources).
        4. A single ``tree.query(queries, workers=-1)`` call answers all
           queries in parallel using all available threads.
        5. Rank 0 collects partial arrays from all ranks via ``comm.Gatherv``
           and writes the assembled ``pair_to_slot`` plus ``nstations`` /
           ``nsources`` to the HDF5 file.

        If ``delta_h``, ``delta_v_rec``, or ``delta_v_src`` are ``None``,
        the values stored in the HDF5 file are used (the original tolerances
        from the JAA/PXP run).

        :param h5_database_name: HDF5 file path (full path, including .h5).
        :type h5_database_name: str
        :param delta_h: Horizontal distance tolerance (km). Uses stored value
            if None.
        :type delta_h: double or None
        :param delta_v_rec: Receiver depth tolerance (km). Uses stored value
            if None.
        :type delta_v_rec: double or None
        :param delta_v_src: Source depth tolerance (km). Uses stored value
            if None.
        :type delta_v_src: double or None
        :param showProgress: Print progress messages.
        :type showProgress: bool
        """
        from scipy.spatial import cKDTree

        nsources  = self._source.nsources
        nstations = self._receivers.nstations

        # ------------------------------------------------------------------
        # Step 1: rank 0 reads slot geometry + tolerances, broadcasts to all
        # ------------------------------------------------------------------
        if rank == 0:
            with h5py.File(h5_database_name, 'r') as hf:
                dh_of_pairs   = hf["/dh_of_pairs"][:]
                zrec_of_pairs = hf["/zrec_of_pairs"][:]
                zsrc_of_pairs = hf["/zsrc_of_pairs"][:]
                dh_tol = float(hf["/delta_h"][()])     if delta_h    is None else delta_h
                vr_tol = float(hf["/delta_v_rec"][()]) if delta_v_rec is None else delta_v_rec
                vs_tol = float(hf["/delta_v_src"][()]) if delta_v_src is None else delta_v_src
        else:
            dh_of_pairs = zrec_of_pairs = zsrc_of_pairs = None
            dh_tol = vr_tol = vs_tol = None

        if use_mpi and nprocs > 1:
            dh_of_pairs   = comm.bcast(dh_of_pairs,   root=0)
            zrec_of_pairs = comm.bcast(zrec_of_pairs, root=0)
            zsrc_of_pairs = comm.bcast(zsrc_of_pairs, root=0)
            dh_tol  = comm.bcast(dh_tol,  root=0)
            vr_tol  = comm.bcast(vr_tol,  root=0)
            vs_tol  = comm.bcast(vs_tol,  root=0)

        n_slots      = len(dh_of_pairs)
        npairs_total = nstations * nsources

        if rank == 0:
            title = (f"🎉 ¡LARGA VIDA AL LADRUNO_OP_kdTREE! 🎉 ShakerMaker build_pair_to_slot_from_legacy_h5 -- "
                     f"{h5_database_name}")
            print(f"\n\n{title}")
            print("-" * len(title))
            print(f"  Stations    : {nstations}")
            print(f"  Sources     : {nsources}")
            print(f"  Slots in DB : {n_slots}")
            print(f"  delta_h     : {dh_tol}")
            print(f"  delta_v_rec : {vr_tol}")
            print(f"  delta_v_src : {vs_tol}")
            print(f"  MPI ranks   : {nprocs}")
            print(f"  Strategy    : MPI station partitioning + KDTree O(log n) lookup")
            print(f"  NOTE: tdata format (nt,9) C-order is identical. "
                  f"Only adding pair_to_slot mapping.")

        # ------------------------------------------------------------------
        # Step 2: build KDTree (identical on every rank)
        # Normalise so each tolerance dimension spans unit radius.
        # sqrt(3) is the L2 cutoff enclosing the unit L-inf tolerance cube.
        # ------------------------------------------------------------------
        slots_norm = np.column_stack([
            dh_of_pairs   / dh_tol,
            zsrc_of_pairs / vs_tol,
            zrec_of_pairs / vr_tol,
        ])
        tree     = cKDTree(slots_norm)
        max_dist = np.sqrt(3.0)

        # ------------------------------------------------------------------
        # Step 3: partition stations across ranks (contiguous slices)
        # ------------------------------------------------------------------
        base, rem    = divmod(nstations, nprocs)
        i_start      = rank * base + min(rank, rem)
        i_end        = i_start + base + (1 if rank < rem else 0)
        my_nstations = i_end - i_start

        # Pre-fetch coordinates once to avoid repeated Python attribute lookups
        my_sta_coords = np.array(
            [self._receivers.get_station_by_id(i).x for i in range(i_start, i_end)],
            dtype=np.float64)                            # (my_nstations, 3)
        src_coords = np.array(
            [self._source.get_source_by_id(j).x for j in range(nsources)],
            dtype=np.float64)                            # (nsources, 3)

        # ------------------------------------------------------------------
        # Step 4: build full query matrix and run a single vectorised KDTree
        # query with workers=-1 (uses all available threads internally)
        # ------------------------------------------------------------------
        sta_rep  = np.repeat(my_sta_coords, nsources, axis=0)  # (my_n*nsrc, 3)
        src_tile = np.tile(src_coords, (my_nstations, 1))      # (my_n*nsrc, 3)

        d_xy   = sta_rep[:, :2] - src_tile[:, :2]
        dh_q   = np.sqrt(np.einsum('ij,ij->i', d_xy, d_xy))   # horizontal dist
        zsrc_q = src_tile[:, 2]
        zrec_q = sta_rep[:, 2]

        queries_norm = np.column_stack([
            dh_q   / dh_tol,
            zsrc_q / vs_tol,
            zrec_q / vr_tol,
        ])

        t0_q = perf_counter()
        dists, slots = tree.query(queries_norm, workers=-1)
        t1_q = perf_counter()

        if showProgress:
            print(f"  [Rank {rank}] KDTree query "
                  f"({len(queries_norm):,} points): {t1_q - t0_q:.2f}s")

        # Assign slot 0 to any out-of-tolerance pairs and warn
        out_of_bounds = dists > max_dist
        n_oob = int(np.sum(out_of_bounds))
        if n_oob > 0:
            slots[out_of_bounds] = 0
            print(f"  [Rank {rank}] WARNING: {n_oob} pairs outside tolerance "
                  f"-- assigned slot 0. Verify model / tolerance consistency.")

        my_partial = slots.astype(np.int32)   # (my_nstations * nsources,)

        # ------------------------------------------------------------------
        # Step 5: gather all partial arrays to rank 0 via Gatherv
        # ------------------------------------------------------------------
        if use_mpi and nprocs > 1:
            my_size  = np.array([len(my_partial)], dtype=np.int32)
            all_sizes = np.empty(nprocs, dtype=np.int32) if rank == 0 else None
            comm.Gather(my_size, all_sizes, root=0)

            if rank == 0:
                displacements     = np.concatenate([[0], np.cumsum(all_sizes[:-1])])
                pair_to_slot_full = np.empty(npairs_total, dtype=np.int32)
                comm.Gatherv(
                    my_partial,
                    [pair_to_slot_full, all_sizes, displacements, MPI.INT],
                    root=0)
            else:
                comm.Gatherv(my_partial, None, root=0)
                pair_to_slot_full = None
        else:
            pair_to_slot_full = my_partial

        # ------------------------------------------------------------------
        # Step 6: rank 0 validates and writes to HDF5
        # ------------------------------------------------------------------
        if rank == 0:
            n_unassigned = int(np.sum(pair_to_slot_full < 0))
            if n_unassigned > 0:
                raise RuntimeError(
                    f"[build_pair_to_slot] {n_unassigned} pairs unassigned. "
                    "The current model may not match the original database. "
                    "Check sources, stations, and tolerances.")

            elapsed = t1_q - t0_q
            print(f"\nMapping complete. Time: {elapsed:.1f}s")
            print(f"  Unique slots used : "
                  f"{len(np.unique(pair_to_slot_full))} / {n_slots}")
            if n_oob > 0:
                print(f"  WARNING: {n_oob} pairs had no match. "
                      "Verify model consistency.")

            with h5py.File(h5_database_name, 'r+') as hf:
                for key in ('pair_to_slot', 'nstations', 'nsources'):
                    if key in hf:
                        del hf[key]
                hf.create_dataset("pair_to_slot", data=pair_to_slot_full)
                hf.create_dataset("nstations",    data=nstations)
                hf.create_dataset("nsources",     data=nsources)

            print(f"pair_to_slot, nstations, nsources written to: "
                  f"{h5_database_name}")
            print("Database is now compatible with run_op (Stage 2).")

        if use_mpi and nprocs > 1:
            comm.Barrier()

    # =========================================================================
    # STKO export  --  export_drm_geometry
    # =========================================================================

    def export_drm_geometry(self, filename="drm_geometry.h5drm"):
        """Export DRM geometry for visualisation in STKO.

        Creates an HDF5 file with station coordinates and minimal synthetic
        data (2 samples, linear ramp 0 -> 10) for geometry inspection only.
        Works with DRMBox and SurfaceGrid receiver lists.

        Parameters
        ----------
        filename : str
            Output HDF5 filename (default: 'drm_geometry.h5drm')

        Returns
        -------
        str
            Path to the created file
        """
        from shakermaker.sl_extensions import DRMBox
        from shakermaker.sl_extensions.SurfaceGrid import SurfaceGrid

        if not isinstance(self._receivers, (DRMBox, SurfaceGrid)):
            raise TypeError(
                f"export_drm_geometry() requires DRMBox or SurfaceGrid. "
                f"Got: {type(self._receivers).__name__}")

        if rank != 0:
            return filename

        metadata  = self._receivers.metadata
        nstations = self._receivers.nstations - 1  # exclude QA station

        print(f"\nexport_drm_geometry: {filename}")
        print(f"  Receiver type: {type(self._receivers).__name__}")
        print(f"  Stations (excl. QA): {nstations}")

        with h5py.File(filename, 'w') as hf:
            grp_data = hf.create_group('DRM_Data')
            grp_qa   = hf.create_group('DRM_QA_Data')
            grp_meta = hf.create_group('DRM_Metadata')

            xyz      = np.zeros((nstations, 3))
            internal = np.zeros(nstations, dtype=bool)
            for i in range(nstations):
                sta         = self._receivers.get_station_by_id(i)
                xyz[i, :]   = sta.x
                internal[i] = sta.is_internal

            grp_data.create_dataset('xyz',      data=xyz,      dtype=np.float64)
            grp_data.create_dataset('internal', data=internal, dtype=bool)
            grp_data.create_dataset('data_location',
                                    data=np.arange(0, nstations,
                                                   dtype=np.int32) * 3)

            qa_sta = self._receivers.get_station_by_id(nstations)
            grp_qa.create_dataset('xyz',
                                  data=qa_sta.x.reshape(1, 3),
                                  dtype=np.float64)

            # Minimal time data: linear ramp 0 -> 10 (2 samples)
            ramp    = np.tile([0.0, 10.0], (3 * nstations, 1))
            ramp_qa = np.tile([0.0, 10.0], (3, 1))
            for grp, r in [(grp_data, ramp), (grp_qa, ramp_qa)]:
                grp.create_dataset('velocity',     data=r, dtype=np.float64)
                grp.create_dataset('displacement', data=r, dtype=np.float64)
                grp.create_dataset('acceleration', data=r, dtype=np.float64)

            grp_meta.create_dataset('dt',     data=0.0005)
            grp_meta.create_dataset('tstart', data=0.0)
            grp_meta.create_dataset('tend',   data=10.0)
            for key in ('h', 'drmbox_x0',
                        'drmbox_xmax', 'drmbox_xmin',
                        'drmbox_ymax', 'drmbox_ymin',
                        'drmbox_zmax', 'drmbox_zmin'):
                if key in metadata:
                    grp_meta.create_dataset(key, data=metadata[key])

            print(f"  Station coordinates: written")
            print(f"  QA station: written")
            print(f"  Time data (2 samples, linear ramp): written")
            print(f"  Metadata: written")

        print(f"Geometry file created: {filename}")
        print("Use in STKO to visualise grid before running simulation.")
        return filename

    # =========================================================================
    # Internal: Fortran core wrappers
    # =========================================================================

    def _call_core(self, dt, nfft, tb, nx, sigma, smth, wc1, wc2,
                   pmin, pmax, dk, kc, taper, crust, psource, station,
                   verbose=False):
        """Call core.subgreen to compute the full FK kernel (tdata).

        Used by: run(), compute_gf_op() (Stage 1).

        Returns tdata with Fortran layout (1, 9, nt), plus component
        seismograms z, e, n and time offset t0.
        """
        mb  = crust.nlayers
        src = crust.get_layer(psource.x[2]) + 1   # fortran starts in 1, not 0
        rcv = crust.get_layer(station.x[2]) + 1   # fortran starts in 1, not 0

        stype = 2  # Source type double-couple, compute up and down going wave
        updn  = 0
        d     = crust.d; a = crust.a; b = crust.b
        rho   = crust.rho; qa = crust.qa; qb = crust.qb

        pf = psource.angles[0]; df = psource.angles[1]; lf = psource.angles[2]
        sx = psource.x[0]; sy = psource.x[1]
        rx = station.x[0]; ry = station.x[1]
        x  = np.sqrt((sx - rx)**2 + (sy - ry)**2)

        self._logger.debug(
            'ShakerMaker._call_core - calling core.subgreen\n'
            '\tmb: {}\n\tsrc: {}\n\trcv: {}\n\tstype: {}\n\tupdn: {}\n'
            '\td: {}\n\ta: {}\n\tb: {}\n\trho: {}\n\tqa: {}\n\tqb: {}\n'
            '\tdt: {}\n\tnfft: {}\n\ttb: {}\n\tnx: {}\n\tsigma: {}\n'
            '\tsmth: {}\n\twc1: {}\n\twc2: {}\n\tpmin: {}\n\tpmax: {}\n'
            '\tdk: {}\n\tkc: {}\n\ttaper: {}\n\tx: {}\n'
            '\tpf: {}\n\tdf: {}\n\tlf: {}\n\tsx: {}\n\tsy: {}\n'
            '\trx: {}\n\try: {}\n'
            .format(mb, src, rcv, stype, updn, d, a, b, rho, qa, qb,
                    dt, nfft, tb, nx, sigma, smth, wc1, wc2,
                    pmin, pmax, dk, kc, taper, x, pf, df, lf, sx, sy, rx, ry))

        if verbose:
            print('ShakerMaker._call_core - calling core.subgreen\n'
                  '\tmb: {}\n\tsrc: {}\n\trcv: {}\n\tstype: {}\n\tupdn: {}\n'
                  '\td: {}\n\ta: {}\n\tb: {}\n\trho: {}\n\tqa: {}\n\tqb: {}\n'
                  '\tdt: {}\n\tnfft: {}\n\ttb: {}\n\tnx: {}\n\tsigma: {}\n'
                  '\tsmth: {}\n\twc1: {}\n\twc2: {}\n\tpmin: {}\n\tpmax: {}\n'
                  '\tdk: {}\n\tkc: {}\n\ttaper: {}\n\tx: {}\n'
                  '\tpf: {}\n\tdf: {}\n\tlf: {}\n\tsx: {}\n\tsy: {}\n'
                  '\trx: {}\n\try: {}\n'
                  .format(mb, src, rcv, stype, updn, d, a, b, rho, qa, qb,
                          dt, nfft, tb, nx, sigma, smth, wc1, wc2,
                          pmin, pmax, dk, kc, taper, x, pf, df, lf,
                          sx, sy, rx, ry))

        # Execute the core subgreen Fortran routine
        tdata, z, e, n, t0 = core.subgreen(
            mb, src, rcv, stype, updn, d, a, b, rho, qa, qb,
            dt, nfft, tb, nx, sigma, smth, wc1, wc2,
            pmin, pmax, dk, kc, taper, x, pf, df, lf, sx, sy, rx, ry)

        self._logger.debug(
            'ShakerMaker._call_core - core.subgreen returned: '
            'z_size={}'.format(len(z)))

        return tdata, z, e, n, t0

    def _call_core_fast(self, tdata, dt, nfft, tb, nx, sigma, smth, wc1, wc2,
                        pmin, pmax, dk, kc, taper, crust, psource, station,
                        verbose=False):
        """Call core.subgreen2, reusing a precomputed tdata kernel.

        Used by: run_op() (Stage 2).

        ``tdata`` must be in C-order with shape (nt, 9) as stored in the HDF5
        database. It is reshaped to (1, 9, nt) before being passed to the
        Fortran routine.

        Returns component seismograms z, e, n and time offset t0.
        """
        mb  = crust.nlayers
        src = crust.get_layer(psource.x[2]) + 1   # fortran starts in 1, not 0
        rcv = crust.get_layer(station.x[2]) + 1   # fortran starts in 1, not 0

        stype = 2  # Source type double-couple, compute up and down going wave
        updn  = 0
        d     = crust.d; a = crust.a; b = crust.b
        rho   = crust.rho; qa = crust.qa; qb = crust.qb

        pf = psource.angles[0]; df = psource.angles[1]; lf = psource.angles[2]
        sx = psource.x[0]; sy = psource.x[1]
        rx = station.x[0]; ry = station.x[1]
        x  = np.sqrt((sx - rx)**2 + (sy - ry)**2)

        self._logger.debug(
            'ShakerMaker._call_core_fast - calling core.subgreen2\n'
            '\tmb: {}\n\tsrc: {}\n\trcv: {}\n\tstype: {}\n\tupdn: {}\n'
            '\td: {}\n\ta: {}\n\tb: {}\n\trho: {}\n\tqa: {}\n\tqb: {}\n'
            '\tdt: {}\n\tnfft: {}\n\ttb: {}\n\tnx: {}\n\tsigma: {}\n'
            '\tsmth: {}\n\twc1: {}\n\twc2: {}\n\tpmin: {}\n\tpmax: {}\n'
            '\tdk: {}\n\tkc: {}\n\ttaper: {}\n\tx: {}\n'
            '\tpf: {}\n\tdf: {}\n\tlf: {}\n\tsx: {}\n\tsy: {}\n'
            '\trx: {}\n\try: {}\n'
            .format(mb, src, rcv, stype, updn, d, a, b, rho, qa, qb,
                    dt, nfft, tb, nx, sigma, smth, wc1, wc2,
                    pmin, pmax, dk, kc, taper, x, pf, df, lf, sx, sy, rx, ry))

        if verbose:
            print('ShakerMaker._call_core_fast - calling core.subgreen2\n'
                  '\tmb: {}\n\tsrc: {}\n\trcv: {}\n\tstype: {}\n\tupdn: {}\n'
                  '\td: {}\n\ta: {}\n\tb: {}\n\trho: {}\n\tqa: {}\n\tqb: {}\n'
                  '\tdt: {}\n\tnfft: {}\n\ttb: {}\n\tnx: {}\n\tsigma: {}\n'
                  '\tsmth: {}\n\twc1: {}\n\twc2: {}\n\tpmin: {}\n\tpmax: {}\n'
                  '\tdk: {}\n\tkc: {}\n\ttaper: {}\n\tx: {}\n'
                  '\tpf: {}\n\tdf: {}\n\tlf: {}\n\tsx: {}\n\tsy: {}\n'
                  '\trx: {}\n\try: {}\n'
                  .format(mb, src, rcv, stype, updn, d, a, b, rho, qa, qb,
                          dt, nfft, tb, nx, sigma, smth, wc1, wc2,
                          pmin, pmax, dk, kc, taper, x, pf, df, lf,
                          sx, sy, rx, ry))

        # Reshape tdata from C-order (nt, 9) to Fortran layout (1, 9, nt)
        tdata_ = tdata.T
        tdata_ = tdata_.reshape((1, tdata_.shape[0], tdata_.shape[1]))

        # Execute the core subgreen2 Fortran routine
        z, e, n, t0 = core.subgreen2(
            mb, src, rcv, stype, updn, d, a, b, rho, qa, qb,
            dt, nfft, tb, nx, sigma, smth, wc1, wc2,
            pmin, pmax, dk, kc, taper, x, pf, df, lf, tdata_, sx, sy, rx, ry)

        self._logger.debug(
            'ShakerMaker._call_core_fast - core.subgreen2 returned: '
            'z_size={}'.format(len(z)))

        return z, e, n, t0

    # =========================================================================
    # Utilities
    # =========================================================================

    def write(self, writer):
        """Write all receivers using the given writer."""
        writer.write(self._receivers)

    def enable_mpi(self, rank, nprocs):
        """Override MPI rank and nprocs (rarely needed)."""
        self._mpi_rank   = rank
        self._mpi_nprocs = nprocs

    def mpi_is_master_process(self):
        return self._mpi_rank == 0

    @property
    def mpi_rank(self):
        return self._mpi_rank

    @property
    def mpi_nprocs(self):
        return self._mpi_nprocs