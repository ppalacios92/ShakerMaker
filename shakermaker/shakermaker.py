"""
shakermaker.py - ShakerMaker nueva arquitectura OP.

Pipeline en 3 etapas con O(1) lookup via pair_to_slot:

  Stage 0 - gen_pairs_op:       identifica slots únicos (dh, z_src, z_rec)
                                 construye pair_to_slot[i_sta*nsrc + i_src] = k
  Stage 1 - compute_gf_op:      calcula GF (tdata) para cada slot k
  Stage 2 - run_op:             convuelve GF con STF usando lookup O(1)

Orquestador:
  run_fast_faster_op(stage=0|1|2|'all')

Debug/validación:
  run()                         método JAA original, sin base de datos

Visualización STKO:
  export_drm_geometry()
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
except ImportError:
    rank = 0
    nprocs = 1
    use_mpi = False


# ---------------------------------------------------------------------------
# Helpers de performance (compartidos por todos los métodos)
# ---------------------------------------------------------------------------

def _perf_counters():
    """Retorna dict de contadores de tiempo inicializados en cero."""
    keys = ['core', 'send', 'recv', 'conv', 'add']
    return {k: np.zeros(1, dtype=np.double) for k in keys}


def _print_perf_stats(counters, perf_time_total):
    """Imprime estadísticas MPI de performance (solo rank 0)."""
    if not (use_mpi and nprocs > 1):
        return
    labels = {'core': 'time_core', 'send': 'time_send',
              'recv': 'time_recv', 'conv': 'time_conv', 'add': 'time_add'}
    stats = {}
    for k, arr in counters.items():
        mx = np.array([-np.inf])
        mn = np.array([ np.inf])
        comm.Reduce(arr, mx, op=MPI.MAX, root=0)
        comm.Reduce(arr, mn, op=MPI.MIN, root=0)
        stats[k] = (mx[0], mn[0])

    if rank == 0:
        print("\nPerformance statistics (all MPI processes):")
        for k, (mx, mn) in stats.items():
            pct_mx = mx / perf_time_total * 100 if perf_time_total > 0 else 0
            pct_mn = mn / perf_time_total * 100 if perf_time_total > 0 else 0
            print(f"  {labels[k]:12s}: max={mx:.3f}s ({pct_mx:.2f}%)  "
                  f"min={mn:.3f}s ({pct_mn:.2f}%)")


def _eta_str(elapsed, done, total):
    """Retorna string 'H:MM:SS' con el ETA estimado."""
    if done == 0:
        return "??:??:??"
    remaining = elapsed / done * (total - done)
    hh = int(remaining) // 3600
    mm = (int(remaining) % 3600) // 60
    ss = remaining % 60
    return f"{hh}:{mm:02d}:{ss:04.1f}"


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class ShakerMaker:
    """
    ShakerMaker - nueva arquitectura OP.

    Uso típico::

        model = ShakerMaker(crust, source, receivers)

        # Pipeline completo (recomendado)
        model.run_fast_faster_op(
            stage='all',
            h5_database_name='mi_db',
            writer=mi_writer,
            dt=0.05, nfft=4096, ...
        )

        # O por etapas
        model.run_fast_faster_op(stage=0, h5_database_name='mi_db')
        model.run_fast_faster_op(stage=1, h5_database_name='mi_db', dt=0.05)
        model.run_fast_faster_op(stage=2, h5_database_name='mi_db', writer=mi_writer)

        # Debug/validación (sin base de datos)
        model.run(dt=0.05, writer=mi_writer)

    :param crust: Modelo cortical.
    :type crust: CrustModel
    :param source: Fuente sísmica.
    :type source: FaultSource
    :param receivers: Lista de estaciones receptoras.
    :type receivers: StationList
    """

    def __init__(self, crust, source, receivers):
        assert isinstance(crust, CrustModel), \
            "crust must be an instance of shakermaker.CrustModel"
        assert isinstance(source, FaultSource), \
            "source must be an instance of shakermaker.FaultSource"
        assert isinstance(receivers, StationList), \
            "receivers must be an instance of shakermaker.StationList"

        self._crust     = crust
        self._source    = source
        self._receivers = receivers
        self._mpi_rank  = rank
        self._mpi_nprocs = nprocs
        self._logger    = logging.getLogger(__name__)

    # =========================================================================
    # MÉTODO DE DEBUG: run() — JAA puro, sin base de datos
    # =========================================================================

    def run(self,
            dt=0.05, nfft=4096, tb=1000,
            smth=1, sigma=2, taper=0.9,
            wc1=1, wc2=2, pmin=0, pmax=1,
            dk=0.3, nx=1, kc=15.0,
            writer=None, writer_mode='progressive',
            verbose=False, debugMPI=False,
            tmin=0., tmax=100, showProgress=True):
        """
        Simulación directa, par a par. Sin base de datos.

        Útil para debug y validación de resultados contra el pipeline OP.
        Calcula cada par (fuente, receptor) independientemente → no reutiliza GFs.

        :param dt: Paso de tiempo (s)
        :param nfft: Número de puntos FFT
        :param tb: Muestras antes de la primera llegada
        :param smth: Factor de densificación de salida
        :param sigma: Factor de amortiguamiento (exp(-sigma*t))
        :param taper: Filtro pasa-bajo (0-1)
        :param wc1, wc2: Frecuencias de corte del filtro
        :param pmin, pmax: Límites de velocidad de fase (1/vs)
        :param dk: Intervalo en número de onda (Pi/x)
        :param nx: Número de rangos de distancia
        :param kc: Número de onda máximo (1/hs)
        :param writer: StationListWriter para guardar resultados
        :param verbose: Salida detallada del core Fortran
        :param debugMPI: Escribe archivos de debug por rank
        :param tmin, tmax: Ventana de tiempo de salida (s)
        :param showProgress: Imprime progreso en rank 0
        """
        title = f"[run] ShakerMaker. {dt=} {nfft=} {dk=} {tb=} {tmin=} {tmax=}"

        if rank == 0:
            print(f"\n{'='*len(title)}")
            print(title)
            print(f"{'='*len(title)}")
            print(f"  MPI processes : {nprocs}")
            print(f"  OMP threads   : {os.environ.get('OMP_NUM_THREADS', 'not set')}")
            print(f"  Sources       : {self._source.nsources}")
            print(f"  Stations      : {self._receivers.nstations}")
            print(f"{'='*len(title)}")

        perf_time_begin = perf_counter()
        c = _perf_counters()

        if debugMPI:
            fid = open(f"rank_{rank}_run.debuginfo", "w")
            def printMPI(*args): fid.write(" ".join(str(a) for a in args) + "\n")
        else:
            fid = open(os.devnull, "w")
            printMPI = lambda *args: None

        self._logger.info(
            f'ShakerMaker.run: {self._source.nsources} sources, '
            f'{self._receivers.nstations} stations, dt={dt}, nfft={nfft}')

        if rank > 0:
            writer = None
        if writer and rank == 0:
            assert isinstance(writer, StationListWriter)
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
                    if nprocs == 1 or (rank > 0 and nprocs > 1):
                        t1 = perf_counter()
                        tdata, z, e, n, t0 = self._call_core(
                            dt, nfft, tb, nx, sigma, smth,
                            wc1, wc2, pmin, pmax, dk, kc,
                            taper, aux_crust, psource, station, verbose)
                        c['core'] += perf_counter() - t1

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
                            comm.Send(np.array([nt], dtype=np.int32), dest=0, tag=2 * ipair)
                            data = np.column_stack([z_stf, e_stf, n_stf, t])
                            comm.Send(data, dest=0, tag=2 * ipair + 1)
                            c['send'] += perf_counter() - t1
                            next_pair += skip_pairs

                    if rank == 0:
                        if nprocs > 1:
                            remote = ipair % (nprocs - 1) + 1
                            t1 = perf_counter()
                            ant = np.empty(1, dtype=np.int32)
                            comm.Recv(ant, source=remote, tag=2 * ipair)
                            nt = ant[0]
                            data = np.empty((nt, 4), dtype=np.float64)
                            comm.Recv(data, source=remote, tag=2 * ipair + 1)
                            z_stf, e_stf, n_stf, t = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
                            c['recv'] += perf_counter() - t1

                        next_pair += 1
                        try:
                            t1 = perf_counter()
                            station.add_to_response(z_stf, e_stf, n_stf, t, tmin, tmax)
                            c['add'] += perf_counter() - t1
                        except Exception:
                            traceback.print_exc()
                            if use_mpi and nprocs > 1:
                                comm.Abort()

                        if showProgress:
                            elapsed = perf_counter() - tstart
                            print(f"  [run] {ipair+1}/{npairs}  ETA={_eta_str(elapsed, ipair+1, npairs)}"
                                  f"  t=[{t[0]:.3f}, {t[-1]:.3f}]")

                ipair += 1

            if writer and rank == 0:
                writer.write_station(station, i_station)

        if writer and rank == 0:
            writer.close()

        fid.close()
        perf_time_total = perf_counter() - perf_time_begin

        if rank == 0:
            print(f"\n[run] Done. Total time: {perf_time_total:.2f} s")
            print("-" * 50)

        _print_perf_stats(c, perf_time_total)

    # =========================================================================
    # STAGE 0: Construir slots únicos + mapping pair_to_slot
    # =========================================================================

    def gen_pairs_op(self,
                     h5_database_name,
                     delta_h=0.04,
                     delta_v_rec=0.002,
                     delta_v_src=0.2,
                     npairs_max=200000,
                     showProgress=True):
        """
        Stage 0: Identifica geometrías únicas (dh, z_src, z_rec) y construye
        el mapping pair_to_slot[i_station * nsources + i_psource] = k.

        Solo corre en rank 0 (serial). Los demás ranks esperan en Barrier.

        Guarda en ``h5_database_name.h5``:

        - ``pairs_to_compute[k, 2]``  : par representativo del slot k
        - ``dh_of_pairs[k]``          : distancia horizontal del slot k
        - ``dv_of_pairs[k]``          : distancia vertical del slot k
        - ``zrec_of_pairs[k]``        : profundidad receptor del slot k
        - ``zsrc_of_pairs[k]``        : profundidad fuente del slot k
        - ``pair_to_slot[nsta*nsrc]`` : mapping plano → slot
        - ``delta_h, delta_v_rec, delta_v_src, nstations, nsources``

        :param h5_database_name: Ruta HDF5 (sin extensión .h5)
        :param delta_h: Tolerancia distancia horizontal (km)
        :param delta_v_rec: Tolerancia profundidad receptor (km)
        :param delta_v_src: Tolerancia profundidad fuente (km)
        :param npairs_max: Máximo de slots únicos pre-asignados
        :param showProgress: Imprime progreso cada 50k pares
        """
        if rank != 0:
            if use_mpi and nprocs > 1:
                comm.Barrier()
            return

        nsources   = self._source.nsources
        nstations  = self._receivers.nstations
        npairs_total = nstations * nsources

        print(f"\n{'='*70}")
        print(f"STAGE 0: gen_pairs_op")
        print(f"{'='*70}")
        print(f"  Stations     : {nstations}")
        print(f"  Sources      : {nsources}")
        print(f"  Total pairs  : {npairs_total}")
        print(f"  delta_h      : {delta_h} km")
        print(f"  delta_v_rec  : {delta_v_rec} km")
        print(f"  delta_v_src  : {delta_v_src} km")
        print(f"  Max slots    : {npairs_max}")
        if nprocs > 1:
            print(f"  ⚠  Stage 0 es SERIAL — {nprocs-1} proceso(s) MPI inactivo(s)")
        print(f"{'='*70}")

        t0_start = perf_counter()

        # Pre-allocate slot arrays
        pairs_to_compute = np.empty((npairs_max, 2), dtype=np.int32)
        dh_of_pairs      = np.empty(npairs_max, dtype=np.float64)
        dv_of_pairs      = np.empty(npairs_max, dtype=np.float64)
        zrec_of_pairs    = np.empty(npairs_max, dtype=np.float64)
        zsrc_of_pairs    = np.empty(npairs_max, dtype=np.float64)

        # Mapping plano: (i_station * nsources + i_psource) → slot k
        pair_to_slot = np.full(npairs_total, -1, dtype=np.int32)

        n_slots = 0

        for i_station, station in enumerate(self._receivers):
            z_rec = station.x[2]
            for i_psource, psource in enumerate(self._source):
                z_src = psource.x[2]
                d  = station.x - psource.x
                dh = np.sqrt(d[0]**2 + d[1]**2)
                dv = abs(d[2])

                flat_idx = i_station * nsources + i_psource

                if n_slots == 0:
                    k = 0
                    pairs_to_compute[0] = [i_station, i_psource]
                    dh_of_pairs[0]   = dh
                    dv_of_pairs[0]   = dv
                    zrec_of_pairs[0] = z_rec
                    zsrc_of_pairs[0] = z_src
                    n_slots = 1
                else:
                    # Chequeo vectorizado contra todos los slots existentes
                    not_covered = (
                        (np.abs(dh    - dh_of_pairs[:n_slots])   > delta_h)   |
                        (np.abs(z_src - zsrc_of_pairs[:n_slots]) > delta_v_src) |
                        (np.abs(z_rec - zrec_of_pairs[:n_slots]) > delta_v_rec)
                    )

                    if np.all(not_covered):
                        # Geometría nueva → slot nuevo
                        if n_slots >= npairs_max:
                            raise RuntimeError(
                                f"[Stage 0] Superado npairs_max={npairs_max}. "
                                f"Aumenta npairs_max o amplía las tolerancias."
                            )
                        k = n_slots
                        pairs_to_compute[k] = [i_station, i_psource]
                        dh_of_pairs[k]   = dh
                        dv_of_pairs[k]   = dv
                        zrec_of_pairs[k] = z_rec
                        zsrc_of_pairs[k] = z_src
                        n_slots += 1
                    else:
                        # Cubierto: slot más cercano en distancia L1
                        covered_idx = np.where(~not_covered)[0]
                        dist = (np.abs(dh    - dh_of_pairs[covered_idx]) +
                                np.abs(z_src - zsrc_of_pairs[covered_idx]) +
                                np.abs(z_rec - zrec_of_pairs[covered_idx]))
                        k = covered_idx[np.argmin(dist)]

                pair_to_slot[flat_idx] = k

                if showProgress and flat_idx % 50000 == 0 and flat_idx > 0:
                    elapsed = perf_counter() - t0_start
                    print(f"  {flat_idx}/{npairs_total} pares | {n_slots} slots | "
                          f"elapsed={elapsed:.1f}s ETA={_eta_str(elapsed, flat_idx, npairs_total)}")

        # Trim arrays
        pairs_to_compute = pairs_to_compute[:n_slots]
        dh_of_pairs      = dh_of_pairs[:n_slots]
        dv_of_pairs      = dv_of_pairs[:n_slots]
        zrec_of_pairs    = zrec_of_pairs[:n_slots]
        zsrc_of_pairs    = zsrc_of_pairs[:n_slots]

        elapsed = perf_counter() - t0_start
        reduction = (1 - n_slots / npairs_total) * 100

        assert np.all(pair_to_slot >= 0), \
            "[Stage 0] BUG: pair_to_slot contiene -1 (pares sin slot asignado)"
        assert np.all(pair_to_slot < n_slots), \
            "[Stage 0] BUG: pair_to_slot contiene índice fuera de rango"

        print(f"\n  ✓ Slots únicos : {n_slots} de {npairs_total} ({reduction:.1f}% reducción)")
        print(f"  ✓ Tiempo       : {elapsed:.1f}s")

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

        print(f"  ✓ Guardado en  : {h5_database_name}.h5")
        print(f"{'='*70}\n")

        if use_mpi and nprocs > 1:
            comm.Barrier()

    # =========================================================================
    # STAGE 1: Calcular GF (tdata) para cada slot único
    # =========================================================================

    def compute_gf_op(self,
                      h5_database_name,
                      dt=0.05, nfft=4096, tb=1000,
                      smth=1, sigma=2, taper=0.9,
                      wc1=1, wc2=2, pmin=0, pmax=1,
                      dk=0.3, nx=1, kc=15.0,
                      verbose=False, debugMPI=False,
                      showProgress=True):
        """
        Stage 1: Calcula el kernel FK (tdata) para cada slot único k.

        Lee ``h5_database_name.h5`` (producido por gen_pairs_op).
        Escribe el grupo ``tdata_dict`` en el mismo archivo.

        MPI: rank 0 coordina y escribe; workers calculan y envían.

        :param h5_database_name: Ruta HDF5 (sin .h5)
        :param dt: Paso de tiempo (s)
        :param nfft: Puntos FFT
        :param tb: Muestras antes de primera llegada
        :param smth: Factor de densificación
        :param sigma: Amortiguamiento
        :param taper: Filtro pasa-bajo (0-1)
        :param wc1, wc2: Frecuencias de corte
        :param pmin, pmax: Límites velocidad de fase
        :param dk: Intervalo en número de onda
        :param nx: Número de rangos de distancia
        :param kc: Número de onda máximo
        :param verbose: Salida detallada core Fortran
        :param debugMPI: Archivos debug por rank
        :param showProgress: Imprime ETA en rank 0
        """
        title = f"[Stage 1] compute_gf_op: {dt=} {nfft=} {dk=} {tb=}"

        if rank == 0:
            print(f"\n{'='*70}")
            print(title)
            print(f"  MPI processes : {nprocs}")
            print(f"  OMP threads   : {os.environ.get('OMP_NUM_THREADS', 'not set')}")
            hfile = h5py.File(h5_database_name + '.h5', 'r+')
        else:
            hfile = h5py.File(h5_database_name + '.h5', 'r')

        pairs_to_compute = hfile["/pairs_to_compute"][:]
        npairs = len(pairs_to_compute)

        if rank == 0:
            print(f"  Slots a computar: {npairs}")
            print(f"{'='*70}")
            if "tdata_dict" in hfile:
                print("  ⚠  tdata_dict existente encontrado. Sobreescribiendo.")
                del hfile["tdata_dict"]
            tdata_group = hfile.create_group("tdata_dict")

        if debugMPI:
            fid = open(f"rank_{rank}_stage1.debuginfo", "w")
            def printMPI(*args): fid.write(" ".join(str(a) for a in args) + "\n")
        else:
            fid = open(os.devnull, "w")
            printMPI = lambda *args: None

        if nprocs == 1 or rank == 0:
            next_pair, skip_pairs = rank, 1
        else:
            next_pair, skip_pairs = rank - 1, nprocs - 1

        perf_time_begin = perf_counter()
        c = _perf_counters()
        tstart = perf_counter()
        ipair = 0

        for i_station, i_psource in pairs_to_compute:
            station = self._receivers.get_station_by_id(int(i_station))
            psource = self._source.get_source_by_id(int(i_psource))

            aux_crust = copy.deepcopy(self._crust)
            aux_crust.split_at_depth(psource.x[2])
            aux_crust.split_at_depth(station.x[2])

            if ipair == next_pair:
                if nprocs == 1 or (rank > 0 and nprocs > 1):
                    t1 = perf_counter()
                    tdata, z, e, n, t0 = self._call_core(
                        dt, nfft, tb, nx, sigma, smth,
                        wc1, wc2, pmin, pmax, dk, kc,
                        taper, aux_crust, psource, station, verbose)
                    c['core'] += perf_counter() - t1

                    nt = len(z)
                    t0_arr = np.array([t0], dtype=np.double)
                    # Convertir tdata a C-order (nt, 9)
                    tdata_c = np.empty((nt, 9), dtype=np.float64)
                    for comp in range(9):
                        tdata_c[:, comp] = tdata[0, comp, :]

                    if rank > 0:
                        t1 = perf_counter()
                        comm.Send(np.array([nt], dtype=np.int32), dest=0, tag=3 * ipair)
                        comm.Send(t0_arr,  dest=0, tag=3 * ipair + 1)
                        comm.Send(tdata_c, dest=0, tag=3 * ipair + 2)
                        c['send'] += perf_counter() - t1
                        next_pair += skip_pairs

                if rank == 0:
                    if nprocs > 1:
                        remote = ipair % (nprocs - 1) + 1
                        t1 = perf_counter()
                        ant   = np.empty(1, dtype=np.int32)
                        t0_arr = np.empty(1, dtype=np.double)
                        comm.Recv(ant,    source=remote, tag=3 * ipair)
                        comm.Recv(t0_arr, source=remote, tag=3 * ipair + 1)
                        nt = ant[0]
                        tdata_c = np.empty((nt, 9), dtype=np.float64)
                        comm.Recv(tdata_c, source=remote, tag=3 * ipair + 2)
                        c['recv'] += perf_counter() - t1

                    tdata_group[f"{ipair}_t0"]    = t0_arr[0]
                    tdata_group[f"{ipair}_tdata"] = tdata_c

                    next_pair += 1

                    if showProgress:
                        elapsed = perf_counter() - tstart
                        print(f"  {ipair+1}/{npairs}  ETA={_eta_str(elapsed, ipair+1, npairs)}")

            ipair += 1

        fid.close()
        hfile.close()

        perf_time_total = perf_counter() - perf_time_begin
        if rank == 0:
            print(f"\n  ✓ Stage 1 done. Total time: {perf_time_total:.2f}s")
            print(f"{'='*70}\n")

        _print_perf_stats(c, perf_time_total)

        if use_mpi and nprocs > 1:
            comm.Barrier()

    # =========================================================================
    # STAGE 2: Convolucionar GF con STF usando O(1) pair_to_slot
    # =========================================================================

    def run_op(self,
               h5_database_name,
               dt=0.05, nfft=4096, tb=1000,
               smth=1, sigma=2, taper=0.9,
               wc1=1, wc2=2, pmin=0, pmax=1,
               dk=0.3, nx=1, kc=15.0,
               writer=None, writer_mode='progressive',
               verbose=False, debugMPI=False,
               tmin=0., tmax=100, showProgress=True):
        """
        Stage 2: Para cada par (estación, fuente), busca tdata via O(1)
        pair_to_slot, llama a _call_core_fast, convuelve con STF, acumula.

        MPI: cada rank procesa sus estaciones asignadas. Luego rank 0
        recolecta y escribe via writer.

        :param h5_database_name: Ruta HDF5 (sin .h5)
        :param writer: StationListWriter para guardar resultados
        :param writer_mode: 'progressive' o 'legacy'
        :param tmin, tmax: Ventana de tiempo de salida (s)
        :param verbose: Salida detallada core Fortran
        :param debugMPI: Archivos debug por rank
        :param showProgress: Imprime ETA en rank 0
        (resto de parámetros: igual que run())
        """
        title = f"[Stage 2] run_op: {dt=} {nfft=} {tmin=} {tmax=}"

        if rank == 0:
            print(f"\n{'='*70}")
            print(title)
            print(f"  MPI processes : {nprocs}")
            print(f"  OMP threads   : {os.environ.get('OMP_NUM_THREADS', 'not set')}")
            hfile = h5py.File(h5_database_name + '.h5', 'r+')
        else:
            hfile = h5py.File(h5_database_name + '.h5', 'r')

        # Cargar mapping O(1)
        pair_to_slot = hfile["/pair_to_slot"][:]
        nsources_db  = int(hfile["/nsources"][()])
        nstations_db = int(hfile["/nstations"][()])

        if rank == 0:
            print(f"  pair_to_slot  : ✓ O(1) lookup activo")
            print(f"  DB            : {nstations_db} estaciones × {nsources_db} fuentes")
            print(f"{'='*70}")

        # Validar coherencia entre HDF5 y modelo actual
        assert nsources_db == self._source.nsources, \
            f"[Stage 2] nsources HDF5={nsources_db} ≠ modelo={self._source.nsources}"
        assert nstations_db == self._receivers.nstations, \
            f"[Stage 2] nstations HDF5={nstations_db} ≠ modelo={self._receivers.nstations}"

        if rank > 0:
            writer = None
        if writer and rank == 0:
            assert isinstance(writer, StationListWriter)
            writer.initialize(self._receivers, 2 * nfft,
                              tmin=tmin, tmax=tmax, dt=dt,
                              writer_mode=writer_mode)
            writer.write_metadata(self._receivers.metadata)

        if debugMPI:
            fid = open(f"rank_{rank}_stage2.debuginfo", "w")
            def printMPI(*args): fid.write(" ".join(str(a) for a in args) + "\n")
        else:
            fid = open(os.devnull, "w")
            printMPI = lambda *args: None

        self._logger.info(
            f'ShakerMaker.run_op: {self._source.nsources} sources, '
            f'{self._receivers.nstations} stations, dt={dt}, nfft={nfft}')

        nsources  = self._source.nsources
        nstations = self._receivers.nstations
        next_station  = rank
        skip_stations = nprocs

        perf_time_begin = perf_counter()
        c = _perf_counters()
        n_my_stations = 0

        # ------------------------------------------------------------------
        # 2a: Cada rank procesa sus estaciones asignadas
        # ------------------------------------------------------------------
        for i_station, station in enumerate(self._receivers):

            if i_station != next_station:
                continue

            tstart_sta = perf_counter()

            for i_psource, psource in enumerate(self._source):

                # Lookup O(1)
                k = int(pair_to_slot[i_station * nsources + i_psource])

                tdata = hfile[f"/tdata_dict/{k}_tdata"][:]

                aux_crust = copy.deepcopy(self._crust)
                aux_crust.split_at_depth(psource.x[2])
                aux_crust.split_at_depth(station.x[2])

                t1 = perf_counter()
                z, e, n, t0 = self._call_core_fast(
                    tdata, dt, nfft, tb, nx, sigma, smth,
                    wc1, wc2, pmin, pmax, dk, kc,
                    taper, aux_crust, psource, station, verbose)
                c['core'] += perf_counter() - t1

                t1 = perf_counter()
                t_arr = np.arange(0, len(z) * dt, dt) + psource.tt + t0
                psource.stf.dt = dt
                z_stf = psource.stf.convolve(z, t_arr)
                e_stf = psource.stf.convolve(e, t_arr)
                n_stf = psource.stf.convolve(n, t_arr)
                c['conv'] += perf_counter() - t1

                try:
                    t1 = perf_counter()
                    station.add_to_response(z_stf, e_stf, n_stf, t_arr, tmin, tmax)
                    c['add'] += perf_counter() - t1
                except Exception:
                    traceback.print_exc()
                    if use_mpi and nprocs > 1:
                        comm.Abort()

                station.add_greens_function(z, e, n, t_arr, tdata, t0, i_psource)

                if showProgress and rank == 0 and i_psource % 1000 == 0:
                    elapsed = perf_counter() - tstart_sta
                    pct = i_psource / nsources * 100
                    print(f"  Sta {i_station} src {i_psource}/{nsources} "
                          f"({pct:.1f}%)  ETA={_eta_str(elapsed, i_psource+1, nsources)}")

            n_my_stations += 1

            elapsed_sta = perf_counter() - tstart_sta
            pct = i_station / nstations * 100
            nsta_left = (nstations - i_station - 1) // skip_stations
            eta_s = nsta_left * elapsed_sta
            print(f"  {rank=} sta {i_station}/{nstations} ({pct:.1f}%)  "
                  f"sta_time={elapsed_sta:.1f}s  ETA={_eta_str(eta_s, 1, 2)}")

            next_station += skip_stations

        hfile.close()

        # ------------------------------------------------------------------
        # 2b: Workers envían a rank 0; rank 0 recolecta y escribe
        # ------------------------------------------------------------------
        if use_mpi and nprocs > 1:
            if rank > 0:
                my_sta = rank
                print(f"Rank {rank} enviando datos...")
                while my_sta < nstations:
                    station = self._receivers.get_station_by_id(my_sta)
                    z, e, n, t = station.get_response()
                    t1 = perf_counter()
                    comm.Send(np.array([len(z)], dtype=np.int32), dest=0, tag=2 * my_sta)
                    comm.Send(np.column_stack([z, e, n, t]), dest=0, tag=2 * my_sta + 1)
                    c['send'] += perf_counter() - t1
                    my_sta += nprocs
                print(f"Rank {rank} DONE enviando.")

            if rank == 0:
                print("Rank 0 recolectando resultados...")
                count = 0
                for remote in range(1, nprocs):
                    rsta = remote
                    while rsta < nstations:
                        station = self._receivers.get_station_by_id(rsta)
                        t1 = perf_counter()
                        ant = np.empty(1, dtype=np.int32)
                        comm.Recv(ant, source=remote, tag=2 * rsta)
                        nt = ant[0]
                        data = np.empty((nt, 4), dtype=np.float64)
                        comm.Recv(data, source=remote, tag=2 * rsta + 1)
                        c['recv'] += perf_counter() - t1
                        station.add_to_response(
                            data[:, 0], data[:, 1], data[:, 2], data[:, 3], tmin, tmax)
                        if writer:
                            writer.write_station(station, rsta)
                        count += 1
                        rsta += nprocs

                # Escribe las estaciones propias de rank 0
                my_sta = 0
                while my_sta < nstations:
                    station = self._receivers.get_station_by_id(my_sta)
                    if writer:
                        writer.write_station(station, my_sta)
                    my_sta += nprocs

                count += n_my_stations
                print(f"Rank 0: {count}/{nstations} estaciones recolectadas.")

                if writer:
                    writer.close()

        else:
            # Single process
            if writer:
                for i_station, station in enumerate(self._receivers):
                    writer.write_station(station, i_station)
                writer.close()

        fid.close()
        perf_time_total = perf_counter() - perf_time_begin

        if rank == 0:
            print(f"\n  ✓ Stage 2 done. Total time: {perf_time_total:.2f}s")
            print(f"{'='*70}\n")

        _print_perf_stats(c, perf_time_total)

    # =========================================================================
    # ORQUESTADOR: run_fast_faster_op
    # =========================================================================

    def run_fast_faster_op(self,
                           stage='all',
                           h5_database_name=None,
                           # Stage 0
                           delta_h=0.04,
                           delta_v_rec=0.002,
                           delta_v_src=0.2,
                           npairs_max=200000,
                           # Core (stages 1 y 2)
                           dt=0.05, nfft=4096, tb=1000,
                           smth=1, sigma=2, taper=0.9,
                           wc1=1, wc2=2, pmin=0, pmax=1,
                           dk=0.3, nx=1, kc=15.0,
                           # Stage 2
                           writer=None,
                           writer_mode='progressive',
                           tmin=0., tmax=100,
                           # General
                           verbose=False, debugMPI=False,
                           showProgress=True):
        """
        Orquestador del pipeline OP completo.

        Corre Stage 0, 1 y/o 2 según el parámetro ``stage``.
        Cada stage puede correrse por separado para workflows HPC
        donde Stage 0 es serial y Stages 1-2 son MPI paralelos.

        :param stage: Etapa(s) a correr: ``0``, ``1``, ``2`` o ``'all'``
        :param h5_database_name: Ruta HDF5 sin extensión. Requerido.

        **Stage 0 params:**

        :param delta_h: Tolerancia horizontal (km)
        :param delta_v_rec: Tolerancia profundidad receptor (km)
        :param delta_v_src: Tolerancia profundidad fuente (km)
        :param npairs_max: Máximo slots únicos

        **Core params (stages 1 y 2):**

        :param dt: Paso de tiempo (s)
        :param nfft: Puntos FFT
        :param tb: Muestras antes de primera llegada
        :param smth: Factor de densificación
        :param sigma: Amortiguamiento
        :param taper: Filtro pasa-bajo (0-1)
        :param wc1, wc2: Frecuencias de corte
        :param pmin, pmax: Límites velocidad de fase (1/vs)
        :param dk: Intervalo número de onda (Pi/x)
        :param nx: Número de rangos de distancia
        :param kc: Número de onda máximo (1/hs)

        **Stage 2 params:**

        :param writer: StationListWriter para salida. Requerido en stage 2.
        :param writer_mode: ``'progressive'`` o ``'legacy'``
        :param tmin, tmax: Ventana de tiempo de salida (s)

        **General:**

        :param verbose: Salida detallada core Fortran
        :param debugMPI: Archivos debug por rank
        :param showProgress: Imprime ETA
        """
        assert h5_database_name is not None, \
            "run_fast_faster_op: h5_database_name es requerido"
        assert stage in (0, 1, 2, 'all'), \
            "run_fast_faster_op: stage debe ser 0, 1, 2 o 'all'"

        perf_time_begin = perf_counter()

        if rank == 0:
            title = (f"🚀 run_fast_faster_op | stage={stage} | "
                     f"{dt=} {nfft=} {dk=} {tb=} {tmin=} {tmax=}")
            print(f"\n\n{'='*len(title)}")
            print(title)
            print(f"{'='*len(title)}")
            omp = os.environ.get('OMP_NUM_THREADS', 'not set')
            print(f"  MPI processes : {nprocs}")
            print(f"  OMP threads   : {omp}")
            if omp != 'not set':
                print(f"  Total threads : {nprocs} × {omp} = {nprocs * int(omp)}")
            print(f"  DB file       : {h5_database_name}.h5")
            print(f"{'='*len(title)}\n")

        # ── Stage 0 ──────────────────────────────────────────────────────────
        if stage in (0, 'all'):
            self.gen_pairs_op(
                h5_database_name=h5_database_name,
                delta_h=delta_h,
                delta_v_rec=delta_v_rec,
                delta_v_src=delta_v_src,
                npairs_max=npairs_max,
                showProgress=showProgress)

            if stage == 0:
                if rank == 0:
                    print(f"✓ Stage 0 completo → {h5_database_name}.h5")
                return

        # ── Stage 1 ──────────────────────────────────────────────────────────
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
                    print(f"✓ Stage 1 completo → {h5_database_name}.h5")
                return

        # ── Stage 2 ──────────────────────────────────────────────────────────
        if stage in (2, 'all'):
            if writer is None and rank == 0:
                print("⚠  Stage 2 requiere un writer. Abortando.")
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

        # ── Resumen final ────────────────────────────────────────────────────
        if rank == 0 and stage == 'all':
            total = perf_counter() - perf_time_begin
            print(f"\n{'='*70}")
            print(f"✓ PIPELINE COMPLETO")
            print(f"  Tiempo total : {total:.2f}s", end="")
            if total > 60:
                print(f"  ({total/60:.1f} min)", end="")
            if total > 3600:
                print(f"  ({total/3600:.2f} hrs)", end="")
            print(f"\n{'='*70}\n")

    # =========================================================================
    # STKO: Exportar geometría DRM
    # =========================================================================

    def export_drm_geometry(self, filename="drm_geometry.h5drm"):
        """
        Exporta geometría DRM para visualización en STKO.

        Crea un archivo HDF5 con coordenadas de estaciones y datos mínimos
        (2 muestras, rampa lineal 0→10) solo para visualización.

        Funciona con receptores DRMBox y SurfaceGrid.

        :param filename: Nombre del archivo HDF5 de salida
        :returns: Ruta al archivo creado
        """
        from shakermaker.sl_extensions import DRMBox
        from shakermaker.sl_extensions.SurfaceGrid import SurfaceGrid

        if not isinstance(self._receivers, (DRMBox, SurfaceGrid)):
            raise TypeError(
                f"export_drm_geometry() requiere receptores DRMBox o SurfaceGrid. "
                f"Tipo actual: {type(self._receivers).__name__}"
            )

        if rank != 0:
            return filename

        metadata  = self._receivers.metadata
        nstations = self._receivers.nstations - 1  # excluye estación QA

        dt     = 0.0005
        tstart = 0.0
        tend   = 10.0

        print(f"\n{'='*70}")
        print(f"export_drm_geometry")
        print(f"  Archivo   : {filename}")
        print(f"  Estaciones: {nstations + 1} (incluyendo QA)")
        print(f"{'='*70}")

        with h5py.File(filename, 'w') as hf:
            grp_data = hf.create_group('DRM_Data')
            grp_qa   = hf.create_group('DRM_QA_Data')
            grp_meta = hf.create_group('DRM_Metadata')

            # Coordenadas y flags de estaciones
            xyz      = np.zeros((nstations, 3))
            internal = np.zeros(nstations, dtype=bool)
            for i in range(nstations):
                sta = self._receivers.get_station_by_id(i)
                xyz[i, :]    = sta.x
                internal[i]  = sta.is_internal

            grp_data.create_dataset('xyz',          data=xyz,      dtype=np.float64)
            grp_data.create_dataset('internal',     data=internal, dtype=bool)
            grp_data.create_dataset('data_location',
                                    data=np.arange(0, nstations, dtype=np.int32) * 3)

            # Estación QA
            qa_sta = self._receivers.get_station_by_id(nstations)
            grp_qa.create_dataset('xyz', data=qa_sta.x.reshape(1, 3), dtype=np.float64)

            # Datos mínimos: rampa 0→10 (2 muestras, 3 componentes por estación)
            ramp    = np.tile([0.0, 10.0], (3 * nstations, 1))
            ramp_qa = np.tile([0.0, 10.0], (3, 1))

            for grp, r in [(grp_data, ramp), (grp_qa, ramp_qa)]:
                grp.create_dataset('velocity',     data=r, dtype=np.float64)
                grp.create_dataset('displacement', data=r, dtype=np.float64)
                grp.create_dataset('acceleration', data=r, dtype=np.float64)

            # Metadata
            grp_meta.create_dataset('dt',     data=dt)
            grp_meta.create_dataset('tstart', data=tstart)
            grp_meta.create_dataset('tend',   data=tend)

            for key in ['h', 'drmbox_x0', 'drmbox_xmax', 'drmbox_xmin',
                        'drmbox_ymax', 'drmbox_ymin', 'drmbox_zmax', 'drmbox_zmin']:
                if key in metadata:
                    grp_meta.create_dataset(key, data=metadata[key])

        print(f"  ✓ Archivo creado: {filename}")
        print(f"{'='*70}\n")
        return filename

    # =========================================================================
    # Internos: wrappers al core Fortran
    # =========================================================================

    def _call_core(self, dt, nfft, tb, nx, sigma, smth, wc1, wc2,
                   pmin, pmax, dk, kc, taper, crust, psource, station, verbose=False):
        """
        Llama a core.subgreen para calcular el kernel FK completo (tdata).
        Usado por: run() y compute_gf_op().
        """
        mb  = crust.nlayers
        src = crust.get_layer(psource.x[2]) + 1  # Fortran indexing
        rcv = crust.get_layer(station.x[2]) + 1

        stype = 2   # double-couple, up y down going
        updn  = 0

        pf = psource.angles[0]
        df = psource.angles[1]
        lf = psource.angles[2]
        sx = psource.x[0]; sy = psource.x[1]
        rx = station.x[0]; ry = station.x[1]
        x  = np.sqrt((sx - rx)**2 + (sy - ry)**2)

        self._logger.debug(
            f'_call_core: mb={mb} src={src} rcv={rcv} x={x:.4f} '
            f'pf={pf} df={df} lf={lf}')

        if verbose:
            print(f'  _call_core: mb={mb} src={src} rcv={rcv} '
                  f'x={x:.4f} pf={pf} df={df} lf={lf}')

        tdata, z, e, n, t0 = core.subgreen(
            mb, src, rcv, stype, updn,
            crust.d, crust.a, crust.b, crust.rho, crust.qa, crust.qb,
            dt, nfft, tb, nx, sigma, smth, wc1, wc2,
            pmin, pmax, dk, kc, taper, x, pf, df, lf, sx, sy, rx, ry)

        return tdata, z, e, n, t0

    def _call_core_fast(self, tdata, dt, nfft, tb, nx, sigma, smth, wc1, wc2,
                        pmin, pmax, dk, kc, taper, crust, psource, station, verbose=False):
        """
        Llama a core.subgreen2 reutilizando tdata precomputado.
        Usado por: run_op().
        """
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

        self._logger.debug(
            f'_call_core_fast: mb={mb} src={src} rcv={rcv} x={x:.4f}')

        # Reshape tdata de (nt, 9) → (1, 9, nt) que espera subgreen2
        tdata_ = tdata.T
        tdata_ = tdata_.reshape((1, tdata_.shape[0], tdata_.shape[1]))

        z, e, n, t0 = core.subgreen2(
            mb, src, rcv, stype, updn,
            crust.d, crust.a, crust.b, crust.rho, crust.qa, crust.qb,
            dt, nfft, tb, nx, sigma, smth, wc1, wc2,
            pmin, pmax, dk, kc, taper, x, pf, df, lf, tdata_, sx, sy, rx, ry)

        return z, e, n, t0

    # =========================================================================
    # Utilidades
    # =========================================================================

    def write(self, writer):
        """Escribe todos los receptores usando el writer dado."""
        writer.write(self._receivers)

    def enable_mpi(self, rank, nprocs):
        """Sobreescribe los valores MPI (raro, solo si necesitas override manual)."""
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