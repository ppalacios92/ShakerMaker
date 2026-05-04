"""
build_h5drm_from_sw4_case
=========================
Construye un H5DRM completo a partir de los resultados de SW4.

Geometría
---------
- Estaciones ShakerMaker (DRM boundary): xyz_km exacto + internal/is_qa desde
  model_geometry.h5  (fuente autoritativa, coordenadas en SW4 km).
- Estaciones topo_surface / topo_z0 / domain_grid: xyz calculado desde las
  líneas rec del .in + lookup de topografía.

Convención de signos / ejes (ShakerMaker ↔ SW4)
-------------------------------------------------
ShakerMaker: Z positivo hacia abajo (mismo que SW4).
SW4 columnas txt: time | X-vel (=North) | Y-vel (=East) | Z-vel (down+).
H5DRM component order: E, N, Z  →  vel[0]=E, vel[1]=N, vel[2]=Z.

Escritura rápida
----------------
- Sin copiar geometry.h5 (reconstruye geometría completa desde cero).
- ThreadPoolExecutor por lotes → lectura paralela de .txt.
- Layout contiguo (sin chunks) → viewer rápido.
- Sin MPI, sin partes, sin merge.

Uso típico:
    python build_h5drm_from_sw4_case.py
"""

import datetime
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def build_h5drm_from_sw4_case(
    case_path,
    input_file=None,
    output_name="motions.h5drm",
    use_filter=False,
    freqmin=0.25,
    freqmax=10.0,
    corners=4,
    zerophase=True,
    n_workers=None,
    batch_size=2000,
    float_dtype=np.float32,
):
    """Construye un H5DRM completo (geometría + señales) desde resultados de SW4.

    Parameters
    ----------
    case_path : str | Path
        Raíz del caso. Dentro debe existir ``sw4/`` y ``shakermakerexports/``.
    input_file : str | Path | None
        Archivo .in de SW4. Por defecto ``sw4/shakermaker2sw4.in``.
    output_name : str
        Nombre del archivo de salida en ``shakermakerexports/``.
    use_filter : bool
        Aplicar filtro bandpass (requiere obspy).
    freqmin, freqmax : float
        Frecuencias de corte del filtro [Hz].
    corners : int
        Orden del filtro Butterworth.
    zerophase : bool
        Si True usa filtro de fase cero (doble pasada).
    n_workers : int | None
        Hilos para lectura paralela. None → min(cpu_count, 8).
    batch_size : int
        Estaciones por lote de escritura.
    float_dtype : dtype
        Tipo de dato para velocity/displacement/acceleration.
        np.float32 (default) → mitad del tamaño en disco.
        np.float64 → máxima precisión.
    """
    case_path    = Path(case_path)
    sw4_dir      = case_path / "sw4"
    exports_dir  = case_path / "shakermakerexports"
    geometry_h5  = exports_dir / "model_geometry.h5"
    output_h5drm = exports_dir / output_name

    exports_dir.mkdir(parents=True, exist_ok=True)

    if input_file is None:
        input_file = sw4_dir / "shakermaker2sw4.in"
    else:
        input_file = Path(input_file)
        if not input_file.is_absolute():
            input_file = sw4_dir / input_file

    print(f"\n{'='*60}")
    print(f"case   : {case_path}")
    print(f"output : {output_h5drm}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Parsear el .in de SW4
    # ------------------------------------------------------------------
    fileio_path, topo_rel_file, rec_lines = _read_sw4_input_full(input_file)
    if fileio_path is None:
        raise ValueError(f"No se encontró 'fileio path=...' en {input_file}")
    station_dir = (input_file.parent / fileio_path).resolve()

    topo_path   = (input_file.parent / topo_rel_file) if topo_rel_file else None
    topo_lookup = _build_topo_lookup(topo_path)

    # ------------------------------------------------------------------
    # 2. Geometría ShakerMaker desde geometry.h5
    #    Fuente autoritativa para xyz_km + internal + is_qa.
    # ------------------------------------------------------------------
    sm_geometry = _read_geometry_h5(geometry_h5)   # dict name → {xyz_km, internal, is_qa}

    # ------------------------------------------------------------------
    # 3. Construir lista ordenada de todas las estaciones (igual que .in)
    # ------------------------------------------------------------------
    stations    = []   # todas menos QA
    qa_station  = None
    missing     = []

    for rec in rec_lines:
        name = Path(rec["file"]).name          # e.g. "sf00001"
        txt  = station_dir / f"{name}.txt"
        if not txt.exists():
            missing.append(name)
            continue

        geom = sm_geometry.get(name)
        if geom is not None:
            # Estación ShakerMaker — geometry.h5 es la fuente
            xyz_km   = np.asarray(geom["xyz_km"], dtype=np.float64)
            internal = bool(geom["internal"])
            is_qa    = bool(geom["is_qa"])
            kind     = "shakermaker"
        else:
            # Estación topo / z0 / domain grid — coordenadas desde rec line
            # SW4: x=North, y=East, z=down (positivo hacia abajo, igual que ShakerMaker)
            xyz_sw4_m = _xyz_from_rec(rec, topo_lookup)
            xyz_km    = np.asarray(xyz_sw4_m, dtype=np.float64) / 1000.0
            internal  = True    # no es nodo DRM boundary
            is_qa     = False
            if "depth" in rec:
                kind = "topo_surface" if float(rec["depth"]) == 0.0 else "topo_z0"
            else:
                kind = "domain_grid"

        item = {
            "name"    : name,
            "txt"     : txt,
            "xyz_km"  : xyz_km,
            "internal": internal,
            "is_qa"   : is_qa,
            "kind"    : kind,
        }
        if is_qa:
            qa_station = item
        else:
            stations.append(item)

    if missing:
        print(f"  WARNING: {len(missing)} archivo(s) .txt no encontrado(s) "
              f"(primero: {missing[0]}.txt) — estaciones omitidas")
    if not stations:
        raise FileNotFoundError(
            f"No se encontraron archivos .txt en {station_dir}\n"
            f"Se esperan archivos como sf00001.txt"
        )

    kind_counts = {}
    for s in stations:
        kind_counts[s["kind"]] = kind_counts.get(s["kind"], 0) + 1

    nstations = len(stations)

    # ------------------------------------------------------------------
    # 4. Eje de tiempo desde la primera estación disponible
    # ------------------------------------------------------------------
    first_data = _fast_loadtxt(stations[0]["txt"])
    t          = first_data[:, 0]
    nt         = len(t)
    dt         = float((t[-1] - t[0]) / (nt - 1))

    print(f"  Estaciones totales : {nstations}")
    for k, n in sorted(kind_counts.items()):
        print(f"    {k:<16}: {n}")
    print(f"  nt={nt}  dt={dt:.6f}s  duración={t[-1]:.3f}s")

    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 8)
    n_batches = math.ceil(nstations / batch_size)
    print(f"  Lotes={n_batches}  batch_size={batch_size}  hilos={n_workers}")

    # ------------------------------------------------------------------
    # 5. Geometría QA
    #    Si existe una estación QA dedicada en el .in se usa su xyz.
    #    Si no, se coloca en el centroide horizontal a z=0.
    # ------------------------------------------------------------------
    if qa_station is not None:
        qa_xyz  = qa_station["xyz_km"].reshape(1, 3)
        qa_name = qa_station["name"]
        qa_defined = True
    else:
        all_xyz = np.asarray([s["xyz_km"] for s in stations], dtype=np.float64)
        xy      = 0.5 * (all_xyz[:, :2].min(axis=0) + all_xyz[:, :2].max(axis=0))
        qa_xyz  = np.array([[xy[0], xy[1], 0.0]], dtype=np.float64)
        qa_name = ""
        qa_defined = False

    # ------------------------------------------------------------------
    # 6. Crear H5DRM desde cero con geometría completa
    # ------------------------------------------------------------------
    xyz_all      = np.asarray([s["xyz_km"]   for s in stations], dtype=np.float64)
    internal_all = np.asarray([s["internal"] for s in stations], dtype=bool)
    # data_location: índice de fila base para cada estación (layout contiguo)
    data_location = np.arange(nstations, dtype=np.int32) * 3
    string_dtype  = h5py.string_dtype(encoding="utf-8")

    print("  Creando H5DRM con geometría completa...")
    with h5py.File(output_h5drm, "w") as f:

        # ── DRM_Data: geometría ──────────────────────────────────────────
        g = f.create_group("DRM_Data")
        g.create_dataset("xyz",           data=xyz_all)
        g.create_dataset("internal",      data=internal_all)
        g.create_dataset("data_location", data=data_location)
        g.create_dataset("kind",  data=np.array([s["kind"] for s in stations], dtype=object),
                         dtype=string_dtype)
        g.create_dataset("name",  data=np.array([s["name"] for s in stations], dtype=object),
                         dtype=string_dtype)
        # chunks=(3, nt): un chunk = un nodo completo → column-access eficiente en el viewer
        g.create_dataset("velocity",     shape=(3 * nstations, nt), dtype=float_dtype, chunks=(3, nt))
        g.create_dataset("displacement", shape=(3 * nstations, nt), dtype=float_dtype, chunks=(3, nt))
        g.create_dataset("acceleration", shape=(3 * nstations, nt), dtype=float_dtype, chunks=(3, nt))

        # ── DRM_QA_Data ──────────────────────────────────────────────────
        q = f.create_group("DRM_QA_Data")
        q.create_dataset("xyz",          data=qa_xyz)
        q.create_dataset("velocity",     shape=(3, nt), dtype=float_dtype)
        q.create_dataset("displacement", shape=(3, nt), dtype=float_dtype)
        q.create_dataset("acceleration", shape=(3, nt), dtype=float_dtype)

        # ── DRM_Metadata ─────────────────────────────────────────────────
        m = f.create_group("DRM_Metadata")
        m.create_dataset("dt",              data=dt)
        m.create_dataset("tstart",          data=float(t[0]))
        m.create_dataset("tend",            data=float(t[-1]))
        m.create_dataset("nt",              data=int(nt))
        m.create_dataset("receiver_count",  data=int(nstations))
        m.create_dataset("qa_defined",      data=qa_defined)
        m.create_dataset("qa_index",        data=-1)
        m.create_dataset("qa_receiver_file",data=qa_name,            dtype=string_dtype)
        m.create_dataset("created_on",      data=datetime.datetime.now().isoformat(), dtype=string_dtype)
        m.create_dataset("writer_mode",     data="sequential_fast",  dtype=string_dtype)
        m.create_dataset("component_order", data="E,N,Z",            dtype=string_dtype)
        m.create_dataset("component_map",   data="E=SW4_Y, N=SW4_X, Z=SW4_Z(down+)", dtype=string_dtype)
        m.create_dataset("coordinate_units",data="km",               dtype=string_dtype)
        m.create_dataset("signal_units",    data="vel SW4 txt; disp integrado trapecio; acc diferencia central",
                         dtype=string_dtype)
        m.create_dataset("sw4_input_file",  data=str(input_file),    dtype=string_dtype)
        m.create_dataset("station_dir",     data=str(station_dir),   dtype=string_dtype)
        m.create_dataset("geometry_h5",     data=str(geometry_h5),   dtype=string_dtype)
        m.create_dataset("filter_enabled",  data=bool(use_filter))
        m.create_dataset("filter_freqmin",  data=float(freqmin))
        m.create_dataset("filter_freqmax",  data=float(freqmax))
        m.create_dataset("filter_corners",  data=int(corners))
        m.create_dataset("filter_zerophase",data=bool(zerophase))

    # ------------------------------------------------------------------
    # 7. Escribir señales por lotes (threaded)
    # ------------------------------------------------------------------
    with h5py.File(output_h5drm, "r+") as f:
        ds_vel  = f["DRM_Data/velocity"]
        ds_disp = f["DRM_Data/displacement"]
        ds_acc  = f["DRM_Data/acceleration"]

        for b in range(n_batches):
            b0 = b * batch_size
            b1 = min(b0 + batch_size, nstations)
            nb = b1 - b0
            print(f"  Lote {b+1}/{n_batches}: estaciones {b0}–{b1-1}...", flush=True)

            vel_buf  = np.empty((3 * nb, nt), dtype=float_dtype)
            disp_buf = np.empty((3 * nb, nt), dtype=float_dtype)
            acc_buf  = np.empty((3 * nb, nt), dtype=float_dtype)

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        _read_one_station,
                        stations[b0 + j], t, nt, dt,
                        use_filter, freqmin, freqmax, corners, zerophase,
                        float_dtype,
                    ): j
                    for j in range(nb)
                }
                for future in as_completed(futures):
                    j = futures[future]
                    vel, disp, acc = future.result()
                    row = 3 * j
                    vel_buf [row:row + 3] = vel
                    disp_buf[row:row + 3] = disp
                    acc_buf [row:row + 3] = acc

            # Escribir usando data_location del lote actual
            for j in range(nb):
                row = int(data_location[b0 + j])
                ds_vel [row:row + 3] = vel_buf [3 * j:3 * j + 3]
                ds_disp[row:row + 3] = disp_buf[3 * j:3 * j + 3]
                ds_acc [row:row + 3] = acc_buf [3 * j:3 * j + 3]

            print(f"  Lote {b+1}/{n_batches}: listo  "
                  f"[{b1}/{nstations} = {100 * b1 / nstations:.1f}%]", flush=True)
            del vel_buf, disp_buf, acc_buf

        # ------------------------------------------------------------------
        # 8. Señal QA
        # ------------------------------------------------------------------
        if qa_station is not None:
            print(f"  QA station: {qa_station['name']} (dedicada en .in)")
            vel, disp, acc = _read_one_station(
                qa_station, t, nt, dt,
                use_filter, freqmin, freqmax, corners, zerophase, float_dtype,
            )
        else:
            # Usar la estación ShakerMaker más cercana al centroide QA
            sm_stations = [s for s in stations if s["kind"] == "shakermaker"]
            if sm_stations:
                sm_xyz   = np.asarray([s["xyz_km"] for s in sm_stations], dtype=np.float64)
                nearest  = int(np.argmin(np.sum((sm_xyz - qa_xyz[0]) ** 2, axis=1)))
                proxy    = sm_stations[nearest]
                print(f"  QA station: {proxy['name']} (más cercana al centroide)")
                vel, disp, acc = _read_one_station(
                    proxy, t, nt, dt,
                    use_filter, freqmin, freqmax, corners, zerophase, float_dtype,
                )
            else:
                print("  QA station: ceros (no hay estaciones ShakerMaker)")
                vel = disp = acc = np.zeros((3, nt), dtype=float_dtype)

        f["DRM_QA_Data/velocity"][:]     = vel
        f["DRM_QA_Data/displacement"][:] = disp
        f["DRM_QA_Data/acceleration"][:] = acc

    print(f"\n  ✓ Listo → {output_h5drm}")
    print(f"{'='*60}\n")
    return output_h5drm


# ---------------------------------------------------------------------------
# Lectura rápida de .txt SW4
# ---------------------------------------------------------------------------

def _fast_loadtxt(path):
    """Lee un .txt SW4 con I/O binario — mucho más rápido que np.loadtxt."""
    with open(path, "rb") as fh:
        raw = fh.read()
    lines = [l for l in raw.split(b"\n") if l and l[0:1] != b"#"]
    return np.fromstring(b" ".join(lines), dtype=np.float64, sep=" ").reshape(-1, 4)


def _read_one_station(station, t, nt, dt,
                      use_filter, freqmin, freqmax, corners, zerophase,
                      dtype=np.float32):
    """Lee un .txt de SW4 y retorna (velocity, displacement, acceleration).

    SW4 USGS columnas: time | X-vel (=North) | Y-vel (=East) | Z-vel (down+)
    Convención ShakerMaker: Z positivo hacia abajo  →  Z SW4 ya coincide.
    H5DRM component order: E, N, Z  →  vel[0]=E=SW4_Y, vel[1]=N=SW4_X, vel[2]=Z=SW4_Z.
    """
    data = _fast_loadtxt(station["txt"])
    if data.shape[0] != nt:
        raise ValueError(
            f"nt mismatch en {station['txt']}: esperado {nt}, got {data.shape[0]}"
        )

    # SW4: col1=X(N), col2=Y(E), col3=Z(down+)
    n_v = data[:, 1]   # North  = SW4 X
    e_v = data[:, 2]   # East   = SW4 Y
    z_v = data[:, 3]   # Z down+ (positivo hacia abajo, igual que ShakerMaker)

    if use_filter:
        from obspy import Stream, Trace
        traces = [Trace(data=arr.astype(np.float32)) for arr in (e_v, n_v, z_v)]
        for tr in traces:
            tr.stats.delta = dt
        stream = Stream(traces)
        stream.filter("bandpass", freqmin=freqmin, freqmax=freqmax,
                      corners=corners, zerophase=zerophase)
        e_v = stream[0].data
        n_v = stream[1].data
        z_v = stream[2].data

    # Orden E, N, Z en el H5DRM
    velocity = np.vstack((e_v, n_v, z_v)).astype(dtype)

    # Desplazamiento: integración trapezoidal acumulativa
    displacement = np.zeros_like(velocity)
    displacement[:, 1:] = np.cumsum(
        0.5 * (velocity[:, 1:] + velocity[:, :-1]) * dt, axis=1
    )

    # Aceleración: diferencia central (bordes: diferencia hacia adelante/atrás)
    acceleration = np.empty_like(velocity)
    acceleration[:, 1:-1] = (velocity[:, 2:] - velocity[:, :-2]) / (2.0 * dt)
    acceleration[:, 0]    = (velocity[:, 1]  - velocity[:, 0])   / dt
    acceleration[:, -1]   = (velocity[:, -1] - velocity[:, -2])  / dt

    return velocity, displacement, acceleration


# ---------------------------------------------------------------------------
# Parser del .in de SW4
# ---------------------------------------------------------------------------

def _read_sw4_input_full(input_file):
    """Parsea el .in de SW4. Retorna (fileio_path, topo_rel_file, rec_lines)."""
    fileio_path    = None
    topo_rel_file  = None
    rec_lines      = []
    with open(input_file, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("fileio "):
                fileio_path = _parse_kv(line).get("path")
            elif line.startswith("topography "):
                topo_rel_file = _parse_kv(line).get("file")
            elif line.startswith("rec "):
                kv = _parse_kv(line)
                if "file" in kv:
                    rec_lines.append(kv)
    return fileio_path, topo_rel_file, rec_lines


def _parse_kv(line):
    out = {}
    for token in line.split():
        if "=" in token:
            k, v = token.split("=", 1)
            out[k.strip()] = v.strip()
    return out


# ---------------------------------------------------------------------------
# Coordenadas
# ---------------------------------------------------------------------------

def _build_topo_lookup(topo_path):
    """Construye {(round_x_m, round_y_m): tz_m} desde un archivo de topografía cartesiana."""
    if topo_path is None or not Path(topo_path).exists():
        return {}
    lines = [l.strip() for l in Path(topo_path).read_text(encoding="ascii").splitlines() if l.strip()]
    lookup = {}
    for line in lines[1:]:    # saltar cabecera (nx ny)
        cols = line.split()
        if len(cols) >= 3:
            x, y, z = float(cols[0]), float(cols[1]), float(cols[2])
            lookup[(round(x), round(y))] = z
    return lookup


def _xyz_from_rec(rec, topo_lookup):
    """Convierte una línea rec del .in a coordenadas SW4 locales en metros.

    Convención SW4: z=0 datum, z>0 abajo del datum.
    ShakerMaker: Z positivo hacia abajo → mismo signo que SW4.

    depth=0  → en superficie topo          → z = -topo_z
    depth=d  → d m bajo topo               → z = -(topo_z - d)
    z=val    → z SW4 absoluto              → z = val
    """
    x = float(rec.get("x", 0.0))
    y = float(rec.get("y", 0.0))
    if "depth" in rec:
        topo_z = topo_lookup.get((round(x), round(y)), 0.0)
        depth  = float(rec["depth"])
        z      = -(topo_z - depth)
    elif "z" in rec:
        z = float(rec["z"])
    else:
        z = 0.0
    return [x, y, z]


# ---------------------------------------------------------------------------
# Lectura de geometry.h5
# ---------------------------------------------------------------------------

def _read_geometry_h5(path):
    """Lee SW4_Receivers de model_geometry.h5.

    Retorna dict {name: {xyz_km, internal, is_qa}}.
    Retorna {} si el archivo no existe o no tiene SW4_Receivers.
    """
    path = Path(path)
    if not path.exists():
        return {}
    out = {}
    with h5py.File(path, "r") as f:
        if "SW4_Receivers" not in f:
            return out
        g     = f["SW4_Receivers"]
        files = [_decode(v) for v in g["file"][()]]
        xyz   = g["xyz_km"][()]
        internal = g["internal"][()]
        is_qa    = g["is_qa"][()]
        for i, name in enumerate(files):
            out[name] = {
                "xyz_km"  : xyz[i],
                "internal": bool(internal[i]),
                "is_qa"   : bool(is_qa[i]),
            }
    return out


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

# build_h5drm_from_sw4_case(
#     case_path="/mnt/deadmanschest/pxpalacios/SW4/shakermaker_2_sw4/T1_sw4_LOH1_02_Plane_Z_0",
#     output_name="motions_filtered00_test.h5drm",
#     use_filter=False,
# )

build_h5drm_from_sw4_case(
    case_path="/mnt/deadmanschest/pxpalacios/SW4/shakermaker_2_sw4/T2_sw4_LOH1_02_Plane_Z_0_gauus",
    output_name="motions_filtered00_test00.h5drm",
    use_filter=False,
    batch_size=100_000,
)
