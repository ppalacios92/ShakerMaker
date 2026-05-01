from pathlib import Path
import numpy as np


def source_rows(model, transform):
    rows = []
    for i_source, psource in enumerate(model._source):
        x_km = np.asarray(psource.x, dtype=float)
        x_m = x_km * 1000.0
        x_sw4_m = transform.from_shakermaker_km_to_sw4_m(x_km)
        angles_deg = np.degrees(np.asarray(psource.angles, dtype=float))
        stf = psource.stf
        dt = float(stf.dt)
        if dt <= 0:
            raise ValueError(
                f"Source {i_source} source-time function has no valid dt. "
                "Set psource.stf.dt before exporting to SW4.")
        rows.append({
            "id": i_source,
            "x_km": x_km[0], "y_km": x_km[1], "z_km": x_km[2],
            "x_m": x_m[0], "y_m": x_m[1], "z_m": x_m[2],
            "x_sw4_m": x_sw4_m[0], "y_sw4_m": x_sw4_m[1], "z_sw4_m": x_sw4_m[2],
            "strike_deg": angles_deg[0], "dip_deg": angles_deg[1], "rake_deg": angles_deg[2],
            "trigger_time_s": float(psource.tt),
            "stf_local_t0_s": float(stf.t[0]) if len(stf.t) else 0.0,
            "dt": dt,
            "stf_type": type(stf).__name__,
            "dfile": f"sw4/sources/source_{i_source:06d}.txt",
            "stf": stf,
        })
    return rows


def write_source_files(rows, sources_path):
    sources_path = Path(sources_path)
    sources_path.mkdir(parents=True, exist_ok=True)
    for row in rows:
        dfile_path = sources_path / Path(row["dfile"]).name
        data = np.asarray(row["stf"].data, dtype=float).reshape(-1)
        with dfile_path.open("w", encoding="utf-8") as f:
            f.write(f"{row['trigger_time_s']:.16g} {row['dt']:.16g} {len(data)}\n")
            for value in data:
                f.write(f"{float(value):.16g}\n")


def sw4_source_lines(rows, m0):
    lines = []
    for row in rows:
        dfile = f"sources/{Path(row['dfile']).name}"
        lines.append(
            f"source x={float(row['x_sw4_m']):.16g} "
            f"y={float(row['y_sw4_m']):.16g} "
            f"z={float(row['z_sw4_m']):.16g} "
            f"m0={float(m0):.16g} "
            f"strike={float(row['strike_deg']):.16g} "
            f"dip={float(row['dip_deg']):.16g} "
            f"rake={float(row['rake_deg']):.16g} "
            f"t0={float(row['trigger_time_s']):.16g} "
            f"dfile={dfile}"
        )
    return lines
