from pathlib import Path
import csv
import io
import numpy as np

from shakermaker.sl_extensions import DRMBox, SurfaceGrid, PointCloudDRMReceiver

from .config import SW4ExportConfig
from .coordinates import CoordinateTransform
from .geometry_plot import plot_sw4_geometry
from .grid import grid_line
from .input_writer import sw4_input_text
from .materials import material_lines
from .package_h5 import write_sw4_package_h5, write_unpack_script
from .receivers import (
    domain_receiver_lines,
    model_receiver_lines,
    model_receiver_surface_lines,
    topography_receiver_lines,
    topography_z0_receiver_lines,
    _values_between_topography_and_z0,
)
from .sources import source_file_text, source_rows, sw4_source_lines
from .topography import (
    SEPARATOR,
    print_active_geometry_bounds,
    print_domain_diagnostics,
    print_topography_diagnostics,
    read_cartesian_topography,
    rebuild_cartesian_topography,
    rotate_topography_to_shakermaker,
    extend_topography_to_domain,
    cartesian_topography_text,
)


class SW4Exporter:
    """Exporter orchestrator for compact SW4 HDF5 transport packages."""

    def __init__(self, model, config: SW4ExportConfig):
        self.model = model
        self.config = config
        self.base_path = Path(config.path).resolve()
        self.exports_path = self.base_path / "shakermakerexports"
        self.sw4_path = self.base_path / "sw4"
        self.sources_path = self.sw4_path / "sources"
        self.topo_path = self.sw4_path / "topo"
        self.input_file = self.sw4_path / "shakermaker2sw4.in"
        self.package_h5 = self.exports_path / self.config.h5_export_name
        self.unpack_script = self.exports_path / "unpack_sw4_package.py"

    def write(self):
        self.exports_path.mkdir(parents=True, exist_ok=True)
        stale_structure = self.exports_path / "PACKAGE_STRUCTURE.txt"
        if stale_structure.is_file():
            stale_structure.unlink()

        topo_line = None
        topo_points = None
        topo_points_local = None
        topo_points_sw4 = None
        topo_nx = topo_ny = None
        local_topo = None
        topo_nx_sw4 = topo_ny_sw4 = None
        topo_original_bounds = None

        if self.config.topo_file is not None:
            topo_nx, topo_ny, topo_points = read_cartesian_topography(self.config.topo_file)
            topo_points = rotate_topography_to_shakermaker(topo_points)
            topo_nx, topo_ny, topo_points = rebuild_cartesian_topography(topo_points)

        original_points = self._active_geometry_points_m(topo_points)
        x_domain, y_domain, z_domain, domain_origin = self._resolve_domain(original_points)
        transform = CoordinateTransform(domain_origin)
        self._store_domain_in_config(x_domain, y_domain, z_domain, transform)

        if topo_points is not None:
            topo_points_local = np.array([transform.from_original_m_to_sw4_m(p) for p in topo_points])
            topo_nx_sw4, topo_ny_sw4, topo_points_sw4 = extend_topography_to_domain(
                topo_nx, topo_ny, topo_points_local, x_domain, y_domain)
            topo_points_shaker = np.asarray(
                [transform.to_original_m(point) for point in topo_points_sw4],
                dtype=float,
            )
            local_topo = self.topo_path / f"{Path(self.config.topo_file).stem}_local{Path(self.config.topo_file).suffix}"
            print_topography_diagnostics(
                topo_points, topo_points_local, x_domain, y_domain, z_domain,
                h=self.config.h, topo_nx=topo_nx, topo_ny=topo_ny, topo_zmax=self.config.topo_zmax)
            topo_original_bounds = self._topography_bounds_array(topo_points)
            topo_line = self._topography_line(local_topo, topo_points_local)
            if self.config.topo_zmax is not None:
                topo_line += f" zmax={float(self.config.topo_zmax):.16g}"
        else:
            topo_points_shaker = None
            print_domain_diagnostics(x_domain, y_domain, z_domain, h=self.config.h)

        rows = source_rows(self.model, transform)
        source_points = np.array([[row["x_sw4_m"], row["y_sw4_m"], row["z_sw4_m"]] for row in rows], dtype=float)

        has_qa = isinstance(self.model._receivers, (DRMBox, SurfaceGrid, PointCloudDRMReceiver))
        station_count = self.model._receivers.nstations
        n_drm_stations = station_count - 1 if has_qa else station_count
        qa_index = n_drm_stations if has_qa else -1

        receiver_lines = []
        receiver_records = []
        rec_index = 1
        topo_xy_z0 = self._topography_xy_at_z0(topo_points_local)
        active_points = [source_points]

        if self.config.shakermaker_stations:
            lines = model_receiver_lines(
                self.model._receivers, transform, self.config.station_prefix,
                start_index=rec_index, topo_xy_z0=topo_xy_z0)
            if lines:
                receiver_lines.append("# ShakerMaker stations")
                receiver_lines += lines
                receiver_records += self._model_receiver_records(
                    transform, start_index=rec_index, topo_xy_z0=topo_xy_z0,
                    qa_index=qa_index)
                rec_index += len(lines)

        if self.config.shakermaker_stations_to_surface:
            surf_lines = model_receiver_surface_lines(
                self.model._receivers, transform, self.config.station_prefix,
                start_index=rec_index)
            if surf_lines:
                receiver_lines.append("# ShakerMaker stations Surface")
                receiver_lines += surf_lines
                receiver_records += self._model_receiver_surface_records(
                    transform, topo_points_local, start_index=rec_index,
                    qa_index=qa_index)
                rec_index += len(surf_lines)

        receiver_points = np.array(
            [transform.from_shakermaker_km_to_sw4_m(station.x) for station in self.model._receivers],
            dtype=float)
        active_points.append(receiver_points)

        print_active_geometry_bounds(np.vstack(active_points))

        if topo_points_local is not None:
            lines = topography_receiver_lines(
                topo_points_local, start_index=rec_index, prefix=self.config.station_prefix)
            if lines:
                receiver_lines.append("# Topography surface stations (depth=0)")
                receiver_lines += lines
                receiver_records += self._topography_surface_records(
                    transform, topo_points_local, start_index=rec_index)
                rec_index += len(lines)

            if self.config.write_topography_z0_stations:
                lines = topography_z0_receiver_lines(
                    topo_points_local, h=self.config.h,
                    start_index=rec_index, prefix=self.config.station_prefix)
                if lines:
                    receiver_lines.append("# Between topography and z=0")
                    receiver_lines += lines
                    receiver_records += self._topography_z0_records(
                        transform, topo_points_local, start_index=rec_index)
                    rec_index += len(lines)

        if self.config.domain_sw4:
            domain_size = [
                self.config.domain_sw4_x if self.config.domain_sw4_x is not None else x_domain,
                self.config.domain_sw4_y if self.config.domain_sw4_y is not None else y_domain,
                self.config.domain_sw4_z if self.config.domain_sw4_z is not None else z_domain,
            ]
            domain_origin_xy = self._domain_sw4_origin_xy(domain_size, x_domain, y_domain)
            lines = domain_receiver_lines(
                self.config.h,
                domain_size,
                start_index=rec_index,
                prefix=self.config.station_prefix,
                topo_xy_z0=topo_xy_z0,
                origin_xy=domain_origin_xy)
            if lines:
                receiver_lines.append("# SW4 domain grid stations")
                receiver_lines += lines
                receiver_records += self._records_from_rec_lines(
                    transform, lines, kind="sw4_domain", start_model_index=-1)
                rec_index += len(lines)

        grid = grid_line(self.config.h, x_domain, y_domain, z_domain)
        materials = material_lines(self.model._crust)
        source_lines = sw4_source_lines(rows, self.config.m0)
        input_text = sw4_input_text(
            grid,
            self.config.tmax,
            self.config.fileio_path,
            self.config.supergrid_gp,
            materials,
            source_lines,
            receiver_lines,
            topo_line,
        )

        paths = self.paths()
        file_payloads = {"sw4/shakermaker2sw4.in": input_text}
        for row in rows:
            file_payloads[f"sw4/sources/{Path(row['dfile']).name}"] = source_file_text(row)
        file_payloads["sw4/sources/sources_summary.csv"] = self._sources_summary_text(rows)
        topo_relpath = None
        if local_topo is not None:
            topo_relpath = f"sw4/topo/{local_topo.name}"
            file_payloads[topo_relpath] = cartesian_topography_text(topo_nx_sw4, topo_ny_sw4, topo_points_sw4)
        write_sw4_package_h5(
            paths["package_h5"],
            self.model,
            self.config,
            paths,
            rows,
            receiver_records,
            file_payloads,
            input_text,
            topography_relpath=topo_relpath,
            topography_shape=(topo_nx_sw4, topo_ny_sw4) if local_topo is not None else None,
            topography_points=topo_points_shaker,
            topography_original_bounds=topo_original_bounds,
        )
        write_unpack_script(paths["unpack_script"])

        self.model.sw4_export_paths = paths
        self.model.sw4_export_config = self.config

        print(SEPARATOR)
        print("SW4 export files")
        print(SEPARATOR)
        print(f"SW4 package   : {paths['package_h5']}")
        print(f"Unpack script : {paths['unpack_script']}")
        print(SEPARATOR)

        if self.config.plot_geometry:
            plot_sw4_geometry(paths["package_h5"], origin_m=transform.domain_origin_m)
        if self.config.plot_geometry_sw4:
            plot_sw4_geometry(paths["package_h5"])

    def _active_geometry_points_m(self, topo_points=None):
        points = []
        points.extend(np.asarray(psource.x, dtype=float) * 1000.0 for psource in self.model._source)
        points.extend(np.asarray(station.x, dtype=float) * 1000.0 for station in self.model._receivers)
        if topo_points is not None:
            points.extend(np.asarray(topo_points, dtype=float))
        if not points:
            raise ValueError("Cannot build SW4 domain from empty geometry.")
        return np.asarray(points, dtype=float)

    def _resolve_domain(self, points):
        if self.config.z_domain is None:
            raise ValueError("size_domain must provide a z value.")

        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = 0.5 * (mins + maxs)
        x_domain = self._domain_length(0, mins, maxs)
        y_domain = self._domain_length(1, mins, maxs)
        z_domain = float(self.config.z_domain)
        origin = np.array([
            center[0] - 0.5 * x_domain,
            center[1] - 0.5 * y_domain,
            0.0,
        ], dtype=float)
        return x_domain, y_domain, z_domain, origin

    def _domain_length(self, axis, mins, maxs):
        configured = self.config.x_domain if axis == 0 else self.config.y_domain
        extent = float(maxs[axis] - mins[axis])
        if configured is None:
            length = np.ceil(max(extent, float(self.config.h)) / float(self.config.h)) * float(self.config.h)
        else:
            length = float(configured)
            if extent > length + 1.0e-9:
                axis_name = "x" if axis == 0 else "y"
                raise ValueError(f"size_domain {axis_name}={length} does not contain active geometry extent {extent}.")
        return float(length)

    def _store_domain_in_config(self, x_domain, y_domain, z_domain, transform):
        self.config.x_domain = float(x_domain)
        self.config.y_domain = float(y_domain)
        self.config.z_domain = float(z_domain)
        self.config.x_origin = float(transform.origin_m[0])
        self.config.y_origin = float(transform.origin_m[1])
        self.config.z_origin = float(transform.origin_m[2])

    def _topography_line(self, local_topo, topo_points_local):
        xmin = float(topo_points_local[:, 0].min())
        xmax = float(topo_points_local[:, 0].max())
        ymin = float(topo_points_local[:, 1].min())
        ymax = float(topo_points_local[:, 1].max())
        bounds = (
            f"# ShakerMaker topography_original_bounds "
            f"xmin={xmin:.16g} xmax={xmax:.16g} ymin={ymin:.16g} ymax={ymax:.16g}"
        )
        line = f"topography input=cartesian file=topo/{local_topo.name}"
        return bounds + "\n" + line

    def _topography_bounds_array(self, topo_points_local):
        return np.asarray([
            float(topo_points_local[:, 0].min()),
            float(topo_points_local[:, 0].max()),
            float(topo_points_local[:, 1].min()),
            float(topo_points_local[:, 1].max()),
        ], dtype=float)

    def _topography_xy_at_z0(self, topo_points_local):
        out = set()
        if topo_points_local is None:
            return out
        h_tol = 0.5 * float(self.config.h)
        for x, y, z in topo_points_local:
            if abs(float(z)) < h_tol:
                out.add((round(float(x)), round(float(y))))
        return out

    def _model_receiver_records(self, transform, start_index, topo_xy_z0, qa_index):
        records = []
        offset = int(start_index)
        for i_station, station in enumerate(self.model._receivers):
            xyz_m = transform.from_shakermaker_km_to_sw4_m(station.x)
            if abs(float(xyz_m[2])) < 1.0e-9 and (
                    round(float(xyz_m[0])), round(float(xyz_m[1]))) in topo_xy_z0:
                continue
            records.append({
                "file": f"{self.config.station_prefix}{offset:05d}",
                "kind": "shakermaker",
                "xyz_km": np.asarray(station.x, dtype=float),
                "internal": bool(station.is_internal),
                "is_qa": bool(i_station == qa_index),
                "model_index": int(i_station),
                "metadata": repr(station.metadata),
            })
            offset += 1
        return records

    def _model_receiver_surface_records(self, transform, topo_points_local, start_index, qa_index):
        records = []
        offset = int(start_index)
        for i_station, station in enumerate(self.model._receivers):
            xyz_m = transform.from_shakermaker_km_to_sw4_m(station.x)
            topo_z = self._topography_z_at(topo_points_local, xyz_m[0], xyz_m[1])
            records.append({
                "file": f"{self.config.station_prefix}{offset:05d}",
                "kind": "shakermaker_surface",
                "xyz_km": transform.to_original_m(np.asarray([xyz_m[0], xyz_m[1], -topo_z], dtype=float)) / 1000.0,
                "internal": bool(station.is_internal),
                "is_qa": bool(i_station == qa_index),
                "model_index": int(i_station),
                "metadata": "depth=0; " + repr(station.metadata),
            })
            offset += 1
        return records

    def _topography_z_at(self, topo_points_local, x, y):
        if topo_points_local is None or len(topo_points_local) == 0:
            return 0.0
        topo_points_local = np.asarray(topo_points_local, dtype=float)
        d2 = (topo_points_local[:, 0] - float(x)) ** 2 + (topo_points_local[:, 1] - float(y)) ** 2
        return float(topo_points_local[int(np.argmin(d2)), 2])

    def _topography_surface_records(self, transform, topo_points_local, start_index):
        records = []
        for offset, (x, y, topo_z) in enumerate(
                np.asarray(topo_points_local, dtype=float), start=int(start_index)):
            records.append({
                "file": f"{self.config.station_prefix}{offset:05d}",
                "kind": "topography_surface",
                "xyz_km": transform.to_original_m(np.asarray([x, y, -topo_z], dtype=float)) / 1000.0,
                "internal": True,
                "is_qa": False,
                "model_index": -1,
                "metadata": "depth=0",
            })
        return records

    def _topography_z0_records(self, transform, topo_points_local, start_index):
        records = []
        offset = int(start_index)
        for x, y, topo_z in np.asarray(topo_points_local, dtype=float):
            for z in _values_between_topography_and_z0(topo_z, self.config.h):
                records.append({
                    "file": f"{self.config.station_prefix}{offset:05d}",
                    "kind": "topography_to_z0",
                    "xyz_km": transform.to_original_m(np.asarray([x, y, z], dtype=float)) / 1000.0,
                    "internal": True,
                    "is_qa": False,
                    "model_index": -1,
                    "metadata": "between topography and z=0",
                })
                offset += 1
        return records

    def _records_from_rec_lines(self, transform, lines, kind, start_model_index):
        records = []
        for line in lines:
            values = self._parse_rec_line(line)
            xyz_local_m = np.asarray(
                [values["x"], values["y"], values.get("z", 0.0)],
                dtype=float,
            )
            records.append({
                "file": values["file"],
                "kind": kind,
                "xyz_km": transform.to_original_m(xyz_local_m) / 1000.0,
                "internal": True,
                "is_qa": False,
                "model_index": int(start_model_index),
                "metadata": "",
            })
        return records

    def _parse_rec_line(self, line):
        values = {}
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key == "file":
                values[key] = value
            elif key in ("x", "y", "z", "depth"):
                values[key] = float(value)
        return values

    def _domain_sw4_origin_xy(self, domain_size, x_domain, y_domain):
        lx, ly, _lz = [float(value) for value in domain_size]
        if lx > float(x_domain) + 1.0e-9 or ly > float(y_domain) + 1.0e-9:
            raise ValueError("domain_sw4_size x/y must fit inside size_domain.")
        return (
            0.5 * (float(x_domain) - lx),
            0.5 * (float(y_domain) - ly),
        )

    def paths(self):
        return {
            "base": self.base_path,
            "exports": self.exports_path,
            "sw4": self.sw4_path,
            "sources": self.sources_path,
            "topo": self.topo_path,
            "package_h5": self.package_h5,
            "unpack_script": self.unpack_script,
            "input": self.input_file,
        }

    def _sources_summary_text(self, rows):
        headers = [
            "id", "x_km", "y_km", "z_km", "x_m", "y_m", "z_m",
            "x_sw4_m", "y_sw4_m", "z_sw4_m",
            "strike_deg", "dip_deg", "rake_deg", "trigger_time_s",
            "stf_local_t0_s", "dt", "stf_type", "dfile",
        ]
        f = io.StringIO()
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return f.getvalue()
