from pathlib import Path
import csv
import numpy as np

from shakermaker.sl_extensions import DRMBox, SurfaceGrid, PointCloudDRMReceiver

from .config import SW4ExportConfig
from .coordinates import CoordinateTransform
from .grid import domain_from_topography, grid_line
from .geometry_plot import plot_sw4_geometry
from .hdf5_geometry import write_geometry_h5
from .hdf5_summary import write_summary_h5
from .input_writer import write_sw4_input
from .materials import material_lines
from .receivers import (
    model_receiver_lines,
    topography_receiver_lines,
    topography_z0_receiver_lines,
    domain_receiver_lines,
)
from .sources import source_rows, write_source_files, sw4_source_lines
from .topography import (
    SEPARATOR,
    read_cartesian_topography,
    write_cartesian_topography,
    print_topography_diagnostics,
    print_domain_diagnostics,
    print_active_geometry_bounds,
    print_coordinate_alignment,
    minimum_corner,
)


class SW4Exporter:
    """Exporter orchestrator for SW4 files and ShakerMaker HDF5 summaries."""

    def __init__(self, model, config: SW4ExportConfig):
        self.model = model
        self.config = config
        self.base_path = Path(config.path).resolve()
        self.exports_path = self.base_path / "shakermakerexports"
        self.sw4_path = self.base_path / "sw4"
        self.sources_path = self.sw4_path / "sources"
        self.topo_path = self.sw4_path / "topo"
        self.geometry_h5 = self.exports_path / "model_geometry.h5"
        self.summary_h5 = self.exports_path / "model_summary.h5"
        self.input_file = self.sw4_path / "shakermaker2sw4.in"

    def write(self):
        self.exports_path.mkdir(parents=True, exist_ok=True)
        self.sources_path.mkdir(parents=True, exist_ok=True)

        transform = CoordinateTransform(
            [self.config.x_origin, self.config.y_origin, self.config.z_origin],
            self.config.topo_reference,
            self.config.topo_target,
        )

        topo_line = None
        topo_points_local = None
        x_domain = self.config.x_domain
        y_domain = self.config.y_domain
        if self.config.topo_file is not None:
            self.topo_path.mkdir(parents=True, exist_ok=True)
            nx, ny, topo_points = read_cartesian_topography(self.config.topo_file)
            if transform.topo_reference is None or transform.topo_target is None:
                coor_origin = minimum_corner(topo_points)
                coor_target = np.array([0.0, 0.0, 0.0], dtype=float)
                transform.set_topography_transform(coor_origin, coor_target)
                self.config.coor_origin = coor_origin
                self.config.coor_target = coor_target
                self.config.topo_reference = coor_origin
                self.config.topo_target = coor_target
                print_coordinate_alignment(coor_origin, coor_target, automatic=True)
            else:
                print_coordinate_alignment(transform.topo_reference, transform.topo_target)
            if self.config.coor_target_shaker is not None:
                transform.set_shakermaker_origin_from_original(self.config.coor_target_shaker)
            topo_points_local = np.array([transform.from_original_m_to_sw4_m(p) for p in topo_points])
            topo_x_domain, topo_y_domain, _ = domain_from_topography(
                topo_points_local, self.config.h, self.config.z_domain)
            topo_xmax = float(topo_points_local[:, 0].max())
            topo_ymax = float(topo_points_local[:, 1].max())
            x_domain = topo_xmax if self.config.x_domain is None else self.config.x_domain
            y_domain = topo_ymax if self.config.y_domain is None else self.config.y_domain
            local_topo = self.topo_path / f"{Path(self.config.topo_file).stem}_local{Path(self.config.topo_file).suffix}"
            write_cartesian_topography(local_topo, nx, ny, topo_points_local)
            print_topography_diagnostics(
                topo_points, topo_points_local, x_domain, y_domain, self.config.z_domain,
                h=self.config.h, topo_nx=nx, topo_ny=ny, topo_zmax=self.config.topo_zmax)
            topo_line = f"topography input=cartesian file=topo/{local_topo.name}"
            if self.config.topo_zmax is not None:
                topo_line += f" zmax={float(self.config.topo_zmax):.16g}"
        else:
            if x_domain is None or y_domain is None or self.config.z_domain is None:
                raise ValueError("export_sw4 requires size_domain=[x, y, z] without None values.")
            print_domain_diagnostics(x_domain, y_domain, self.config.z_domain, h=self.config.h)

        rows = source_rows(self.model, transform)
        write_source_files(rows, self.sources_path)
        self._write_sources_summary(rows)
        source_points = np.array([[row["x_sw4_m"], row["y_sw4_m"], row["z_sw4_m"]] for row in rows], dtype=float)

        has_qa = isinstance(self.model._receivers, (DRMBox, SurfaceGrid, PointCloudDRMReceiver))
        station_count = self.model._receivers.nstations
        n_drm_stations = station_count - 1 if has_qa else station_count
        qa_index = n_drm_stations if has_qa else -1

        # Build set of topo (x,y) positions where topo height ≈ 0 (flat ground at datum).
        # Used to avoid duplicating z=0 stations from other groups at those positions.
        topo_xy_z0 = set()
        if topo_points_local is not None:
            h_tol = 0.5 * float(self.config.h)
            for tx, ty, tz in topo_points_local:
                if abs(float(tz)) < h_tol:
                    topo_xy_z0.add((round(float(tx)), round(float(ty))))

        receiver_lines = []
        rec_index = 1
        active_points = [source_points]

        # ShakerMaker model stations (z= absolute coordinates)
        if self.config.shakermaker_stations:
            lines = model_receiver_lines(
                self.model._receivers, transform, self.config.station_prefix,
                start_index=rec_index, topo_xy_z0=topo_xy_z0,
            )
            if lines:
                receiver_lines.append("# ShakerMaker stations")
                receiver_lines += lines
                rec_index += len(lines)
            receiver_points = np.array(
                [transform.from_shakermaker_km_to_sw4_m(station.x) for station in self.model._receivers],
                dtype=float,
            )
            active_points.append(receiver_points)

        print_active_geometry_bounds(np.vstack(active_points))

        # Topography free-surface stations (depth=0 — the only use of depth= in the .in)
        if topo_points_local is not None:
            lines = topography_receiver_lines(
                topo_points_local,
                start_index=rec_index,
                prefix=self.config.station_prefix,
            )
            if lines:
                receiver_lines.append("# Topography surface stations (depth=0)")
                receiver_lines += lines
                rec_index += len(lines)

            # Stations between topography surface and z=0 plane (z= negative values)
            if self.config.write_topography_z0_stations:
                lines = topography_z0_receiver_lines(
                    topo_points_local,
                    h=self.config.h,
                    start_index=rec_index,
                    prefix=self.config.station_prefix,
                )
                if lines:
                    receiver_lines.append("# Between topography and z=0 (z= negative)")
                    receiver_lines += lines
                    rec_index += len(lines)

        # Regular grid receivers inside a sub-domain (z= positive = depth below datum)
        if self.config.domain_sw4 and self.config.domain_sw4_size is not None:
            lines = domain_receiver_lines(
                self.config.h,
                [self.config.domain_sw4_x, self.config.domain_sw4_y, self.config.domain_sw4_z],
                start_index=rec_index,
                prefix=self.config.station_prefix,
                topo_xy_z0=topo_xy_z0,
            )
            if lines:
                receiver_lines.append("# SW4 domain grid stations (z= positive = depth)")
                receiver_lines += lines
                rec_index += len(lines)

        write_sw4_input(
            self.input_file,
            grid_line(self.config.h, x_domain, y_domain, self.config.z_domain),
            self.config.tmax,
            self.config.fileio_path,
            self.config.supergrid_gp,
            material_lines(self.model._crust),
            sw4_source_lines(rows, self.config.m0),
            receiver_lines,
            topo_line,
        )

        self.config.x_domain = x_domain
        self.config.y_domain = y_domain
        paths = self.paths()
        write_geometry_h5(
            self.geometry_h5, self.model, transform, rows[0]["dt"], self.config.tmax,
            self.config, has_qa, n_drm_stations, qa_index)
        write_summary_h5(
            self.summary_h5, self.model, self.config, paths, rows,
            transform, has_qa, n_drm_stations, qa_index)

        self.model.sw4_export_paths = paths
        self.model.sw4_export_config = self.config

        print(SEPARATOR)
        print("SW4 export files")
        print(SEPARATOR)
        print(f"SW4 folder    : {self.sw4_path}")
        print(f"Geometry HDF5 : {self.geometry_h5}")
        print(f"Summary HDF5  : {self.summary_h5}")
        print(f"SW4 input     : {self.input_file}")
        print(SEPARATOR)

        if self.config.plot_geometry:
            plot_sw4_geometry(self.input_file)

    def paths(self):
        return {
            "base": self.base_path,
            "exports": self.exports_path,
            "sw4": self.sw4_path,
            "sources": self.sources_path,
            "geometry_h5": self.geometry_h5,
            "summary_h5": self.summary_h5,
            "input": self.input_file,
        }

    def _write_sources_summary(self, rows):
        path = self.sources_path / "sources_summary.csv"
        headers = [
            "id", "x_km", "y_km", "z_km", "x_m", "y_m", "z_m",
            "x_sw4_m", "y_sw4_m", "z_sw4_m",
            "strike_deg", "dip_deg", "rake_deg", "trigger_time_s",
            "stf_local_t0_s", "dt", "stf_type", "dfile",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
