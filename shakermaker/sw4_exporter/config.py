from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class SW4ExportConfig:
    """Configuration for exporting a ShakerMaker model to SW4."""

    path: str | Path
    h: float = 50.0
    x_domain: Optional[float] = None
    y_domain: Optional[float] = None
    z_domain: Optional[float] = None
    x_origin: float = 0.0
    y_origin: float = 0.0
    z_origin: float = 0.0
    tmax: float = 50.0
    m0: float = 1.0
    size_domain: Optional[Sequence[float]] = None
    fileio_path: str = "shakermaker2sw4_fileio"
    supergrid_gp: int = 30
    station_prefix: str = "sf"
    topo_file: Optional[str | Path] = None
    topo_zmax: Optional[float] = None
    write_topography_z0_stations: bool = False
    shakermaker_stations: bool = True
    shakermaker_stations_to_surface: bool = False
    domain_sw4: bool = False
    domain_sw4_size: Optional[Sequence[float]] = None
    domain_sw4_x: Optional[float] = None
    domain_sw4_y: Optional[float] = None
    domain_sw4_z: Optional[float] = None
    plot_geometry: bool = False
    plot_geometry_sw4: bool = False

    def __post_init__(self):
        if self.size_domain is not None:
            self.x_domain, self.y_domain, self.z_domain = _as_xyz(self.size_domain, "size_domain")
        if self.domain_sw4_size is not None:
            self.domain_sw4_x, self.domain_sw4_y, self.domain_sw4_z = _as_xyz(
                self.domain_sw4_size, "domain_sw4_size")


def _as_xyz(values, name):
    if len(values) != 3:
        raise ValueError(f"{name} must have three values: [x, y, z].")
    return _optional_float(values[0]), _optional_float(values[1]), _optional_float(values[2])


def _optional_float(value):
    return None if value is None else float(value)
