from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class SW4ExportConfig:
    """Configuration for exporting a ShakerMaker model to SW4 files."""

    path: str | Path
    h: float = 50.0
    x_domain: float = 0.0
    y_domain: float = 0.0
    z_domain: float = 0.0
    x_origin: float = 0.0
    y_origin: float = 0.0
    z_origin: float = 0.0
    tmax: float = 50.0
    m0: float = 1.0
    size_domain: Optional[Sequence[float]] = None
    coor_target_shaker: Optional[Sequence[float]] = None
    coor_origin_topo: Optional[Sequence[float]] = None
    coor_target_topo: Optional[Sequence[float]] = None
    coor_origin: Optional[Sequence[float]] = None
    coor_target: Optional[Sequence[float]] = None
    fileio_path: str = "shakermaker2sw4_fileio"
    supergrid_gp: int = 30
    station_prefix: str = "sf"
    topo_file: Optional[str | Path] = None
    topo_reference: Optional[Sequence[float]] = None
    topo_target: Optional[Sequence[float]] = None
    topo_zmax: Optional[float] = None
    write_topography_stations: bool = False
    depth_from_topography: bool = False
    shakermaker_stations: bool = True
    plot_geometry: bool = False

    def __post_init__(self):
        if self.size_domain is not None:
            self.x_domain, self.y_domain, self.z_domain = _as_xyz(self.size_domain, "size_domain")

        shaker_target = self.coor_target_shaker
        if shaker_target is None:
            shaker_target = self.coor_target
        if shaker_target is not None:
            self.x_origin, self.y_origin, self.z_origin = _as_xyz(shaker_target, "coor_target_shaker")

        if self.coor_origin_topo is not None:
            self.topo_reference = self.coor_origin_topo
        elif self.coor_origin is not None and self.topo_reference is None:
            self.topo_reference = self.coor_origin

        if self.coor_target_topo is not None:
            self.topo_target = self.coor_target_topo


def _as_xyz(values, name):
    if len(values) != 3:
        raise ValueError(f"{name} must have three values: [x, y, z].")
    return _optional_float(values[0]), _optional_float(values[1]), _optional_float(values[2])


def _optional_float(value):
    return None if value is None else float(value)
