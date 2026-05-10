import numpy as np


class CoordinateTransform:
    """Map original ShakerMaker/topography coordinates to the local SW4 box."""

    def __init__(self, domain_origin_m):
        self.domain_origin_m = np.asarray(domain_origin_m, dtype=float)
        self.origin_m = -self.domain_origin_m
        self.origin_km = self.origin_m / 1000.0

    def from_shakermaker_km_to_sw4_m(self, xyz_km):
        return np.asarray(xyz_km, dtype=float) * 1000.0 - self.domain_origin_m

    def from_shakermaker_km_to_sw4_km(self, xyz_km):
        return self.from_shakermaker_km_to_sw4_m(xyz_km) / 1000.0

    def from_original_m_to_sw4_m(self, xyz_m):
        return np.asarray(xyz_m, dtype=float) - self.domain_origin_m

    def from_original_m_to_sw4_km(self, xyz_m):
        return self.from_original_m_to_sw4_m(xyz_m) / 1000.0

    def to_original_m(self, xyz_sw4_m):
        return np.asarray(xyz_sw4_m, dtype=float) + self.domain_origin_m
