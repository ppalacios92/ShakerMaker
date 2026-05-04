import numpy as np


class CoordinateTransform:
    """Small coordinate helper for SW4 local coordinates."""

    def __init__(self, origin_m, topo_reference=None, topo_target=None):
        self.origin_m = np.asarray(origin_m, dtype=float)
        self.local_offset_m = np.zeros(3, dtype=float)
        self.origin_km = self.origin_m / 1000.0
        self.topo_reference = None if topo_reference is None else np.asarray(topo_reference, dtype=float)
        self.topo_target = None if topo_target is None else np.asarray(topo_target, dtype=float)

    def set_topography_transform(self, reference, target):
        self.topo_reference = np.asarray(reference, dtype=float)
        self.topo_target = np.asarray(target, dtype=float)

    def set_shakermaker_origin_from_original(self, origin_original_m):
        if self.topo_reference is None or self.topo_target is None:
            raise ValueError("topo_reference and topo_target are required before setting ShakerMaker origin from original coordinates.")
        self.origin_m = np.asarray(origin_original_m, dtype=float) - self.topo_reference + self.topo_target
        self.origin_km = (self.origin_m + self.local_offset_m) / 1000.0

    def set_local_offset(self, offset_m):
        self.local_offset_m = np.asarray(offset_m, dtype=float)
        self.origin_km = (self.origin_m + self.local_offset_m) / 1000.0

    def from_shakermaker_km_to_sw4_m(self, xyz_km):
        return np.asarray(xyz_km, dtype=float) * 1000.0 + self.origin_m + self.local_offset_m

    def from_shakermaker_km_to_sw4_km(self, xyz_km):
        return np.asarray(xyz_km, dtype=float) + self.origin_km

    def from_original_m_to_sw4_m(self, xyz_m):
        if self.topo_reference is None or self.topo_target is None:
            raise ValueError("topo_reference and topo_target are required for original-meter transforms.")
        return np.asarray(xyz_m, dtype=float) - self.topo_reference + self.topo_target + self.local_offset_m

    def from_original_m_to_sw4_km(self, xyz_m):
        return self.from_original_m_to_sw4_m(xyz_m) / 1000.0
