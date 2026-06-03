"""Tests for supergrid-aware SW4 domain sizing in the exporter."""

import types

import numpy as np
import pytest

from shakermaker.crustmodel import CrustModel
from shakermaker.sw4_exporter.config import SW4ExportConfig
from shakermaker.sw4_exporter.exporter import SW4Exporter, _snap_up_to_h


# Fault-like extremes (sources + receivers), taken from a real export:
#   x in [355.49, 30644.51], y in [2236.96, 28763.04], deepest source z=13061.95
PTS = np.array([[355.49, 2236.96, 0.0], [30644.51, 28763.04, 13061.95]])
X_EXTENT = 30644.51 - 355.49
Y_EXTENT = 28763.04 - 2236.96
SPONGE = 30 * 20  # supergrid_gp * h


def _crust():
    c = CrustModel(4)
    c.add_layer(0.200, 1.32, 0.75, 2.40, 1000.0, 1000.0)
    c.add_layer(0.800, 2.75, 1.57, 2.50, 1000.0, 1000.0)
    c.add_layer(14.50, 5.50, 3.14, 2.50, 1000.0, 1000.0)
    c.add_layer(0.000, 7.00, 4.00, 2.67, 1000.0, 1000.0)
    return c


def _exporter(tmp_path, size):
    model = types.SimpleNamespace(_crust=_crust(), _source=[], _receivers=[])
    cfg = SW4ExportConfig(path=str(tmp_path), h=20, size_domain=size)
    return SW4Exporter(model, cfg)


def test_snap_up_to_h():
    assert _snap_up_to_h(600, 20) == 600
    assert _snap_up_to_h(601, 20) == 620
    assert _snap_up_to_h(25312, 25) == 25325


def test_auto_all_none(tmp_path):
    xd, yd, zd, _origin = _exporter(tmp_path, [None, None, None])._resolve_domain(PTS)
    for value in (xd, yd, zd):
        assert value % 20 == 0
    # geometry centred -> clearance per side must exceed the supergrid sponge
    assert (xd - X_EXTENT) / 2 >= SPONGE
    assert (yd - Y_EXTENT) / 2 >= SPONGE
    # z reaches below the deepest interface (15500) plus the bottom sponge
    assert zd >= 15500 + SPONGE


def test_explicit_z_too_shallow_raises(tmp_path):
    with pytest.raises(ValueError, match="too shallow"):
        _exporter(tmp_path, [None, None, 15000])._resolve_domain(PTS)


def test_explicit_lateral_no_room_raises(tmp_path):
    with pytest.raises(ValueError, match="supergrid"):
        _exporter(tmp_path, [31000, 31000, 15000])._resolve_domain(PTS)


def test_non_multiple_value_snaps_with_warning(tmp_path):
    # Large enough to clear the supergrid, but not a multiple of h.
    with pytest.warns(UserWarning, match="multiple of h"):
        xd, _yd, _zd, _origin = _exporter(tmp_path, [33001, None, None])._resolve_domain(PTS)
    assert xd % 20 == 0
    assert xd >= 33001
