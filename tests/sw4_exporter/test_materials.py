"""Tests for SW4 material-line generation, including harmonic interface nodes."""

import pytest

from shakermaker.crustmodel import CrustModel
from shakermaker.sw4_exporter.materials import deepest_interface, material_lines


def _loh_crust():
    """LOH-style 4-layer crust (layer / layer / thick layer / half-space)."""
    c = CrustModel(4)
    c.add_layer(0.200, 1.32, 0.75, 2.40, 1000.0, 1000.0)
    c.add_layer(0.800, 2.75, 1.57, 2.50, 1000.0, 1000.0)
    c.add_layer(14.50, 5.50, 3.14, 2.50, 1000.0, 1000.0)
    c.add_layer(0.000, 7.00, 4.00, 2.67, 1000.0, 1000.0)
    return c


def _parse(line):
    out = {}
    for token in line.split():
        if "=" in token:
            key, value = token.split("=")
            out[key] = value
    return out


def test_deepest_interface():
    assert deepest_interface(_loh_crust()) == pytest.approx(15500.0)
    assert deepest_interface(CrustModel(1)) == 0.0


def test_layer_blocks_only():
    lines = material_lines(_loh_crust(), interface_blocks=False)
    assert len(lines) == 4
    assert lines[0] == "block vp=7000 vs=4000 rho=2670"
    assert "z1=1000 z2=15500" in lines[1]
    assert lines[-1] == "block vp=1320 vs=750 rho=2400 z2=200"


def test_interface_blocks_values():
    lines = material_lines(_loh_crust(), h=20, interface_blocks=True, interface_block_delta=1.0)
    # 4 layers + 3 interface nodes
    assert len(lines) == 7
    iface = {_parse(l)["z1"]: _parse(l) for l in lines[4:]}
    assert set(iface) == {"199", "999", "15499"}
    assert float(iface["199"]["vp"]) == pytest.approx(1671.92, abs=0.1)
    assert float(iface["199"]["vs"]) == pytest.approx(950.79, abs=0.1)
    assert float(iface["199"]["rho"]) == pytest.approx(2450.0)
    assert float(iface["999"]["vp"]) == pytest.approx(3478.51, abs=0.1)
    assert float(iface["999"]["rho"]) == pytest.approx(2500.0)
    assert float(iface["15499"]["vp"]) == pytest.approx(6089.17, abs=0.1)
    assert float(iface["15499"]["vs"]) == pytest.approx(3477.52, abs=0.1)
    assert float(iface["15499"]["rho"]) == pytest.approx(2585.0)


def test_interface_misaligned_warns_and_skips():
    # h=30 divides none of 200 / 1000 / 15500 -> all interface nodes skipped.
    with pytest.warns(UserWarning):
        lines = material_lines(_loh_crust(), h=30, interface_blocks=True)
    assert len(lines) == 4


def test_interface_requires_h():
    with pytest.warns(UserWarning):
        lines = material_lines(_loh_crust(), h=None, interface_blocks=True)
    assert len(lines) == 4


def test_interface_delta_too_large_warns():
    with pytest.warns(UserWarning):
        material_lines(_loh_crust(), h=20, interface_blocks=True, interface_block_delta=20.0)
