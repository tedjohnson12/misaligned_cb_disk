"""
Tests for params.py
"""
import pytest

from misaligned_cb_disk import params

def test_star_mass():
    cases = (
        (1,1,0,1),
        (1,0,1,0),
        (2,0.5,1,1),
        (0,1,0,0),
        (0,0,0,0)
    )
    for (mb,fb,m1,m2) in cases:
        _m1, _m2 = params.get_star_masses(mb,fb)
        assert _m1 == pytest.approx(m1,abs=1e-6), 'Failed for m1'
        assert _m2 == pytest.approx(m2,abs=1e-6), 'Failed for m2'

def test_Binary_star_masses():
    cases = (
        (1,1,0,1),
        (1,0,1,0),
        (2,0.5,1,1),
        (0,1,0,0),
        (0,0,0,0)
    )
    for (mb,fb,m1,m2) in cases:
        binary = params.Binary(mb,fb,0,0,0)
        assert binary.m1 == pytest.approx(m1,abs=1e-6), 'Failed for m1'
        assert binary.m2 == pytest.approx(m2,abs=1e-6), 'Failed for m2'
