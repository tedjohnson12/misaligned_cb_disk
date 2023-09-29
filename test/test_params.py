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
    for (Mb,fb,M1,M2) in cases:
        _M1, _M2 = params.get_star_masses(Mb,fb)
        assert _M1 == pytest.approx(M1,abs=1e-6), 'Failed for M1'
        assert _M2 == pytest.approx(M2,abs=1e-6), 'Failed for M2'
        
