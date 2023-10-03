"""
Tests for params.py
"""
from rebound import Simulation, OrbitPlot
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

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
        assert binary.mass1 == pytest.approx(m1,abs=1e-6), 'Failed for m1'
        assert binary.mass2 == pytest.approx(m2,abs=1e-6), 'Failed for m2'

def test_Binary_add_to_sim():
    binary = params.Binary(2,0.5,1,0,1)
    sim = Simulation()
    binary.add_to_sim(sim)
    primary = sim.particles['m1']
    secondary = sim.particles['m2']
    assert np.sqrt(
        (secondary.x-primary.x)**2
        +(secondary.y-primary.y)**2
        +(secondary.z-primary.z)**2
    ) == pytest.approx(1,abs=1e-6), 'Failed for m2 distance from m1'

def test_Binary_sim_orbit():
    outdir = Path(__file__).parent / 'output'
    outfile = outdir / 'binary_orbit.png'
    if not outdir.exists():
        outdir.mkdir()

    binary = params.Binary(2,0.5,1,0,1)
    sim = Simulation()
    binary.add_to_sim(sim)

    OrbitPlot(sim,unitlabel='[AU]').fig.savefig(outfile,facecolor='w')
