"""
Tests for misaligned_cb_disk.system
"""
import rebound
import numpy as np

from misaligned_cb_disk import system, params


def test_cross():
    a = np.array([1,0,0]).T
    b = np.array([1,0,0]).T
    assert np.all(system.cross(a,b).T == np.array([0,0,0]))
    
    a = np.array([1,0,0]).T
    b = np.array([0,1,0]).T
    assert np.all(system.cross(a,b).T == np.array([0,0,1]))
    
    a = np.array([1,0,0]).T
    b = np.array([0,0,1]).T
    assert np.all(system.cross(a,b).T == np.array([0,-1,0]))
    
    a = np.array([
        [0,1,0],
        [1,0,0]
    ]).T
    b = np.array([
        [0,1,0],
        [0,1,0]
    ]).T
    assert np.all(system.cross(a,b).T == np.array([[0,0,0],[0,0,1]]))

def test_dot():
    a = np.array([
        [0,1,0],
        [1,0,0]
    ]).T
    b = np.array([
        [0,1,0],
        [0,1,0]
    ]).T
    assert np.all(system.dot(a,b) == np.array([1,0]))
    
   

def test_system_init():
    binary = params.Binary(2,0.5,1,0)
    planet = params.Planet(0,3,0,0,0,0,0)
    sim = rebound.Simulation()
    sys = system.System(binary,planet,sim)
    assert sys.sim.particles['m1'].m == binary.mass1
    assert sys.sim.particles['m2'].m == binary.mass2
    assert sys.sim.particles['p'].m == planet.mass
    sys.sim.particles['p'].ax

def test_system_integerate():
    binary = params.Binary(2,0.5,1,0)
    planet = params.Planet(0,3,0,0,0,0,0)
    sim = rebound.Simulation()
    sys = system.System(binary,planet,sim)
    
    sys.integrate(np.linspace(0,30,15))

def test_system_integrate_orbits():
    binary = params.Binary(2,0.5,1,0)
    planet = params.Planet(0,3,0,0,0,0,0)
    sim = rebound.Simulation()
    sys = system.System(binary,planet,sim)
    
    sys.integrate_orbits(10,1)

def test_properties():
    binary = params.Binary(2,0.5,1,0)
    planet = params.Planet(0,3,0,0,0,0,0)
    sim = rebound.Simulation()
    sys = system.System(binary,planet,sim)
    
    sys.integrate_orbits(10,1)
    
    assert ~np.any(np.isnan(sys.specific_angular_momentum))
    assert ~np.any(np.isnan(sys.specific_torque))
    assert ~np.any(np.isnan(sys.inclination))
    assert ~np.any(np.isnan(sys.inclination_dot))

if __name__ in '__main__':
    test_properties()