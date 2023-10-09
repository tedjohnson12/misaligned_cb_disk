import rebound
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from misaligned_cb_disk import params, system, utils

mb = 1
fb = 0.5
ab = 0.2

mp = 1e-3
ap = 1

n_orbits = 10


def run(i:float,e:float)->system.System:
    binary = params.Binary(mb,fb,ab,e)
    planet = params.Planet(mp,ap,i,np.pi/2,0,0,0)
    sim = rebound.Simulation()
    sys = system.System(binary,planet,sim)
    sys.integrate_to_get_state(step=5,capture_freq=2)
    
    return sys

def run_full(i:float,e:float)->system.System:
    binary = params.Binary(mb,fb,ab,e)
    planet = params.Planet(mp,ap,i,np.pi/2,0,0,0)
    sim = rebound.Simulation()
    sys = system.System(binary,planet,sim)
    sys.integrate_orbits(n_orbits=200,capture_freq=2)
    
    return sys

if __name__ in '__main__':

    fig,ax = plt.subplots(1,2)
    
    incs = np.linspace(0.1,0.99,11) * np.pi
    e = 0.4
    for i in incs:
        sys = run(i,e)
        state = sys.state
        color = utils.STATE_COLORS[state]
        utils.phase_diag(sys,ax[0],c=color)
        sys = run_full(i,e)
        utils.phase_diag(sys,ax[1],c=color)
    
    ax[0].set_aspect('equal')
    ax[0].set_xlim(ax[1].get_xlim())
    ax[0].set_ylim(ax[1].get_ylim())
    ax[1].set_aspect('equal')
    ax[0].set_ylabel('$i \\sin{\\Omega}$')
    
    fig.text(0.5,0.15,'$i \\cos{\\Omega}$')
    
    outfile = Path(__file__).parent / 'output' / 'multiple.png'
    
    fig.savefig(outfile,facecolor='w')