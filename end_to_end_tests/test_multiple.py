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

if __name__ in '__main__':

    fig,ax = plt.subplots(1,1)
    
    incs = np.linspace(0.1,0.99,11) * np.pi
    # incs = [incs[1]]
    e = 0.4
    for i in incs:
        sys = run(i,e)
        state = sys.state
        color = utils.STATE_COLORS[state]
        utils.phase_diag(sys,ax,c=color)
    
    ax.set_aspect('equal')
    
    outfile = Path(__file__).parent / 'output' / 'multiple.png'
    
    fig.savefig(outfile,facecolor='w')