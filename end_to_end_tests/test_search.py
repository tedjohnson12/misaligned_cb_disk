import rebound
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from misaligned_cb_disk import params, system, utils, search


mb = 1
fb = 0.5
ab = 0.2
eb = 0.2

mp = 1e-3
ap = 1
Omega = np.pi/2
nu = 0
ep = 0
omega = 0

precision = np.pi/180 * 3

if __name__ in '__main__':
    searcher = search.Searcher(
        mass_binary=mb,
        mass_fraction=fb,
        semimajor_axis_binary=ab,
        eccentricity_binary=eb,
        mass_planet=mp,
        semimajor_axis_planet=ap,
        lon_ascending_node_planet=Omega,
        true_anomaly_planet=nu,
        eccentricity_planet=ep,
        arg_pariapsis_planet=omega,
        precision=precision
    )
    result = searcher.search()
    for tr in result:
        print(tr)
        
    fig,ax = plt.subplots(1,1)
    for tr in result:
        for val in (tr.low_value,tr.high_value):
            sys = searcher._system(val)
            sys.integrate_to_get_path()
            ax.plot(sys.icosomega,sys.isinomega,c=utils.STATE_COLORS[sys.state])
    
    ax.set_aspect('equal')
    outfile = Path(__file__).parent / 'output' / 'search.png'
    
    fig.savefig(outfile,facecolor='w',dpi=200)
    