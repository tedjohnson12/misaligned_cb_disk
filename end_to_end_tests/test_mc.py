
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from misaligned_cb_disk import mc
from misaligned_cb_disk import utils
from misaligned_cb_disk.analytic import Zanazzi2018

MASS_BINARY = 1
FRAC_BINARY = 0.5
SEP_BINARY = 0.2
ECC_BINARY = 0.2

MASS_PLANET = 1e-2
SEP_PLANET = 1
NU = 0
ECC_PLANET = 0
ARG_PARIAPSIS = 0

precision = 0.07


if __name__ in '__main__':
    sampler = mc.Sampler(
        mass_binary=MASS_BINARY,
        mass_fraction=FRAC_BINARY,
        semimajor_axis_binary=SEP_BINARY,
        eccentricity_binary=ECC_BINARY,
        mass_planet=MASS_PLANET,
        semimajor_axis_planet=SEP_PLANET,
        true_anomaly_planet=NU,
        eccentricity_planet=ECC_PLANET,
        arg_pariapsis_planet=ARG_PARIAPSIS,
        precision=precision
    )
    sampler.sim_until_precision(precision,batch_size=100,max_samples=1000)
    res = sampler.bootstrap('l',confidence_level=0.95)
    print(f'There is a 95% chance it falls between {res.confidence_interval[0]:.3f} and {res.confidence_interval[1]:.3f}')
    
    inclinations = np.array(sampler.inclinations)
    lon_ascending_nodes = np.array(sampler.lon_ascending_nodes)
    states = sampler.states
    
    # analytic_probs = [Zanazzi2018.prob_polar(inclination, ECC_BINARY) for inclination in inclinations]
    # frac_polar = np.mean(analytic_probs)
    frac_polar = Zanazzi2018.frac_polar(ECC_BINARY,n_points=100)
    print(f'According to {Zanazzi2018.citet}, the fraction of polar orbits is {frac_polar:.3f}')
    
    colors = [utils.STATE_COLORS[state] for state in states]
    fig,ax = plt.subplots(1,1)
    isin = inclinations * np.sin(lon_ascending_nodes)
    icos = inclinations * np.cos(lon_ascending_nodes)
    ax.scatter(icos,isin,marker='.',c=colors)
    ax.set_aspect('equal')
    ax.set_xlabel('$i \\cos{\\Omega}$')
    ax.set_ylabel('$i \\sin{\\Omega}$')
    fig.legend(
        handles=[
            Line2D([0],[0],marker='.',linestyle='None',markerfacecolor=utils.STATE_COLORS[state],color=utils.STATE_COLORS[state]) for state in utils.STATE_COLORS
        ],
        labels=[utils.STATE_LONG_NAMES[state] for state in utils.STATE_COLORS],
    )
    fig.savefig(Path(__file__).parent / 'output' / 'mc.png',facecolor='w')
    0
    