"""
Tests for misaligned_cb_disk.search
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from misaligned_cb_disk import utils, search


MASS_BINARY = 1
FRAC_BINARY = 0.5
SEP_BINARY = 0.2
ECC_BINARY = 0.2

MASS_PLANET = 0e-3
SEP_PLANET = 1
LON_ASC_NODE = np.pi/2
NU = 0
ECC_PLANET = 0
ARG_PARIAPSIS = 0

precision = np.pi/180 * 0.5

if __name__ in '__main__':
    searcher = search.Searcher(
        mass_binary=MASS_BINARY,
        mass_fraction=FRAC_BINARY,
        semimajor_axis_binary=SEP_BINARY,
        eccentricity_binary=ECC_BINARY,
        mass_planet=MASS_PLANET,
        semimajor_axis_planet=SEP_PLANET,
        lon_ascending_node_planet=LON_ASC_NODE,
        true_anomaly_planet=NU,
        eccentricity_planet=ECC_PLANET,
        arg_pariapsis_planet=ARG_PARIAPSIS,
        precision=precision
    )
    result = searcher.search()
    for tr in result:
        print(tr)
        
    fig,ax = plt.subplots(1,1)
    areas = []
    for tr in result:
        for val in (tr.low_value,tr.high_value):
            sys = searcher.get_system(val)
            sys.integrate_to_get_path(max_orbits=10000)
            area = sys.normalized_area
            areas.append(area)
            label = f'Area: {area:.2f}'
            ax.plot(sys.icosomega,sys.isinomega,c=utils.STATE_COLORS[sys.state],label=label)

    ax.set_aspect('equal')
    ax.legend()
    area_minus_one = np.array(areas).sum()-1
    ax.set_title(f'$\\sum$ Area-1 = {area_minus_one:.2e} or {100*area_minus_one:.3f} %')
    outfile = Path(__file__).parent / 'output' / 'search.png'

    fig.savefig(outfile,facecolor='w',dpi=200)
    