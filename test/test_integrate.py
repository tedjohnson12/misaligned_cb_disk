
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from misaligned_cb_disk import integrate


def test_integrate():
    r,theta = integrate.get_r_and_theta(
        mass_binary=1,
        mass_fraction=0.5,
        semimajor_axis_binary=0.2,
        eccentricity_binary=0.4,
        mass_planet=1e-3,
        semimajor_axis_planet=1,
        inclination_planet=0.2595*np.pi,
        # inclination_planet=0.99*np.pi,
        lon_ascending_node_planet=np.pi/2,
        arg_pariapsis_planet=0,
        true_anomaly_planet=0,
        eccentricity_planet=0,
        orbit_step=5,
        max_orbits=4000,
        capture_freq=1
    )
    
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(theta,r)
    ax[0].set_aspect('equal')
    ax[0].set_ylabel('i (rad)')
    ax[0].set_xlabel('$\\Omega$ (rad)')
    ax[0].set_ylim(0,2*np.pi)
    ax[0].set_xlim(-np.pi,np.pi)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    ax[1].plot(x,y)
    ax[1].set_aspect('equal')
    ax[1].set_xlabel('$i\\cos{\\Omega}$')
    ax[1].set_ylabel('$i\\sin{\\Omega}$')
    ax[1].set_xlim(-np.pi,np.pi)
    ax[1].set_ylim(-np.pi,np.pi)
    
    outfile = Path(__file__).parent / 'output' / 'r_and_theta.png'
    
    fig.savefig(outfile, facecolor='w')