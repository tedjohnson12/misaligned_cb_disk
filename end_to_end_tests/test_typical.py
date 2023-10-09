
import rebound
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from misaligned_cb_disk import params, system, utils

if __name__ in '__main__':

    i = 170*np.pi/180
    n_orbits = 30

    binary = params.Binary(2,0.5,0.5,0.4)
    planet = params.Planet(0,2,i,np.pi/2,0,0,0)

    sim = rebound.Simulation()

    sys = system.System(binary,planet,sim)

    sys.integrate_orbits(n_orbits,capture_freq=1)
    

    fig,axes = plt.subplots(3,1)
    
    outfile = Path(__file__).parent / 'output' / 'typical.png'
    
    axes[0].set_aspect('equal')
    utils.phase_diag(sys,axes[0])
    axes[0].set_xlabel('$i\\cos{\\Omega}$')
    axes[0].set_ylabel('$i\\sin{\\Omega}$')
        
    axes[1].plot(sys.t,(sys.lon_ascending_node))
    axes[1].set_xlabel('$t$')
    axes[1].set_ylabel('$\\Omega$')
    
    utils.plot_omega_diff(sys,axes[2])
    
    # h = sys.specific_angular_momentum
    # tau = sys.specific_torque
    
    # h_x = system.dot(h,sys.x_hat)
    # h_y = system.dot(h,sys.y_hat)
    # h_x_dot = system.dot(tau,sys.x_hat)
    # h_y_dot = system.dot(tau,sys.y_hat)
    # den = h_x**2 + h_y**2
    # num = h_y * h_x_dot - h_x * h_y_dot
    # a = sys.rp_2dot
    # r = sys.rp
    # costh = system.dot(r,a)/np.sqrt(system.dot(r,r))/np.sqrt(system.dot(a,a))
    # theta = np.arccos(costh)
    # y = theta/np.pi*180
    # axes[2].plot(sys.t[1:],y[1:])
    # axes[2].set_xlabel('$t$')
    # axes[2].set_ylabel('$\\dot{\\Omega}$')
    
    
    
    fig.savefig(outfile,facecolor='w')
    
    



