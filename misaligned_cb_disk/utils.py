"""
Utilities for misaligned disks
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from misaligned_cb_disk.system import System
import misaligned_cb_disk.system as system

STATE_COLORS = {
    system.PROGRADE:'xkcd:crimson',
    system.RETROGRADE:'xkcd:azure',
    system.LIBRATING:'xkcd:violet',
    system.UNKNOWN:'xkcd:mint green'
}

def phase_diag(sys:System,ax:Axes=None,**kwargs):
    """
    Plot the phase diagram in inclination - ascending node space.
    
    Parameters
    ----------
    sys : System
        The system to plot.
    ax : Axes
        The axes to plot on.
    **kwargs
        Additional keyword arguments to pass to the plot.
    """
    if ax is None:
        ax = plt.gca()
    ax.plot(sys.icosomega,sys.isinomega,**kwargs)

def phase_diag_cos(sys:System,ax:Axes=None,**kwargs):
    """
    A true phase diagram for :math:`i\\cos{\\Omega}`, plotting it against its time derivative.
    
    Parameters
    ----------
    sys : System
        The system to plot.
    ax : Axes
        The axes to plot on.
    **kwargs
        Additional keyword arguments to pass to the plot.
    """
    if ax is None:
        ax = plt.gca()
    ax.plot(sys.icosomega,sys.icosomega_dot,**kwargs)
def phase_diag_sin(sys:System,ax:Axes=None,**kwargs):
    """
    A true phase diagram for :math:`i\\sin{\\Omega}`, plotting it against its time derivative.
    
    Parameters
    ----------
    sys : System
        The system to plot.
    ax : Axes
        The axes to plot on.
    **kwargs
        Additional keyword arguments to pass to the plot.
    """
    if ax is None:
        ax = plt.gca()
    ax.plot(sys.isinomega,sys.isinomega_dot,**kwargs)

def lon_ascending_node_phase_diag(sys:System,ax:Axes=None,**kwargs):
    """
    A true phase diagram for :math:`\\Omega`, plotting it against its time derivative.
    
    Parameters
    ----------
    sys : System
        The system to plot.
    ax : Axes
        The axes to plot on.
    **kwargs
        Additional keyword arguments to pass to the plot.
    """
    if ax is None:
        ax = plt.gca()
    ax.plot(sys.lon_ascending_node,sys.lon_ascending_node_dot,**kwargs)
