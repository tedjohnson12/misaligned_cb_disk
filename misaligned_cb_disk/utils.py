"""
Utilities for misaligned disks
"""

import numpy as np
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
    if ax is None:
        ax = plt.gca()
    ax.plot(sys.icosomega,sys.isinomega,**kwargs)

def phase_diag_cos(sys:System,ax:Axes=None,**kwargs):
    if ax is None:
        ax = plt.gca()
    ax.plot(sys.icosomega,sys.icosomega_dot,**kwargs)
def phase_diag_sin(sys:System,ax:Axes=None,**kwargs):
    if ax is None:
        ax = plt.gca()
    ax.plot(sys.isinomega,sys.isinomega_dot,**kwargs)

def lon_ascending_node_phase_diag(sys:System,ax:Axes=None,**kwargs):
    if ax is None:
        ax = plt.gca()
    ax.plot(sys.lon_ascending_node,sys.lon_ascending_node_dot,**kwargs)