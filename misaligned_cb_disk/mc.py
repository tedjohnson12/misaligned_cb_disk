"""
Monte Carlo Simulation module
"""
from typing import List
import numpy as np
import rebound

from misaligned_cb_disk import params
from misaligned_cb_disk.system import System

class Sampler:
    """
    Monte Carlo sampler
    """
    _integration_orbit_step = 5
    _integration_max_orbits = 1000
    _integration_capture_freq = 1
    def __init__(
        self,
        mass_binary: float,
        mass_fraction: float,
        semimajor_axis_binary: float,
        eccentricity_binary: float,
        mass_planet: float,
        semimajor_axis_planet: float,
        true_anomaly_planet: float,
        eccentricity_planet: float = 0,
        arg_pariapsis_planet: float = 0,
        precision: float = np.pi*0.1,
        rng: np.random.Generator = np.random.default_rng()
    ):
        self._binary = params.Binary(
            mass_binary=mass_binary,
            mass_fraction=mass_fraction,
            semimajor_axis_binary=semimajor_axis_binary,
            eccentricity_binary=eccentricity_binary
        )
        self._mass_planet = mass_planet
        self._semimajor_axis_planet = semimajor_axis_planet
        self._true_anomaly_planet = true_anomaly_planet
        self._eccentricity_planet = eccentricity_planet
        self._arg_pariapsis_planet = arg_pariapsis_planet
        self.precision = precision
        self.inclinations: List[float] = []
        self.lon_ascending_nodes: List[float] = []
        self.states: List[str] = []
        self.rng = rng
    
    def _planet(
        self,
        inclination: float,
        lon_ascending_node: float
        ) -> params.Planet:
        """
        Generate a Planet parameter set given an inclination.

        Parameters
        ----------
        inclination : float
            The initial inclination
        lon_ascending_node : float
            The initial longitude of the ascending node

        Returns
        -------
        params.Planet
            The planet parameters.
        """
        return params.Planet(
            mass=self._mass_planet,
            semimajor_axis=self._semimajor_axis_planet,
            inclination=inclination,
            lon_ascending_node=lon_ascending_node,
            true_anomaly=self._true_anomaly_planet,
            eccentricity=self._eccentricity_planet,
            arg_pariapsis=self._arg_pariapsis_planet
        )
    def get_system(
        self,
        inclination: float,
        lon_ascending_node: float,
        sim: rebound.Simulation = None
        ) -> System:
        """
        Generate a system given an inclination.

        Parameters
        ----------
        inclination : float
            The initial inclination
        lon_ascending_node : float
            The initial longitude of the ascending node
        sim : rebound.Simulation, optional
            The simulation to add the particles to, by default None
        """
        return System(
            binary=self._binary,
            planet=self._planet(inclination=inclination, lon_ascending_node=lon_ascending_node),
            sim=sim
        )
    def get_simulation_state(
        self,
        inclination: float,
        lon_ascending_node: float
    ):
        """
        Get the behavior of the system given an inclination.

        This function will create a system and integrate it just long
        enough to see it's long-term behavior.

        Parameters
        ----------
        inclination : float
            The initial inclination
        lon_ascending_node : float
            The initial longitude of the ascending node

        Returns
        -------
        str
            The final state of the system.
        """
        sys = self.get_system(inclination=inclination, lon_ascending_node=lon_ascending_node)
        sys.integrate_to_get_state(
            step=self._integration_orbit_step,
            max_orbits=self._integration_max_orbits,
            capture_freq=self._integration_capture_freq
        )
        return sys.state
    
    def get_next(self):
        next_inclination = self.rng.uniform(0, 2*np.pi)
        next_lon_ascending_node = self.rng.uniform(0, 2*np.pi)
        state = self.get_simulation_state(
            inclination=next_inclination,
            lon_ascending_node=next_lon_ascending_node
        )
        return next_inclination, next_lon_ascending_node, state
    
    @property
    def n_sampled(self):
        return len(self.states)
    @property
    def frac_uncertainty(self):
        return 1/np.sqrt(self.n_sampled)
    def get_frac(self,state):
        return self.states.count(state)/self.n_sampled
    
    
    def sim_n_samples(self,N:int):
        for i in range(N):
            next_inclination, next_lon_ascending_node, state = self.get_next()
            self.inclinations.append(next_inclination)
            self.lon_ascending_nodes.append(next_lon_ascending_node)
            self.states.append(state)
        for state in ['p','r','l']:
            print(state, self.get_frac(state), self.frac_uncertainty)
        
    