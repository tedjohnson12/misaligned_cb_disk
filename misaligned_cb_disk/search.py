
import rebound
import numpy as np

from misaligned_cb_disk.system import System
from misaligned_cb_disk import system
from misaligned_cb_disk import params

class Transition:
    _allowed_states = {system.UNKNOWN,system.PROGRADE,system.RETROGRADE,system.LIBRATING}
    _low_bound = 0.
    _high_bound = np.pi
    def __init__(
        self,
        low_value:float,
        low_state:str,
        high_value:float,
        high_state:str
    ):
        self.low_value = low_value
        self.low_state = low_state
        self.high_value = high_value
        self.high_state = high_state
        self._validate()
    def _validate(self):
        if self.low_value < self._low_bound:
            raise ValueError('Low value is too low')
        if self.low_value > self._high_bound:
            raise ValueError('Low value is too high')
        if self.high_value < self._low_bound:
            raise ValueError('High value is too low')
        if self.high_value > self._high_bound:
            raise ValueError('High value is too high')
        if self.low_value > self.high_value:
            raise ValueError(f'Low value ({self.low_value}) is higher than high value ({self.high_value}).')
        if not self.low_state in self._allowed_states:
            raise ValueError(f'State of {self.low_state} not allowed.')
        if not self.high_state in self._allowed_states:
            raise ValueError(f'State of {self.high_state} not allowed.')
    def __str__(self):
        return f'Transition from {self.low_state} to {self.high_state} between {self.low_value:.3f} and {self.high_value:.3f} radians.'
    @property
    def width(self)->float:
        """
        The width or precision of the transition bounds.
        
        :type:float
        """
        return self.high_value-self.low_value
    @property
    def kind(self)->str:
        """
        The kind of transition. E.g. 'pr' for prograte to retrograde.

        :type:str
        """
        return self.low_state+self.high_state
    def suggest(self):
        """
        Suggest the next simulation to run.

        Returns
        -------
        float
            The inclination of the next simulation.
        """
        if self.low_state == system.UNKNOWN:
            return self.low_value
        elif self.high_state == system.UNKNOWN:
            return self.high_value
        else:
            return 0.5*(self.low_value+self.high_value)
    def is_sufficient(self,required_precision:float):
        """
        If true, the precision of this transition region is less than
        or equal to some desired precision resolution.

        Parameters
        ----------
        required_precision : float
            The desired precision.

        Returns
        -------
        bool
            Whether or not this transition region is sufficient.
        """
        if system.UNKNOWN in (self.low_state,self.high_state):
            return False
        else:
            return self.width<=required_precision
    def _update_unknown(self,value,state):
        if value == self.low_value:
            return [Transition(
                low_value=value,low_state=state,
                high_value=self.high_value,high_state=self.high_state
            )]
        elif value == self.high_value:
            return [Transition(
                low_value=self.low_value,low_state=self.low_state,
                high_value=value,high_state=state
            )]
        else:
            raise ValueError(f'No value matching {value}.')
    def _update_middle(self,value,state):
        if state == self.low_state and state == self.high_state:
            raise RuntimeError(f'No information was gained from latest simulation (i={value:.3f} with state {state}).')
        if state == self.low_state:
            return [Transition(
                low_value=value,low_state=state,
                high_value=self.high_value,high_state=self.high_state
            )]
        elif state == self.high_state:
            return [Transition(
                low_value=self.low_value,low_state=self.low_state,
                high_value=value,high_state=state
            )]
        else:
            return [
                Transition(
                    low_value=self.low_value,low_state=self.low_state,
                    high_value=value,high_state=state
                ),
                Transition(
                    low_value=value,low_state=state,
                    high_value=self.high_value,high_state=self.high_state
                )
            ]
    def update_with(self,value:float,state:str):
        if value < self._low_bound:
            raise ValueError('Value is too low')
        if value > self._high_bound:
            raise ValueError('Value is too high')
        if not state in self._allowed_states:
            raise ValueError(f'State of {state} not allowed.')
        if system.UNKNOWN in (self.low_state,self.high_state):
            return self._update_unknown(value=value,state=state)
        else:
            return self._update_middle(value=value,state=state)
        
        
        

class Searcher:
    _low_end = 1e-3
    _high_end = np.pi-1e-3
    _integration_orbit_step = 5
    _integration_max_orbits = 1000
    _integration_capture_freq = 1
    class __flags__:
        done=False
    def __init__(
        self,
        mass_binary:float,
        mass_fraction:float,
        semimajor_axis_binary: float,
        eccentricity_binary: float,
        mass_planet:float,
        semimajor_axis_planet:float,
        lon_ascending_node_planet:float,
        true_anomaly_planet:float,
        eccentricity_planet:float=0,
        arg_pariapsis_planet:float=0,
        precision:float = np.pi*0.1
    ):
        self._binary = params.Binary(
            mass_binary=mass_binary,
            mass_fraction=mass_fraction,
            semimajor_axis_binary=semimajor_axis_binary,
            eccentricity_binary=eccentricity_binary
        )
        self._mass_planet = mass_planet
        self._semimajor_axis_planet = semimajor_axis_planet
        self._lon_ascending_node_planet = lon_ascending_node_planet
        self._true_anomaly_planet = true_anomaly_planet
        self._eccentricity_planet = eccentricity_planet
        self._arg_pariapsis_planet = arg_pariapsis_planet
        self.transitions = [
            Transition(
                self._low_end,system.UNKNOWN,
                self._high_end,system.UNKNOWN
            )
        ]
        self.precision = precision
    def _planet(self,inclination:float):
        """
        Generate a Planet parameter set given an inclination.

        Parameters
        ----------
        inclination : float
            The initial inclination

        Returns
        -------
        params.Planet
            The planet parameters.
        """
        return params.Planet(
            mass=self._mass_planet,
            semimajor_axis=self._semimajor_axis_planet,
            inclination=inclination,
            lon_ascending_node=self._lon_ascending_node_planet,
            true_anomaly=self._true_anomaly_planet,
            eccentricity=self._eccentricity_planet,
            arg_pariapsis=self._arg_pariapsis_planet
        )
    def _system(self,inclination:float,sim:rebound.Simulation=None):
        return System(
            binary=self._binary,
            planet=self._planet(inclination=inclination),
            sim=sim
        )
    def get_simulation_state(self,inclination:float):
        sys = self._system(inclination=inclination)
        sys.integrate_to_get_state(
            step=self._integration_orbit_step,
            max_orbits=self._integration_max_orbits,
            capture_freq=self._integration_capture_freq
        )
        return sys.state
    def get_current_transition(self):
        current = None
        others = []
        for tr in self.transitions:
            if current is not None:
                others.append(tr)
            elif not tr.is_sufficient(self.precision):
                current = tr
            else:
                others.append(tr)
        return current,others
            
    def update_next(self):
        current, others = self.get_current_transition()
        if current is None:
            self.__flags__.done = True
            return None
        next_inclination = current.suggest()
        state = self.get_simulation_state(next_inclination)
        updated_transitions = current.update_with(next_inclination,state)
        self.transitions = updated_transitions + others
    def search(self):
        while not self.__flags__.done:
            self.update_next()
        return self.transitions
        
    
        