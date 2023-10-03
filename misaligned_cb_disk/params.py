"""
Parameters
----------

Control the input parameters of the rebound simulations.
"""
from rebound import Simulation, Particle

G = 1


def get_star_masses(mass_binary: float, mass_fraction: float):
    """
    Get the masses of two binary stars given
    the total mass of the binary :math:``M_b and the 
    mass fraction parameter :math:`f_b`

    Parameters
    ----------
    mass_binary : float
        The total mass of the binary
    mass_fraction : float
        The fraction :math:`M_2/M_b`

    Returns
    -------
    mass1 : float
        The mass of the primary star.
    mass2 : float
        The mass of the secondary star.
    """
    mass2 = mass_binary * mass_fraction
    mass1 = mass_binary*(1-mass_fraction)
    return mass1, mass2


class Binary:
    """
    Binary system parameters.

    Parameters
    ----------
    mass_binary : float
        The total mass of the binary.
    mass_fraction : float
        The mass fraction :math:`M_2/M_b`.
    semimajor_axis_binary : float
        The semimajor axis of the binary.
    eccentricity_binary : float
        The eccentricity of the binary.

    Attributes
    ----------
    mass_binary : float
        The total mass of the binary.
    mass_fraction : float
        The mass fraction :math:`M_2/M_b`.
    semimajor_axis_binary : float
        The semimajor axis of the binary.
    eccentricity_binary : float
        The eccentricity of the binary.
    name1 : str
        The identifier for the primary star.
    name2 : str
        The indentifier for the secondary star.
    mass1 : float
        The mass of the primary star.
    mass2 : float
        The mass of the secondary star.
    """
    name1 = 'm1'
    name2 = 'm2'

    def __init__(
        self,
        mass_binary: float,
        mass_fraction: float,
        semimajor_axis_binary: float,
        eccentricity_binary: float
    ):
        self.mass_binary = mass_binary
        self.mass_fraction = mass_fraction
        self.semimajor_axis_binary = semimajor_axis_binary
        self.eccentricity_binary = eccentricity_binary

    @property
    def mass1(self) -> float:
        """
        The mass of Star 1

        :type: float
        """
        mass1, _ = get_star_masses(self.mass_binary, self.mass_fraction)
        return mass1

    @property
    def mass2(self) -> float:
        """
        The mass of Star 2

        :type: float
        """
        _, mass2 = get_star_masses(self.mass_binary, self.mass_fraction)
        return mass2

    def add_to_sim(self, sim: Simulation):
        """
        Add binary system to a rebound Simulation.
        Then move to the CoM frame.

        Parameters
        ----------
        sim : rebound.Simulation
            The simulation to add the particles to.
        """
        star1 = Particle(
            simulation=sim,
            hash=self.name1,
            m=self.mass1
        )
        sim.add(star1)
        star2 = Particle(
            simulation=sim,
            a=self.semimajor_axis_binary,
            e=self.eccentricity_binary,
            hash=self.name2,
            m=self.mass2,
            primary=star1
        )
        sim.add(star2)
        sim.move_to_com()
