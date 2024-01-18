"""
Module to find the area inside of the parameterized equation:

.. math::
    x = i \\cos{\\Omega} \\\\
    y = i \\sin{\\Omega}

For one procession of the orbit.
"""
import numpy as np

from misaligned_cb_disk import params, system


def get_r_and_theta(
    mass_binary: float,
    mass_fraction: float,
    semimajor_axis_binary: float,
    eccentricity_binary: float,
    mass_planet: float,
    semimajor_axis_planet: float,
    inclination_planet: float,
    lon_ascending_node_planet: float,
    true_anomaly_planet: float,
    eccentricity_planet: float,
    arg_pariapsis_planet: float,
    orbit_step: int = 5,
    max_orbits: int = 1000,
    capture_freq: int = 1
):
    """
    Calculate the polar coordinates r and theta for a planetary system.

    Parameters
    ----------
    mass_binary : float
        The mass of the binary system.
    mass_fraction : float
        The mass fraction of the binary system.
    semimajor_axis_binary : float
        The semimajor axis of the binary system.
    eccentricity_binary : float
        The eccentricity of the binary system.
    mass_planet : float
        The mass of the planet.
    semimajor_axis_planet : float
        The semimajor axis of the planet.
    inclination_planet : float
        The inclination of the planet.
    lon_ascending_node_planet : float
        The longitude of the ascending node of the planet.
    true_anomaly_planet : float
        The true anomaly of the planet.
    eccentricity_planet : float
        The eccentricity of the planet.
    arg_pariapsis_planet : float
        The argument of periapsis of the planet.
    orbit_step : int, optional
        The step size for integrating the system. Defaults to 5.
    max_orbits : int, optional
        The maximum number of orbits to integrate. Defaults to 1000.
    capture_freq : int, optional
        The capture frequency for saving data. Defaults to 1.

    Returns
    -------
    r : float
        The inclination of the planetary system.
    theta : float
        The longitude of the ascending node of the planetary system.
    """
    binary = params.Binary(
        mass_binary=mass_binary,
        mass_fraction=mass_fraction,
        semimajor_axis_binary=semimajor_axis_binary,
        eccentricity_binary=eccentricity_binary
    )
    planet = params.Planet(
        mass=mass_planet,
        semimajor_axis=semimajor_axis_planet,
        inclination=inclination_planet,
        lon_ascending_node=lon_ascending_node_planet,
        true_anomaly=true_anomaly_planet,
        eccentricity=eccentricity_planet,
        arg_pariapsis=arg_pariapsis_planet
    )
    sys = system.System(binary, planet, sim=None)
    sys.integrate_to_get_path(
        step=orbit_step,
        max_orbits=max_orbits,
        capture_freq=capture_freq
    )
    # math
    r = sys.inclination
    theta = sys.lon_ascending_node
    return r, theta


def cast_r_and_theta(r: np.ndarray, theta: np.ndarray):
    """
    Reorder the polar coordinates r and theta so that
    the discontinuies at pi and -pi are at the bounds
    of integration.

    Parameters
    ----------
    r : np.ndarray
        The polar coordinate r.
    theta : np.ndarray
        The polar coordinate theta.

    Returns
    -------
    r : np.ndarray
        The polar coordinate r, reordered.
    theta : np.ndarray
        The polar coordinate theta, reordered.
    """
    # theta has a discontinuity at pi and -pi
    # cast it to be continuous.
    # order is not important, so we can rearrange
    # the array to make the discontinuity at the ends.
    i_discontinuity = np.argmax(np.abs(np.diff(theta)))
    before = slice(0, i_discontinuity+1)
    after = slice(i_discontinuity+1, None)
    theta = np.concatenate((theta[after], theta[before]))
    r = np.concatenate((r[after], r[before]))
    return r, theta


def integrate_r_and_theta(
    r: np.ndarray,
    theta: np.ndarray
):
    """
    Use the trapezoidal rule to integrate r and theta.

    Parameters
    ----------
    r : np.ndarray
        The polar coordinate r.
    theta : np.ndarray
        The polar coordinate theta.

    Returns
    -------
    float
        The integral of cos(r)-1 as a function of theta.
    """
    return np.trapz(np.cos(r)-1, theta)


def get_area(
    mass_binary: float,
    mass_fraction: float,
    semimajor_axis_binary: float,
    eccentricity_binary: float,
    mass_planet: float,
    semimajor_axis_planet: float,
    inclination_planet: float,
    lon_ascending_node_planet: float,
    true_anomaly_planet: float,
    eccentricity_planet: float,
    arg_pariapsis_planet: float,
    orbit_step: int = 5,
    max_orbits: int = 1000,
    capture_freq: int = 1
) -> float:
    """
    Get the area enclosed by the orbit in the parameterized space.

    Parameters
    ----------
    mass_binary : float
        The mass of the binary.
    mass_fraction : float
        The mass fraction :math:`M_2/M_b`.
    semimajor_axis_binary : float
        The semimajor axis of the binary.
    eccentricity_binary : float
        The eccentricity of the binary.
    mass_planet : float
        The mass of the planet.
    semimajor_axis_planet : float
        The semimajor axis of the planet.
    inclination_planet : float
        The inclination of the planet.
    lon_ascending_node_planet : float
        The longitude of the ascending node of the planet.
    true_anomaly_planet : float
        The true anomaly of the planet.
    eccentricity_planet : float
        The eccentricity of the planet.
    arg_pariapsis_planet : float
        The argument of the periapsis of the planet.
    orbit_step : int
        The step size of the orbit integration.
    max_orbits : int
        The maximum number of orbits to integrate.
    capture_freq : int
        The frequency at which to capture the state in units of 1/orbit.

    Returns
    -------
    float
        The area of the orbit.
    """
    r, theta = get_r_and_theta(
        mass_binary=mass_binary,
        mass_fraction=mass_fraction,
        semimajor_axis_binary=semimajor_axis_binary,
        eccentricity_binary=eccentricity_binary,
        mass_planet=mass_planet,
        semimajor_axis_planet=semimajor_axis_planet,
        inclination_planet=inclination_planet,
        lon_ascending_node_planet=lon_ascending_node_planet,
        true_anomaly_planet=true_anomaly_planet,
        eccentricity_planet=eccentricity_planet,
        arg_pariapsis_planet=arg_pariapsis_planet,
        orbit_step=orbit_step,
        max_orbits=max_orbits,
        capture_freq=capture_freq
    )
    return integrate_r_and_theta(*cast_r_and_theta(r, theta))
