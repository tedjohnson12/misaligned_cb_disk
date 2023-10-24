"""
Module to find the area inside of the parameterized equation:

.. math::
    x = i \\cos{\\Omega} \\\\
    y = i \\sin{\\Omega}

For one procession of the orbit.
"""

from misaligned_cb_disk import params, system


def get_r_and_theta(
    mass_binary:float,
    mass_fraction:float,
    semimajor_axis_binary: float,
    eccentricity_binary: float,
    mass_planet:float,
    semimajor_axis_planet:float,
    inclination_planet:float,
    lon_ascending_node_planet:float,
    true_anomaly_planet:float,
    eccentricity_planet:float,
    arg_pariapsis_planet:float,
    orbit_step:int=5,
    max_orbits:int=1000,
    capture_freq:int=1
    
    
):
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
    sys = system.System(binary,planet,sim=None)
    sys.integrate_to_get_path(
        step=orbit_step,
        max_orbits=max_orbits,
        capture_freq=capture_freq
    )
    # math
    r = sys.inclination
    theta = sys.lon_ascending_node
    return r, theta