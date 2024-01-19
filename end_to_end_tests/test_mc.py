
import numpy as np
from misaligned_cb_disk import mc

MASS_BINARY = 1
FRAC_BINARY = 0.5
SEP_BINARY = 0.2
ECC_BINARY = 0.2

MASS_PLANET = 1e-3
SEP_PLANET = 1
NU = 0
ECC_PLANET = 0
ARG_PARIAPSIS = 0

precision = np.pi/180 * 0.5

if __name__ in '__main__':
    sampler = mc.Sampler(
        mass_binary=MASS_BINARY,
        mass_fraction=FRAC_BINARY,
        semimajor_axis_binary=SEP_BINARY,
        eccentricity_binary=ECC_BINARY,
        mass_planet=MASS_PLANET,
        semimajor_axis_planet=SEP_PLANET,
        true_anomaly_planet=NU,
        eccentricity_planet=ECC_PLANET,
        arg_pariapsis_planet=ARG_PARIAPSIS,
        precision=precision
    )
    sampler.sim_n_samples(100)