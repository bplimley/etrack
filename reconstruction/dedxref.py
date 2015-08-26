# dedxref.py:
# dE/dx table for electrons in silicon.
# for calculation of beta angle in HybridTrack.

import numpy as np


table_energies = np.arange(50,1401,25)
table_dedx = np.array([
    1.2114,
    0.9185,
    0.7200,
    0.5800,
    0.5300,
    0.4400,
    0.3700,
    0.3600,
    0.3200,
    0.3100,
    0.2900,
    0.2900,
    0.2700,
    0.2700,
    0.2600,
    0.2500,
    0.2500,
    0.2400,
    0.2400,
    0.2400,
    0.2300,
    0.2300,
    0.2300,
    0.2300,
    0.2200,
    0.2200,
    0.2200,
    0.2200,
    0.2200,
    0.2200,
    0.2200,
    0.2200,
    0.2100,
    0.2100,
    0.2100,
    0.2100,
    0.2100,
    0.2100,
    0.2100,
    0.2100,
    0.2100,
    0.2100,
    0.2100,
    0.2100,
    0.2100,
    0.2100,
    0.2100,
    0.2100,
    0.2100,
    0.2100,
    0.2100,
    0.2100,
    0.2100,
    0.2100,
    0.2100])


def dedx(energy_kev):
    """
    Table lookup of dE/dx values.

    Input: energy_kev: electron energy(ies) in keV.
    Output: dedx_kevum: expected dE/dx value(s) in keV/um.
    """

    energy_kev = np.array(energy_kev)
    dedx_kevum = np.interp(energy_kev, table_energies, table_dedx,
                            left=table_dedx[0], right=table_dedx[-1])
    return dedx_kevum
