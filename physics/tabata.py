"""
Compute electron ranges using analytical formulae in Tabata's papers.

Tabata 1996a:
Tatsuo Tabata, Pedro Andreo, Kunihiko Shinoda.
"An analytic formula for the extrapolated range of electrons in condensed
materials."
Nucl. Instr. Meth. B 119 (1996) 463-470.

Tabata 2002:
Tatsuo Tabata, Vadim Moskvin, Pedro Andreo, Valentin Lazurik, Yuri Rogov.
"Extrapolated ranges of electrons determined from transmission and projected-
range straggling curves."
Rad. Phys. and Chem. 64 (2002) 161-167.
"""

from __future__ import print_function

import numpy as np
import pint
from pint.unit import ScaleConverter, UnitDefinition

units = pint.UnitRegistry()
units.define(
    UnitDefinition('%', 'percent', (), ScaleConverter(1 / 100.0)))
units.define('keV = kiloelectron_volt')
units.define('MeV = megaelectron_volt')

ELECTRON_REST_ENERGY = 0.510999 * units.MeV

MIN_ENERGY_1996A = 1 * units.keV        # Tabata 1996a p. 2
SEMI_MIN_ENERGY_1996A = 10 * units.keV  # Tabata 1996a p. 2 (shell effects)
MAX_ENERGY_1996A = 100e3 * units.keV    # Tabata 1996a p. 2
MIN_ENERGY_2002 = 10 * units.keV        # Tabata 2002 sec 2.3
MAX_ENERGY_2002 = 50e3 * units.keV      # Tabata 2002 sec 1; 2.1

# paper notation is 1-indexed. so here, elements at index 0 are placeholders
TABATA_1996A_TABLE_1 = (
    0.0,        # (0)
    0.3879,     # 1
    0.2178,     # 2
    0.4541,     # 3
    0.03068,    # 4
    3.326e-16,  # 5
    13.24,      # 6
    1.316,      # 7
    14.03,      # 8
    0.7406,     # 9
    4.294e-3,   # 10
    1.684,      # 11
    0.2264,     # 12
    0.6127,     # 13
    0.1207,     # 14
)
TABATA_1996A_TABLE_2 = (
    0.0,        # (0)
    3.600,      # 1
    0.9882,     # 2
    1.191e-3,   # 3
    0.8622,     # 4
    1.02501,    # 5
    1.0803e-4,  # 6
    0.99628,    # 7
    1.303e-4,   # 8
    1.02441,    # 9
    1.2986e-4,  # 10
    1.030,      # 11
    1.110e-2,   # 12
    1.10e-6,    # 13
    0.959,      # 14
)
TABATA_2002_TABLE_2 = (
    0.0,        # (0)
    0.2946,     # 1
    0.2740,     # 2
    18.4,       # 3
    3.457,      # 4
    1.377,      # 5
    6.59,       # 6
    2.414,      # 7
    1.094,      # 8
    0.05242,    # 9
    0.3709,     # 10
    0.958,      # 11
    2.02,       # 12
    1.099,      # 13
    0.2808,     # 14
    0.2042,     # 15
)


class TabataError(Exception):
    """General error for Tabata module."""
    pass


class EnergyOutOfRange(TabataError):
    """Energy value out of bounds for calculation."""
    pass


def extrapolated_range_mass(energy, Z, A, I_eV, ref=None):
    """Compute electron extrapolated range with Tabata's analytical formulae.

    Args:
      energy: initial electron energy [keV, or use pint]
      Z: atomic number of material
      A: atomic weight of material [AMU]
      I_eV: mean excitation energy of material [eV, or use pint]
      ref: may be '1996a', '2002', 'CSDA', or None.
        '1996a': calculate extrapolated range using 1996a detour factor.
        '2002': calculate extrapolated range using 2002 detour factor.
        'CSDA': calculate CSDA range instead of extrapolated range.
        None: calculated extrapolated range using better paper (usually 2002).
        [Default: None]

    Returns:
      pint quantity of the computed extrapolated range, in mass thickness.

    Raises:
      EnergyOutOfRange: if energy is out of bounds of accuracy.
    """

    try:
        energy = energy.to(units.MeV)
    except AttributeError:
        energy = energy * units.keV

    try:
        I = I_eV.to(units.eV)
    except AttributeError:
        I = I_eV * units.eV

    if ref is None:
        ref = '2002'
        if ((energy > MAX_ENERGY_2002 and energy <= MAX_ENERGY_1996A) or
                (energy < MIN_ENERGY_2002 and energy >= MIN_ENERGY_1996A)):
            ref = '1996a'

    if ref == '1996' or ref.lower() == '1996a':
        _check_energy_1996a(energy)
        f_d = _detour_factor_1996a(energy, Z)
    elif ref == '2002':
        _check_energy_2002(energy)
        f_d = _detour_factor_2002(energy, Z)
    elif ref.upper() == 'CSDA':
        _check_energy_1996a(energy)
        f_d = 1.
    else:
        raise ValueError("ref should be '1996a', '2002', 'CSDA', or None")

    range_mass = f_d * CSDA_range_mass(energy, Z, A, I)

    return range_mass


def extrapolated_range_length(energy, Z, A, I_eV, density, ref=None):
    """Compute electron extrapolated range with Tabata's analytical formulae.

    Args:
      energy: initial electron energy [pint quantity or default keV]
      Z: atomic number of material
      A: atomic weight of material [AMU]
      I_eV: mean excitation energy of material [pint quantity or default eV]
      density: density of material [pint quantity or g/cm^3]
      ref: may be '1996a', '2002', 'CSDA', or None.
        '1996a': calculate extrapolated range using 1996a detour factor.
        '2002': calculate extrapolated range using 2002 detour factor.
        'CSDA': calculate CSDA range instead of extrapolated range.
        None: calculated extrapolated range using better paper (usually 2002).
        [Default: None]

    Returns:
      pint quantity of the computed extrapolated range

    Raises:
      EnergyOutOfRange: if energy is out of bounds of accuracy.
    """

    try:
        energy = energy.to(units.MeV)
    except AttributeError:
        energy = energy * units.keV

    try:
        density = density.to(units.g / units.cm**3)
    except AttributeError:
        density = density * (units.g / units.cm**3)

    range_mass = extrapolated_range_mass(energy, Z, A, I_eV, ref=ref)
    range_length = gcm2_to_mm(range_mass, density)
    return range_length


def CSDA_range_mass(energy, Z, A, I_eV):
    """Compute electron CSDA range, using analytical formula in Tabata 1996a.

    Args:
      energy: initial electron energy [pint quantity, or default MeV]
      Z: atomic number of material
      A: atomic weight of material [AMU]
      I_eV: mean excitation energy of material [eV]

    Returns:
      float of the CSDA range in g/cm^2.

    Raises:
      EnergyOutOfRange: if energy is out of bounds of accuracy.
    """

    try:
        energy = energy.to(units.MeV)
    except AttributeError:
        energy = energy * units.MeV

    try:
        I = I_eV.to(units.eV)
    except AttributeError:
        I = I_eV * units.eV

    t0 = (energy / ELECTRON_REST_ENERGY).to_base_units().magnitude

    d = TABATA_1996A_TABLE_2
    # Tabata 1996a equations 13--19
    c = (
        0.0,                    # (0)
        d[1] * A / Z**d[2],     # 1
        d[3] * Z**d[4],         # 2
        d[5] - d[6] * Z,        # 3
        d[7] - d[8] * Z,        # 4
        d[9] - d[10] * Z,       # 5
        d[11] / Z**d[12],       # 6
        d[13] * Z**d[14],       # 7
    )
    # Tabata 1996a equation 12
    I0 = (I / ELECTRON_REST_ENERGY).to_base_units().magnitude
    B = np.log((t0 / (I0 + c[7] * t0))**2) + np.log(1 + t0/2)
    # Tabata 1996a equation 11
    range_mass = c[1] / B * (
        np.log(1 + c[2] * t0**c[3]) / c[2] -
        c[4] * t0**c[5] / (1 + c[6] * t0)) * units.g / units.cm**2

    return range_mass


def CSDA_range_length(energy, Z, A, I_eV, density):
    """Compute electron CSDA range, using analytical formula in Tabata 1996a.

    Args:
      energy_MeV: initial electron energy [MeV]
      Z: atomic number of material
      A: atomic weight of material [AMU]
      I_eV: mean excitation energy of material [eV]
      density_gcm3: density of material [g/cm^3]

    Returns:
      float of the CSDA range in mm.

    Raises:
      EnergyOutOfRange: if energy is out of bounds of accuracy.
    """

    try:
        energy = energy.to(units.MeV)
    except AttributeError:
        energy = energy * units.MeV

    try:
        density = density.to(units.g / units.cm**3)
    except AttributeError:
        density = density * (units.g / units.cm**3)

    range_mass = CSDA_range_mass(energy, Z, A, I_eV)
    range_length = gcm2_to_mm(range_mass, density)
    return range_length


def gcm2_to_mm(mass_thickness, density):
    """Convert a mass thickness to a length.

    Args:
      mass_thickness: mass thickness [pint quantity]
      density: density of material [pint quantity]

    Returns:
      pint quantity of the length (units mm)
    """

    return (mass_thickness / density).to(units.mm)


def _check_energy_1996a(energy):
    """Check for energy value out of bounds for Tabata 1996a.

    Args:
      energy: initial electron energy [pint quantity]

    Raises:
      EnergyOutOfRange: if energy is out of bounds
    """

    if energy < MIN_ENERGY_1996A:
        raise EnergyOutOfRange(
            'Range is inaccurate below {}'.format(MIN_ENERGY_1996A))
    elif energy < SEMI_MIN_ENERGY_1996A:
        print(
            'Range is less accurate below {}, '.format(
                SEMI_MIN_ENERGY_1996A) +
            'because shell effects are ignored')
    elif energy > MAX_ENERGY_1996A:
        raise EnergyOutOfRange(
            'Range is inaccurate above {}'.format(MAX_ENERGY_1996A))


def _check_energy_2002(energy):
    """Check for energy value out of bounds for Tabata 2002.

    Args:
      energy: initial electron energy [pint quantity]

    Raises:
      EnergyOutOfRange: if energy is out of bounds
    """

    if energy < MIN_ENERGY_2002:
        raise EnergyOutOfRange(
            'Range is inaccurate below {}'.format(MIN_ENERGY_2002))
    elif energy > MAX_ENERGY_2002:
        raise EnergyOutOfRange(
            'Range is inaccurate above {}'.format(MAX_ENERGY_2002))


def _detour_factor_1996a(energy, Z):
    """Ratio of extrapolated range to CSDA range, according to Tabata 1996a.

    Args:
      energy: initial electron energy [pint quantity]
      Z: atomic number of material

    Returns:
      a float representing (extrapolated range) / (CSDA range)
    """

    t0 = (energy / ELECTRON_REST_ENERGY).to_base_units().magnitude
    b = TABATA_1996A_TABLE_1
    # Tabata 1996a equations 3--8
    a = (
        0.0,                                        # (0)
        b[1] * Z**b[2],                             # 1
        b[3] + b[4] * Z,                            # 2
        b[5] * Z**(b[6] - b[7] * np.log(Z)),        # 3
        b[8] / Z**(b[9]),                           # 4
        b[10] * Z**(b[11] - b[12] * np.log(Z)),     # 5
        b[13] * Z**b[14],                           # 6
    )
    # Tabata 1996a equation 2
    detour_factor = 1 / (
        a[1] + a[2] / (
            1 + a[3] / t0**a[4] + a[5] * t0**a[6]))

    return detour_factor


def _detour_factor_2002(energy, Z):
    """Ratio of extrapolated range to CSDA range, according to Tabata 2002.

    Args:
      energy: initial electron energy [pint quantity]
      Z: atomic number of material

    Returns:
      a float representing (extrapolated range) / (CSDA range)
    """

    t0 = (energy / ELECTRON_REST_ENERGY).to_base_units().magnitude
    b = TABATA_2002_TABLE_2
    # Tabata 2002 equations 9--14
    a = (
        0.0,                                        # (0)
        b[1] * Z**b[2],                             # 1
        b[3] * Z**(-b[4] + b[5] * np.log(Z)),       # 2
        b[6] * Z**(-b[7] + b[8] * np.log(Z)),       # 3
        b[9] * Z**(b[10]),                          # 4
        b[11] * Z**(-b[12] + b[13] * np.log(Z)),    # 5
        b[14] * Z**b[15],                           # 6
    )
    # Tabata 2002 equation 8 (identical to 1996a eq 2)
    detour_factor = 1 / (
        a[1] + a[2] / (
            1 + a[3] / t0**a[4] + a[5] * t0**a[6]))

    return detour_factor


# convenience functions

def extrapolated_range_Si(energy_keV, ref="2002"):
    energy_keV = float(energy_keV)
    range_mm = extrapolated_range_mm(energy_keV, 14, 28.085, 171, 2.329, ref)
    range_um = range_mm * 1e3
    return range_um


def NIST_webbook_Xe_density_gcm3(pressure_bar):
    """
    Lookup table of NIST WebBook's density_gcm3 listing for xenon at various
    pressures.

    Input:
      pressure_bar

    Output:
      density_gcm3_gcm3
    """

    density_gcm3_table = {
        '0.5': 0.0013950,
        '1': 0.0027907,
        '1.5': 0.0041873,
        '2': 0.0055846,
        '2.5': 0.0069827,
        '3': 0.0083816,
        '3.5': 0.0097812,
        '4': 0.011182,
        '4.5': 0.012583,
        '5': 0.013985,
        '10': 0.028046,
        '15': 0.042183,
        '20': 0.056394,
        '25': 0.070677,
        '30': 0.085029,
    }

    if str(pressure_bar) in density_gcm3_table:
        return density_gcm3_table[str(pressure_bar)]
    else:
        return np.nan


def extrapolated_range_xe(energy_keV, p_bar, ref="2002"):
    energy_keV = float(energy_keV)
    density_gcm3 = NIST_webbook_Xe_density_gcm3(p_bar)
    Z = 54
    A = 131.29
    I_eV = 482  # NIST ESTAR
    return extrapolated_range(energy_keV, Z, A, I_eV, density_gcm3, ref)


def extrapolated_range_136xe(energy_keV, p_bar, ref="2002"):
    energy_keV = float(energy_keV)
    density_gcm3 = NIST_webbook_Xe_density_gcm3(p_bar)
    Z = 54
    A = 136     # enriched xenon assumed from NEXT simulations
    I_eV = 482  # NIST ESTAR
    return extrapolated_range(energy_keV, Z, A, I_eV, density_gcm3, ref)
