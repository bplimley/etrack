# tabata.py
# 

import numpy    # for log

def computeDeviationFactor1996a(E_Mev,Z):
    E_Mev = float(E_Mev)
    ELECTRON_REST_ENERGY_MEV = 0.510999
    t0 = E_Mev / ELECTRON_REST_ENERGY_MEV
    # constants from fit: Tabata 1996a Table 1
    B = (0.0,       # (0)
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
    # Tabata 1996a equations 3--8
    a = (0.0,                            # (0)
        B[1] * Z**B[2],                  # 1
        B[3] + B[4]*Z,                   # 2
        B[5] * Z**(B[6] - B[7]*numpy.log(Z)),  # 3
        B[8] / Z**(B[9]),                # 4
        B[10]*Z**(B[11] - B[12]*numpy.log(Z)), # 5
        B[13] * Z**B[14],                # 6
        )
    # Tabata 1996a equation 2
    deviationFactor = 1 / (a[1] + a[2]/(1 + a[3]/t0**a[4] + a[5]*t0**a[6]))
    return deviationFactor
def computeDeviationFactor2002(E_Mev,Z):
    E_Mev = float(E_Mev)
    ELECTRON_REST_ENERGY_MEV = 0.510999
    t0 = E_Mev / ELECTRON_REST_ENERGY_MEV
    # constants from fit: Tabata 2002 Table 2
    B = (0.0,       # (0)
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
    # Tabata 2002 equations 9--14
    a = (0.0,                                   # (0)
        B[1] * Z**B[2],                         # 1
        B[3] * Z**(-B[4] + B[5]*numpy.log(Z)),  # 2
        B[6] * Z**(-B[7] + B[8]*numpy.log(Z)),  # 3
        B[9] * Z**(B[10]),                      # 4
        B[11] * Z**(-B[12]+B[13]*numpy.log(Z)), # 5
        B[14] * Z**B[15],                       # 6
        )
    # Tabata 2002 equation 8 (identical to 1996a eq 2)
    deviationFactor = 1 / (a[1] + a[2]/(1 + a[3]/t0**a[4] + a[5]*t0**a[6]))
    return deviationFactor
def returnUnity(E_Mev,Z):
    return 1.0

def extrapolatedRange(Ekev,Z,A,I_EV,density,ref="2002"):
    def getRefMode(ref):
        if str(ref) == '2002':
            return '2002'
        elif str(ref) == '1996' or str(ref) == '1996a':
            return '1996a'
        elif str(ref).upper() == 'CSDA':
            return 'CSDA'
    def getEnergyRange(refMode):
        # (min, semimin, max)
        # energies all in keV
        if refMode == '2002':
            MIN_ENERGY = 10         # 10 keV from Tabata 2002 sec 2.3
            MAX_ENERGY = 50e3       # 50 MeV from Tabata 2002 sec 1; 2.1
            if any(Ekev < MIN_ENERGY):
                print "Range is inaccurate below 10 keV. Try Tabata 1996a."
            if any(Ekev > MAX_ENERGY):
                print "Range is inaccurate above 50 MeV. Try Tabata 1996a."
        elif refMode == '1996a' or refMode == 'CSDA':
            MIN_ENERGY = 1          # 1 keV from Tabata 1996a p. 2
            SEMI_MIN_ENERGY = 10    # 10 keV from p. 2. (shell effects)
            MAX_ENERGY = 100e3      # 100 MeV from Tabata 1996a p. 2
            if any(Ekev < MIN_ENERGY):
                print "Range is inaccurate below 1 keV."
            if any(Ekev > MIN_ENERGY and Ekev < SEMI_MIN_ENERGY):
                print "Range is less accurate below 10 keV, because shell effects are ignored."
            if any(Ekev > MAX_ENERGY):
                print "Range is inaccurate above 100 MeV."
    def getDeviationFunction(refMode):
        # return function. google says you can do this.
        if refMode == '2002':
            return computeDeviationFactor2002
        elif refMode == '1996a':
            return computeDeviationFactor1996a
        elif refMode == 'CSDA':
            return returnUnity

    # input handling
    Ekev = float(Ekev)
    I_EV = float(I_EV)
    density = float(density)
    refMode = getRefMode(ref)
    deviationFunction = getDeviationFunction(refMode)
    
    E_Mev = Ekev / 1e3
    csdaRangeGCm2 = csdaRange(E_Mev, Z, A, I_EV)
    csdaRangeCm = csdaRangeGCm2 / density
    csdaRangeUm = csdaRangeCm * 1e4

    rangeUm = deviationFunction(E_Mev,Z) * csdaRangeUm
    return rangeUm

def csdaRange(E_Mev, Z, A, I_EV):
    # make sure energy is a float
    E_Mev = float(E_Mev)

    ELECTRON_REST_ENERGY_MEV = 0.510999
    I = I_EV / (1e6*ELECTRON_REST_ENERGY_MEV)
    t0 = E_Mev / ELECTRON_REST_ENERGY_MEV
    
    # Tabata 1996a Table 2
    #   0th element is a placeholder, so that the others line up
    #   with the subscripts in Tabata's paper
    D = (0.0,        # (0)
        3.600,       # 1
        0.9882,      # 2
        1.191e-3,    # 3
        0.8622,      # 4
        1.02501,     # 5
        1.0803e-4,   # 6
        0.99628,     # 7
        1.303e-4,    # 8
        1.02441,     # 9
        1.2986e-4,   # 10
        1.030,       # 11
        1.110e-2,    # 12
        1.10e-6,     # 13
        0.959,       # 14
        )
    # Tabata 1996a equations 13--19
    #   0th element is a placeholder again
    c = (0.0,              # (0)
        D[1]*A / Z**D[2],  # 1
        D[3]*Z**D[4],      # 2
        D[5] - D[6]*Z,     # 3
        D[7] - D[8]*Z,     # 4
        D[9] - D[10]*Z,    # 5
        D[11]/Z**D[12],    # 6
        D[13]*Z**D[14],    # 7
        )
    # Tabata 1996a equation 12
    B = numpy.log((t0 / (I + c[7]*t0))**2) + numpy.log(1 + t0/2)
    # Tabata 1996a equation 11
    csdaRangeGCm2 = c[1] / B * (
        numpy.log(1+c[2]*t0**c[3])/c[2] - 
        c[4]*t0**c[5]/(1 + c[6]*t0)
        )
    return csdaRangeGCm2

def extrapolatedRangeSi(E_kev,ref="2002"):
    E_kev = float(E_kev)
    return extrapolatedRange(E_kev, 14, 28.085, 171, 2.329, ref)
