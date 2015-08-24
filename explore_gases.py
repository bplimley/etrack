import numpy as np
import sys
import os

os.chdir('physics')
import tabata
os.chdir('..')

def z_eff(n,z,a):
    n = np.array(n)
    z = np.array(z)
    a = np.array(a)

    f = np.zeros(len(n))
    # weight by mass
    for i in range(len(n)):
        f[i] = n[i]*a[i] / np.sum(n*a)

    zeff = np.sum(f*z)
    aeff = np.sum(f*a)

    return zeff, aeff

def calculate_mfp_cm(mu_cm2g, rho_gL):
    # at 1 bar, 293 K, for 661.6 keV photons
    return 1.0 / (mu_cm2g * rho_gL/1000)

def print_stuff(textlabel,rex_cm,mfp_cm):
    print("  {}:".format(textlabel))
    print("Mean free path of 662 keV photons at 10 bar is:       {:.0f} cm".format(mfp_cm / 10))
    print("Extrapolated range of 300 keV electrons at 10 bar is: {:.1f} cm".format(rex_cm / 10))
    print("2% of range is:     {:.2f} mm".format(rex_cm / 10 * 10 * 0.02))
    print("Rex/MFP = {:.4f}".format(rex_cm/mfp_cm))
    print(" ")

Ekev = 300

# CO2
Z,A = z_eff( (1,2), (6,8), (12.011,15.999) )
rho_gL = 1.8403     # at 1 bar and 293 K
I_eV = 85.0
rex_cm = tabata.extrapolatedRange(Ekev,Z,A,I_eV,rho_gL/1000) / 1e4
mu_cm2g = 0.07712   # for 661.6 keV photons, without coherent scattering
mfp_cm = calculate_mfp_cm(mu_cm2g, rho_gL)
print_stuff("CO2",rex_cm,mfp_cm)
print(Z)

# N2
Z = 7
A = 14.007
rho_gL = 1.1654     # at 1 bar and 293 K
I_eV = 82.0
rex_cm = tabata.extrapolatedRange(Ekev,Z,A,I_eV,rho_gL/1000) / 1e4
mu_cm2g = 0.07712   # for 661.6 keV photons
mfp_cm = calculate_mfp_cm(mu_cm2g, rho_gL)
print_stuff("N2",rex_cm,mfp_cm)

# CH4
Z,A = z_eff( (1,4), (6,1), (12.011,1.008) )
rho_gL = 0.66850    # at 1 bar and 293 K
I_eV = 41.7
rex_cm = tabata.extrapolatedRange(Ekev,Z,A,I_eV,rho_gL/1000) / 1e4
mu_cm2g = 0.09621   # for 661.6 keV photons
mfp_cm = calculate_mfp_cm(mu_cm2g, rho_gL)
print_stuff("CH4",rex_cm,mfp_cm)
print(Z)

# He
Z = 2
A = 4.0026
rho_gL = 0.16640    # at 1 bar and 293 K
I_eV = 41.8
rex_cm = tabata.extrapolatedRange(Ekev,Z,A,I_eV,rho_gL/1000) / 1e4
mu_cm2g = 0.07713   # for 661.6 keV photons
mfp_cm = calculate_mfp_cm(mu_cm2g, rho_gL)
print_stuff("He",rex_cm,mfp_cm)

# Ne
Z = 10
A = 20.180
rho_gL = 0.83890    # at 1 bar and 293 K
I_eV = 137.0
rex_cm = tabata.extrapolatedRange(Ekev,Z,A,I_eV,rho_gL/1000) / 1e4
mu_cm2g = 0.07645   # for 661.6 keV photons
mfp_cm = calculate_mfp_cm(mu_cm2g, rho_gL)
print_stuff("Ne",rex_cm,mfp_cm)

# Ar
Z = 18
A = 39.948
rho_gL = 1.6627     # at 1 bar and 293 K
I_eV = 188.0
rex_cm = tabata.extrapolatedRange(Ekev,Z,A,I_eV,rho_gL/1000) / 1e4
mu_cm2g = 0.06962   # for 661.6 keV photons
mfp_cm = calculate_mfp_cm(mu_cm2g, rho_gL)
print_stuff("Ar",rex_cm,mfp_cm)

# Xe
Z = 54
A = 131.29
rho_gL = 5.4914     # at 1 bar and 293 K
I_eV = 482.0
rex_cm = tabata.extrapolatedRange(Ekev,Z,A,I_eV,rho_gL/1000) / 1e4
mu_cm2g = 0.07354   # for 661.6 keV photons
mfp_cm = calculate_mfp_cm(mu_cm2g, rho_gL)
print_stuff("Xe",rex_cm,mfp_cm)
