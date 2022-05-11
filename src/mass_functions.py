import numpy as np

from amuse.units import units


def stellar_radius (M):
    '''
    Stellar radius, anchored to Sun, indices obtained from SeBa by eye
    '''

    if M > 1. | units.MSun:
        return (1. | units.RSun) * (M.value_in(units.MSun))**0.6
    else:
        return (1. | units.RSun) * (M.value_in(units.MSun))**0.8


def ionizing_flux (M, solar=True):
    '''
    Compute the ionizing flux of stars as a function of their mass, following Avedisova 1979
    Values are computed using linear interpolation in log space between data points
    M: masses of stars to compute ionizing flux of, in Solar masses (1D float array, shape (N))
    solar: use data for z=0.02 if True, use data for z=0.04 if False
    returns a 1D float array, shape (N), containing the ionising fluxes, in s^-1, corresponding to each stellar mass
    '''

    if solar:
        mass, logflux = np.loadtxt('ionising_flux_z02.txt', unpack=True)
    else:
        mass, logflux = np.loadtxt('ionising_flux_z04.txt', unpack=True)

    mass = mass[::-1]
    logflux = logflux[::-1]

    logmass = np.log10(mass)

    m = M.value_in(units.MSun)

    N = len(M)
    n = len(mass)

    Ndot = np.zeros(N)

    for i in range(n-1):

        mask = (m >= mass[i])*(m < mass[i+1])

        Ndot[ mask ] = 10.**( (logflux[i+1] - logflux[i])/(logmass[i+1] - logmass[i])*(np.log10(m[ mask ]) - logmass[i]) + logflux[i] )

    mask = m >= mass[-1]

    Ndot[ mask ] = 10.**( (logflux[-1] - logflux[-2])/(logmass[-1] - logmass[-2])*(np.log10(mass[-1]) - logmass[-2]) + logflux[-2] )

    return Ndot | units.s**-1


def FUV_luminosity_from_mass (M):
    '''
    Compute the FUV-luminosity from stellar mass according to the power
    law fit derived from the UVBLUE spectra (at z=0.0122, 
    Rodriguez-Merino 2005). The file 'ML_fit.txt' is needed in the
    same folder. Masses are put in in solar masses, and the 
    luminosities are put out in solar luminosities.
    '''

    A, B, mass = np.loadtxt('../data/ML_fit.txt', unpack=True)

    m = M.value_in(units.MSun)

    N = len(M)
    n = len(mass)

    L = np.zeros(N)

    for i in range(n-1):

        mask = (m >= mass[i])*(m < mass[i+1])

        L[ mask ] = 10.**(A[i]*np.log10(mass[i]) + B[i])

    mask = m >= mass[-1]

    L[ mask ] = 10.**(A[-1]*np.log10(mass[-1]) + B[-1])

    return L | units.LSun
