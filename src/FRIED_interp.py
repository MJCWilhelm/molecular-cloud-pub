import numpy as np
import scipy.interpolate as si


# Set to true if your environment can run amuse
AMUSE_ENABLED = True

if AMUSE_ENABLED:
    import amuse.lab as al
    G0 = 1.6e-3 * al.units.erg/al.units.s/al.units.cm**2.


class FRIED_interpolator:
    '''
    FRIED interpolator object
    Performs linear interpolation on the FRIED grid
    As there is a degeneracy between disk radius, disk mass and disk outer density, 
    we neglect disk outer density
    Interpolation can optionally be done in lin or log space, separately for each 
    axis
    '''


    def __init__ (self, folder='.', logMstar=True, logF=True, logMdisk=True, 
            logRdisk=True, verbosity=False):
        '''
        Initialize FRIED_interpolator object

        folder: directory of FRIED grid data file (string)
        logMstar: construct grid of the log of the host star mass (else linear) 
            (boolean)
        logF: construct grid of the log of the incident FUV field (else linear) 
            (boolean)
        logMdisk: construct grid of the log of the disk mass (else linear) (boolean)
        logRdisk: construct grid of the log of the disk radius (else linear) 
            (boolean)
        '''

        Mstar_grid, F_grid, Mdisk_grid, Rdisk_grid = np.loadtxt(
            folder+'/friedgrid.dat', usecols=(0,1,2,4), unpack=True)
        self._logMdot_grid = np.loadtxt(folder+'/friedgrid.dat', usecols=(5,))

        self._logMstar = logMstar
        self._logF = logF
        self._logMdisk = logMdisk
        self._logRdisk = logRdisk

        if self._logMstar:
            Mstar_grid = np.log10(Mstar_grid)
        if self._logF:
            F_grid = np.log10(F_grid)
        if self._logMdisk:
            Mdisk_grid = np.log10(Mdisk_grid)
        if self._logRdisk:
            Rdisk_grid = np.log10(Rdisk_grid)

        self._grid = np.array([
            Mstar_grid,
            F_grid,
            Mdisk_grid,
            Rdisk_grid,
        ]).T

        self._interpolator = si.LinearNDInterpolator(self._grid, self._logMdot_grid)
        self._backup_interpolator = si.NearestNDInterpolator(self._grid,
            self._logMdot_grid)

        self.backup_counter = 0

        self.verbosity = verbosity


    def interp (self, Mstar, F, Mdisk, Rdisk):
        '''
        Compute mass loss rate at N positions on the grid
        Input and output is in standard FRIED units
        Lin/log space interpolation is handled in code

        Mstar: host star mass (1D float array, shape (N))
        F: host star mass (1D float array, shape (N))
        Mdisk: disk mass (1D float array, shape (N))
        Rdisk: disk radius (1D float array, shape (N))

        Returns mass loss rate for each set of parameters (1D float array, shape(N))
        '''

        try:
            N = len(Mstar)
        except:
            N = 1

        if self._logMstar:
            Mstar = np.log10(Mstar)
        if self._logF:
            F = np.log10(F)
        if self._logMdisk:
            Mdisk = np.log10(Mdisk)
        if self._logRdisk:
            Rdisk = np.log10(Rdisk)

        x_i = np.array([
            Mstar*np.ones(N),
            F*np.ones(N),
            Mdisk*np.ones(N),
            Rdisk*np.ones(N)
        ]).T

        logMdot = self._interpolator(x_i)

        mask_nan = np.isnan(logMdot)

        logMdot[ mask_nan ] = self._backup_interpolator(x_i[ mask_nan ])

        self.backup_counter += np.sum(mask_nan)
        if self.verbosity:
            print ("[WARNING] {a} points outside interpolation domain, falling back to nearest neighbour".format(a=np.sum(mask_nan)), flush=True)

        return 10.**logMdot


    if AMUSE_ENABLED:

        def interp_amuse (self, Mstar, F, Mdisk, Rdisk):
            '''
            Compute mass loss rate at N positions on the grid
            Input and output is in amuse units
            Lin/log space interpolation is handled in code

            Mstar: host star mass (1D mass unit scalar array, shape (N))
            F: host star mass (1D flux unit scalar array, shape (N))
            Mdisk: disk mass (1D mass unit scalar array, shape (N))
            Rdisk: disk radius (1D length unit scalar array, shape (N))

            Returns mass loss rate for each set of parameters (1D mass flux per 
            time unit scalar array, shape (N))
            '''

            try:
                N = len(Mstar)
            except:
                N = 1

            if self._logMstar:
                Mstar = np.log10(Mstar.value_in(al.units.MSun))
            else:
                Mstar = Mstar.value_in(al.units.MSun)
            if self._logF:
                F = np.log10(F.value_in(G0))
            else:
                F = F.value_in(G0)
            if self._logMdisk:
                Mdisk = np.log10(Mdisk.value_in(al.units.MJupiter))
            else:
                Mdisk = Mdisk.value_in(al.units.MJupiter)
            if self._logRdisk:
                Rdisk = np.log10(Rdisk.value_in(al.units.AU))
            else:
                Rdisk = Rdisk.value_in(al.units.AU)

            x_i = np.array([
                Mstar*np.ones(N),
                F*np.ones(N),
                Mdisk*np.ones(N),
                Rdisk*np.ones(N)
            ]).T

            logMdot = self._interpolator(x_i)

            mask_nan = np.isnan(logMdot)

            logMdot[ mask_nan ] = self._backup_interpolator(x_i[ mask_nan ])

            self.backup_counter += np.sum(mask_nan)
            if self.verbosity:
                print ("[WARNING] {a} points outside interpolation domain, falling back to nearest neighbour".format(a=np.sum(mask_nan)), flush=True)

            return 10.**logMdot | al.units.MSun/al.units.yr


class Haworth2018_interpolator:
    '''
    FRIED interpolator object
    Performs linear interpolation on the FRIED grid
    As there is a degeneracy between disk radius, disk mass and disk outer density, we neglect disk outer density
    Interpolation can optionally be done in lin or log space, separately for each axis
    '''


    def __init__ (self, folder='.', logF=True, logsigma=True, logRdisk=True, verbosity=False):
        '''
        Initialize FRIED_interpolator object

        folder: directory of FRIED grid data file (string)
        logMstar: construct grid of the log of the host star mass (else linear) (boolean)
        logF: construct grid of the log of the incident FUV field (else linear) (boolean)
        logMdisk: construct grid of the log of the disk mass (else linear) (boolean)
        logRdisk: construct grid of the log of the disk radius (else linear) (boolean)
        '''

        sigma_grid_1, Rdisk_grid_1, logMdot_grid_1 = np.loadtxt(folder+'/grid10_upto50_incBig.dat', usecols=(0,1,5), unpack=True)
        sigma_grid_2, Rdisk_grid_2, logMdot_grid_2 = np.loadtxt(folder+'/grid100_upto50_incBig.dat', usecols=(0,1,5), unpack=True)
        sigma_grid_3, Rdisk_grid_3, logMdot_grid_3 = np.loadtxt(folder+'/grid1000_upto50_incBig.dat', usecols=(0,1,5), unpack=True)
        sigma_grid_4, Rdisk_grid_4, logMdot_grid_4 = np.loadtxt(folder+'/grid10000_upto50_incBig.dat', usecols=(0,1,5), unpack=True)

        sigma_grid = np.concatenate((sigma_grid_1, sigma_grid_2, sigma_grid_3, sigma_grid_4))
        Rdisk_grid = np.concatenate((Rdisk_grid_1, Rdisk_grid_2, Rdisk_grid_3, Rdisk_grid_4))
        F_grid = np.concatenate((np.ones(len(sigma_grid_1))*10., np.ones(len(sigma_grid_2))*100., 
                            np.ones(len(sigma_grid_3))*1000., np.ones(len(sigma_grid_4))*10000.))

        logMdot_grid = np.concatenate((logMdot_grid_1, logMdot_grid_2, logMdot_grid_3, logMdot_grid_4))

        self._logMdot_grid = logMdot_grid

        self._logF = logF
        self._logsigma = logsigma
        self._logRdisk = logRdisk

        if self._logF:
            F_grid = np.log10(F_grid)
        if self._logsigma:
            sigma_grid = np.log10(sigma_grid)
        if self._logRdisk:
            Rdisk_grid = np.log10(Rdisk_grid)

        self._grid = np.array([
            F_grid,
            sigma_grid,
            Rdisk_grid,
        ]).T

        self._interpolator = si.LinearNDInterpolator(self._grid, self._logMdot_grid)
        self._backup_interpolator = si.NearestNDInterpolator(self._grid, self._logMdot_grid)

        self.backup_counter = 0

        self.verbosity = verbosity


    def interp (self, F, sigma, Rdisk):
        '''
        Compute mass loss rate at N positions on the grid
        Input and output is in standard FRIED units
        Lin/log space interpolation is handled in code

        Mstar: host star mass (1D float array, shape (N))
        F: host star mass (1D float array, shape (N))
        Mdisk: disk mass (1D float array, shape (N))
        Rdisk: disk radius (1D float array, shape (N))

        Returns mass loss rate for each set of parameters (1D float array, shape (N))
        '''

        try:
            N = len(F)
        except:
            N = 1

        if self._logF:
            F = np.log10(F)
        if self._logsigma:
            sigma = np.log10(sigma)
        if self._logRdisk:
            Rdisk = np.log10(Rdisk)

        x_i = np.array([
            F*np.ones(N),
            sigma*np.ones(N),
            Rdisk*np.ones(N)
        ]).T

        logMdot = self._interpolator(x_i)

        mask_nan = np.isnan(logMdot)

        logMdot[ mask_nan ] = self._backup_interpolator(x_i[ mask_nan ])

        self.backup_counter += np.sum(mask_nan)
        if self.verbosity:
            print ("[WARNING] {a} points outside interpolation domain, falling back to nearest neighbour".format(a=np.sum(mask_nan)), flush=True)

        return 10.**logMdot


    if AMUSE_ENABLED:

        def interp_amuse (self, F, sigma, Rdisk):
            '''
            Compute mass loss rate at N positions on the grid
            Input and output is in amuse units
            Lin/log space interpolation is handled in code

            Mstar: host star mass (1D mass unit scalar array, shape (N))
            F: host star mass (1D flux unit scalar array, shape (N))
            Mdisk: disk mass (1D mass unit scalar array, shape (N))
            Rdisk: disk radius (1D length unit scalar array, shape (N))

            Returns mass loss rate for each set of parameters (1D mass flux per time unit scalar array, shape (N))
            '''

            try:
                N = len(F)
            except:
                N = 1

            if self._logF:
                F = np.log10(F.value_in(G0))
            else:
                F = F.value_in(G0)
            if self._logsigma:
                sigma = np.log10(sigma.value_in(al.units.g/al.units.cm**2))
            else:
                sigma = sigma.value_in(al.units.g/al.units.cm**2)
            if self._logRdisk:
                Rdisk = np.log10(Rdisk.value_in(al.units.AU))
            else:
                Rdisk = Rdisk.value_in(al.units.AU)

            x_i = np.array([
                F*np.ones(N),
                sigma*np.ones(N),
                Rdisk*np.ones(N)
            ]).T

            logMdot = self._interpolator(x_i)

            mask_nan = np.isnan(logMdot)

            logMdot[ mask_nan ] = self._backup_interpolator(x_i[ mask_nan ])

            self.backup_counter += np.sum(mask_nan)
            if self.verbosity:
                print ("[WARNING] {a} points outside interpolation domain, falling back to nearest neighbour".format(a=np.sum(mask_nan)), flush=True)

            return 10.**logMdot | al.units.MSun/al.units.yr


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import time

    N = 1000

    Mdisk = np.logspace(-3., 3., num=N)
    Mstar = 1.*np.ones(len(Mdisk))
    F = 1000.*np.ones(len(Mdisk))
    Rdisk = 100.*np.ones(len(Mdisk))


    print ('Interpolations in log space:')
    start = time.time()

    interpolator = FRIED_interpolator(folder='../../DiskClass/', verbosity=False)

    end = time.time()

    print ('Building the interpolator for a {a} point grid took {b} s'.format(a=interpolator._grid.shape[0], b=end-start))

    Mdot = np.zeros(len(F))

    start = time.time()

    Mdot = interpolator.interp(Mstar, F, Mdisk, Rdisk)

    end = time.time()

    print ('Performing {a} interpolations took {b} s, \nor {c} s per interpolation'.format(a=N, b=end-start, c=(end-start)/N))


    on_grid_mask = (interpolator._grid[:,0] == np.log10(Mstar[0]))*\
                   (interpolator._grid[:,1] == np.log10(F[0]))*\
                   (interpolator._grid[:,3] == np.log10(Rdisk[0]))

    Mdisk_on_grid = interpolator._grid[ on_grid_mask, 2 ]

    plt.figure()
    plt.scatter(Mdisk_on_grid, interpolator._logMdot_grid[ on_grid_mask ])
    plt.plot(np.log10(Mdisk), np.log10(Mdot))

    plt.xlabel('log10 Disk Mass (MJup)')
    plt.ylabel('log10 Disk Mass Loss Rate (MSun/yr)')

    plt.title('Interpolations along disk mass slice, in log space')

    print (interpolator.backup_counter)




    '''
    print ('Interpolations in lin space:')
    start = time.time()

    interpolator = FRIED_interpolator(logMstar=False, logF=False, logMdisk=False, logRdisk=False)

    end = time.time()

    print ('Building the interpolator for a {a} point grid took {b} s'.format(a=interpolator._grid.shape[0], b=end-start))


    start = time.time()

    Mdot = interpolator.interp(Mstar, F, Mdisk, Rdisk)

    end = time.time()

    print ('Performing {a} interpolations took {b} s, \nor {c} s per interpolation'.format(a=N, b=end-start, c=(end-start)/N))


    on_grid_mask = (interpolator._grid[:,0] == Mstar[0])*\
                   (interpolator._grid[:,1] == F[0])*\
                   (interpolator._grid[:,3] == Rdisk[0])

    Mdisk_on_grid = np.log10(interpolator._grid[ on_grid_mask, 2 ])


    plt.figure()
    plt.scatter(Mdisk_on_grid, interpolator._logMdot_grid[ on_grid_mask ])
    plt.plot(np.log10(Mdisk), np.log10(Mdot))

    plt.xlabel('log10 Disk Mass (MJup)')
    plt.ylabel('log10 Disk Mass Loss Rate (MSun/yr)')

    plt.title('Interpolations along disk mass slice, in lin space')
    '''

    plt.show()
