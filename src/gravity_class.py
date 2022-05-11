import numpy as np

import time

from amuse.units import units, nbody_system, constants
from amuse.datamodel import Particles
from amuse.io import write_set_to_file

from amuse.community.ph4.interface import ph4
from amuse.community.huayno.interface import Huayno
from amuse.community.seba.interface import SeBa

from mass_functions import FUV_luminosity_from_mass


class Gravity:

    def __init__ (self, sys_params, stellar_evolution=False, begin_time=0.|units.Myr,
            ppd_population=None):

        self.converter = nbody_system.nbody_to_si(1e4 | units.MSun, 3. | units.pc)

        '''
        if sys_params['verbosity']:
            #self.grav_code = Huayno(self.converter, redirection='none')
            self.grav_code = ph4(self.converter, redirection='none')
        else:
            #self.grav_code = Huayno(self.converter)
            self.grav_code = ph4(self.converter)
        '''
        self.grav_code = ph4(self.converter)

        if stellar_evolution:
            self.stellar_code = SeBa()
        else:
            self.stellar_code = None

        self.model_time = begin_time

        # Potential pointer to particle set that should be removed from radiative
        # Since stars are also particles in the radiative code of hydro_class,
        # the particles removed here must also be removed there
        #self.rad_particles_to_remove = None

        self.verbosity = sys_params['verbosity']
        self.box_size = sys_params['box_size']

        #self.grav_code.parameters.timestep_parameter = sys_params['eta']
        self.grav_code.parameters.force_sync = True
        self.grav_code.evolve_model(begin_time)

        self.dE = 0.

        #self.grav_code.parameters.stopping_conditions_out_of_box_size = self.box_size/2.
        #self.oob_detection = self.grav_code.stopping_conditions.out_of_box_detection
        #self.oob_detection.enable()

        self.collision_detection = self.grav_code.stopping_conditions.collision_detection
        self.collision_detection.enable()

        self.star_particles = Particles()
        self.bridge_particles = Particles()
        self.removed_particles = Particles()

        # Necessary for bridge
        self.get_gravity_at_point = self.grav_code.get_gravity_at_point
        self.get_potential_at_point = self.grav_code.get_potential_at_point

        # Channels running between class particles and code particles
        self.channels = {
            'star_to_grav': self.star_particles.new_channel_to(self.grav_code.particles),
            'grav_to_star': self.grav_code.particles.new_channel_to(self.star_particles),
            'star_to_stellar': None,
            'stellar_to_star': None,
            'star_to_ext': None,
            'grav_to_bridge': self.grav_code.particles.new_channel_to(self.bridge_particles)
        }

        if stellar_evolution:
            self.channels['star_to_stellar'] = \
                self.star_particles.new_channel_to(self.stellar_code.particles)
            self.channels['stellar_to_star'] = \
                self.stellar_code.particles.new_channel_to(self.star_particles)

        self.ppd_population = ppd_population
        if ppd_population is not None:
            self.channels['star_to_ppd'] = self.star_particles.new_channel_to(
                self.ppd_population.star_particles)
            self.channels['ppd_to_star'] = self.ppd_population.star_particles.new_channel_to(
                self.star_particles)
            self.ppd_population.collision_detector = self.collision_detection


    @property
    def particles (self):
        # For bridge purposes; needs total mass in mass, not just stellar mass
        return self.bridge_particles


    def stop (self):
        self.grav_code.stop()
        if self.stellar_code is not None:
            self.stellar_code.stop()


    def add_new_star_particles (self, particles):

        self.star_particles.add_particles(particles)
        self.bridge_particles.add_particles(particles)
        self.grav_code.particles.add_particles(particles)

        if self.stellar_code is not None:
            self.stellar_code.particles.add_particles(particles)
            self.channels['stellar_to_star'].copy()
            self.star_particles[-len(particles):].initial_mass = particles.mass
            self.star_particles[-len(particles):].fuv_luminosity = \
                FUV_luminosity_from_mass(particles.mass)


    def add_old_star_particles (self, particles):
        '''
        Add evolved stars. Since SeBa assumes ZAMS stars, old stars must be
        evolved to their actual age.

        [IMPORTANT] Only call this function once, with the entire old stellar
        population! 
        '''

        # safeguard; 
        if self.stellar_code is None:
            self.add_new_star_particles(particles)
            return

        self.star_particles.add_particles(particles)
        self.bridge_particles.add_particles(particles)

        particles_copy = particles.copy()
        particles_copy.mass = particles_copy.initial_mass

        age_order = np.argsort(particles_copy.age.value_in(units.Myr))[::-1]

        stellar_age = particles.age.max()

        print ("[GRAV] Starting stellar reaging process", flush=True)
        start = time.time()

        for i in range(len(particles_copy)):
            self.stellar_code.particles.add_particle(particles_copy[age_order[i]])
            if i < len(particles_copy)-1:
                self.stellar_code.evolve_model(
                    stellar_age-particles_copy.age[age_order[i+1]])
            else:
                self.stellar_code.evolve_model(stellar_age)

        end = time.time()
        print ("[GRAV] Stellar reaging finished in {a} s".format(a=end-start),
            flush=True)

        # Check reaging process
        counter = 0
        for i in range(len(particles_copy)):
            # Ages must agree within numerical precision
            if np.abs((particles_copy[i].age - \
                    self.stellar_code.particles.select_array(
                    lambda key: key == particles_copy[i].key, ['key']).age[0]
                    ).value_in(units.Myr)) > 1e-15:
                counter += 1

        if counter > 0:
            print ("[GRAV] {a} stars wrongly aged".format(a=counter), flush=True)
            print ("[GRAV]", particles_copy.age.value_in(units.Myr), self.stellar_code.particles.age.value_in(units.Myr), flush=True)

        self.channels['stellar_to_star'].copy()
        self.channels['star_to_grav'].copy_attributes(['gravity_mass'], ['mass'])
        self.channels['grav_to_bridge'].copy()

        self.star_particles.fuv_luminosity = FUV_luminosity_from_mass(self.star_particles.mass)

        self.grav_code.particles.add_particles(self.star_particles)


    def evolve_model (self, end_time):

        dt = end_time - self.model_time
        if self.verbosity:
            start = time.time()


        if self.stellar_code is not None:
            # No pre-copy because stellar evolution is not influenced by other codes
            self.stellar_code.evolve_model( self.stellar_code.model_time + dt/2. )
            # Stellar evolution changed mass
            self.channels['stellar_to_star'].copy_attributes(['mass'])


        # Stellar evolution changed mass, updates rotation curve
        self.channels['star_to_ppd'].copy_attributes(['mass'])
        # Propagate updated stellar mass to gravity_mass, reset collision radii
        self.channels['ppd_to_star'].copy_attributes(['gravity_mass', 'radius'])
        # Further propagate above properties; stars have been kicked, do this too
        self.channels['star_to_grav'].copy_attributes(
            ['gravity_mass', 'radius', 'vx', 'vy', 'vz'],
            target_names=['mass', 'radius', 'vx', 'vy', 'vz'])


        E0 = self.grav_code.kinetic_energy + self.grav_code.potential_energy


        # ph4 hangs with 1 particle; this drifts a single particle, and evolves the code
        # so there is no spurious evolution later
        if len(self.star_particles) == 1:
            self.grav_code.particles.remove_particles(self.star_particles)
            self.star_particles.position += self.star_particles.velocity * dt
            self.grav_code.evolve_model(end_time)
            self.grav_code.particles.add_particles(self.star_particles)
        else:
            self.grav_code.evolve_model(end_time)


        # Attributes that gravity alters
        self.channels['grav_to_star'].copy_attributes(['x', 'y', 'z', 'vx', 'vy', 'vz'])


        while self.collision_detection.is_set():#or self.oob_detection.is_set():

            ''' OOB detection is not present in ph4, not necessary in this version
            if self.oob_detection.is_set():
                bs = self.grav_code.parameters.stopping_conditions_out_of_box_size

                oob_particles = self.star_particles.select_array(
                    lambda R: R.lengths() > bs,
                    ['position'])

                self.grav_code.particles.remove_particles(oob_particles)
                self.removed_particles.add_particles(oob_particles)

                print ("[GRAV] Removing {a} particles".format(a=len(oob_particles), 
                    flush=True))
            '''

            #if self.collision_detection.is_set():
            print ("[GRAV] Encounter between stars!", flush=True)

            # Update ppd's stellar attributes to resolve encounters
            self.channels['star_to_ppd'].copy_attributes(['x', 'y', 'z', 'vx', 'vy', 'vz'])

            self.ppd_population.resolve_encounters()

            # Remove truncated disk mass, radius updated to below periastron
            self.channels['ppd_to_star'].copy_attributes(['gravity_mass', 'radius'])
            self.channels['star_to_grav'].copy_attributes(['gravity_mass', 'radius'],
                ['mass', 'radius'])

            self.grav_code.evolve_model(end_time)

            # Attributes that gravity alters
            self.channels['grav_to_star'].copy_attributes(
                ['x', 'y', 'z', 'vx', 'vy', 'vz'],
                target_names=['x', 'y', 'z', 'vx', 'vy', 'vz'])


        self.model_time = end_time

        E1 = self.grav_code.kinetic_energy + self.grav_code.potential_energy
        self.dE = np.abs((E1 - E0)/E0)

        '''
        # Handle removed particles
        if len(self.removed_particles):
            self.star_particles.remove_particles(self.removed_particles)
            if self.stellar_code is not None:
                self.stellar_code.particles.remove_particles(self.removed_particles)
            if self.ppd_population is not None:
                self.ppd_population.select_array(
                    lambda key: key in self.removed_particles.key, ['key']).star_ejected = True
            self.removed_particles = Particles()
        '''

        if self.stellar_code is not None:
            # No pre-copy because stellar evolution is not influenced by other codes
            self.stellar_code.evolve_model( self.stellar_code.model_time + dt/2. )
            # Stellar evolution changed mass
            self.channels['stellar_to_star'].copy_attributes(['mass', 'age'])

        if self.channels['star_to_ext'] is not None:
            self.channels['star_to_ext'].copy()


        self.star_particles.fuv_luminosity = FUV_luminosity_from_mass(self.star_particles.mass)
        # Subsequent evolution of ppd_population requires positions and fuv_luminosity
        # (for radiation field) and mass (for rotation curve)
        self.channels['star_to_ppd'].copy_attributes(['x', 'y', 'z', 'mass', 'fuv_luminosity'])
        self.channels['grav_to_bridge'].copy()


        if self.verbosity:
            end = time.time()
            print ("[GRAV] Gravity ran {b} Myr in {a} s".format(a=end-start,
                b=dt.value_in(units.Myr)), flush=True)


    def save_particle_set (self, filepath, index):

        if len(self.star_particles):
            filename = "{a}/grav_star_particles_i{b:05}.hdf5".format(a=filepath, b=index)
            write_set_to_file(self.star_particles, filename, "hdf5", 
                timestamp=self.model_time, append_to_file=False, 
                overwrite_file=True)


    def print_diagnostics (self):

        print ("[GRAV] Current class time: {a} Myr; code time: {b} Myr".format(
            a=self.model_time.value_in(units.Myr), 
            b=self.grav_code.model_time.value_in(units.Myr)))
        print ("[GRAV] #Particles in class: {a}; #Particles in code: {b}".format(
            a=len(self.star_particles), b=len(self.grav_code.particles)))
        print ("[GRAV] Total mass: {a} MSun".format(
            a=self.star_particles.mass.sum().value_in(units.MSun)))
        print ("[GRAV] Center of mass: {a} pc".format(
            a=self.star_particles.center_of_mass().value_in(units.parsec)))
        print ("[GRAV] Center of mass velocity: {a} pc/Myr".format(
            a=self.star_particles.center_of_mass_velocity().value_in(units.parsec/units.Myr)))
        print ("[GRAV] Relative energy error in last step: {a}".format(a=self.dE))
