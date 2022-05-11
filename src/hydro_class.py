import numpy as np

import time

from amuse.units import units, nbody_system, constants
from amuse.datamodel import Particles
from amuse.ext.sink import new_sink_particles
from amuse.io import write_set_to_file

from amuse.community.fi.interface import Fi

from cooling_class import SimplifiedThermalModelEvolver

from amuse.ext.stellar_wind import new_stellar_wind


class Hydro:

    def __init__ (self, sys_params, begin_time=0.|units.Myr):

        self.converter = nbody_system.nbody_to_si(1e4 | units.MSun, 3.|units.pc)

        if sys_params['verbosity']:
            self.hydro_code = Fi(self.converter, mode='openmp', redirection='none')

        else:
            self.hydro_code = Fi(self.converter, mode='openmp')

        self.verbosity = sys_params['verbosity']
        self.model_time = begin_time
        self.begin_time = begin_time
        self.box_size = sys_params['box_size']
        self.density_threshold = sys_params['density_threshold']
        self.merge_radius = sys_params['merge_radius']
        self.delay_time = sys_params['delay_time']

        self.hydro_code.parameters.use_hydro_flag = True
        self.hydro_code.parameters.self_gravity_flag = True
        self.hydro_code.parameters.periodic_box_size = self.box_size
        self.hydro_code.parameters.isothermal_flag = True
        self.hydro_code.parameters.integrate_entropy_flag = False
        self.hydro_code.parameters.gamma = 1
        self.hydro_code.parameters.begin_time = begin_time
        self.hydro_code.parameters.timestep = sys_params['dt_hydro']
        self.hydro_code.parameters.gas_epsilon = sys_params['gas_epsilon']

        self.hydro_code.parameters.direct_sum_flag = True

        self.hydro_code.parameters.stopping_condition_maximum_density = self.density_threshold
        self.sf_detection = self.hydro_code.stopping_conditions.density_limit_detection
        self.sf_detection.enable()


        self.gas_particles   = Particles()
        self.sink_particles  = Particles()

        # Necessary for bridge
        self.get_gravity_at_point = self.hydro_code.get_gravity_at_point
        self.get_potential_at_point = self.hydro_code.get_potential_at_point
        self.get_hydro_state_at_point = self.hydro_code.get_hydro_state_at_point


        # Channels running between main gas/sink sets 
        self.channels = {
            'gas_to_hydro': self.gas_particles.new_channel_to(self.hydro_code.gas_particles),
            'hydro_to_gas': self.hydro_code.gas_particles.new_channel_to(self.gas_particles),
            'sink_to_hydro': self.sink_particles.new_channel_to(self.hydro_code.dm_particles),
            'hydro_to_sink': self.hydro_code.dm_particles.new_channel_to(self.sink_particles),
        }


        if sys_params['stellar_wind']:
            self.stellar_wind_code = new_stellar_wind(sys_params['M_wind_particles'], 
                target_gas = self.gas_particles, timestep=sys_params['dt_wind'],
                derive_from_evolution=True, mode='simple')
        else:
            self.stellar_wind_code = None


        if sys_params['cooling']:
            self.cooling = SimplifiedThermalModelEvolver(self.gas_particles)
            self.cooling.model_time = begin_time
        else:
            self.cooling = None


    def add_gas_particles (self, particles):

        self.gas_particles.add_particles(particles)
        self.hydro_code.gas_particles.add_particles(particles)


    def add_sink_particles (self, particles):

        self.sink_particles.add_particles(particles)
        self.hydro_code.dm_particles.add_particles(particles)


    @property
    def particles (self):
        return self.hydro_code.particles


    def evolve_model (self, end_time):

        dt = end_time - self.model_time

        if self.cooling is not None:
            self.cooling.evolve_for( dt/2. )

        # Add new stellar wind particles
        if self.stellar_wind_code is not None and \
                len(self.stellar_wind_code.particles) > 0:
            self.stellar_wind_code.evolve_model( self.model_time + dt/2. )

            if len(self.gas_particles) != len(self.hydro_code.gas_particles):
                print ("[HYDRO] {a} new stellar wind particle(s)".format(
                    a=len(self.gas_particles)-len(self.hydro_code.gas_particles)), 
                    flush=True)
                self.gas_particles.synchronize_to(self.hydro_code.gas_particles)
                print ("[HYDRO] Now {a} in hydro gas".format(
                    a=len(self.hydro_code.gas_particles)), flush=True)
            else:
                print ("[HYDRO] No new wind particles", flush=True)

        self.channels['gas_to_hydro'].copy()


        # HYDRODYNAMICAL STEP
        if self.verbosity:
            start = time.time()

        self.hydro_code.evolve_model( self.model_time + dt )

        self.hydro_code.update_particle_set()
        self.hydro_code.gas_particles.synchronize_to(self.gas_particles)
        self.hydro_code.dm_particles.synchronize_to(self.sink_particles)

        self.channels['hydro_to_gas'].copy()
        self.channels['hydro_to_sink'].copy()


        while self.sf_detection.is_set():

            self.form_sinks()

            self.hydro_code.evolve_model( self.model_time + dt )

            self.hydro_code.update_particle_set()
            self.hydro_code.gas_particles.synchronize_to(self.gas_particles)
            self.hydro_code.dm_particles.synchronize_to(self.sink_particles)

            self.channels['hydro_to_gas'].copy()
            self.channels['hydro_to_sink'].copy()


        if self.verbosity:
            print ("[HYDRO] Most dense particle at {a} SF density".format(
                a=self.gas_particles.rho.max()/self.density_threshold), flush=True)


        # SINK HANDLING AND ACCRETION
        self.merge_sinks()

        self.accrete_sinks()

        if self.verbosity:
            end = time.time()
            print ("[HYDRO] Hydro ran {b} Myr in {a} s".format(a=end-start, 
                b=dt.value_in(units.Myr)), flush=True)  


        if self.cooling is not None:
            self.cooling.evolve_for( dt/2. )

        # Add new stellar wind particles
        if self.stellar_wind_code is not None and len(self.stellar_wind_code.particles) > 0:
            self.stellar_wind_code.evolve_model( end_time )
            if len(self.gas_particles) != len(self.hydro_code.gas_particles):
                print ("[HYDRO] {a} new stellar wind particle(s)".format(
                    a=len(self.gas_particles)-len(self.hydro_code.gas_particles)), flush=True)
                self.gas_particles.synchronize_to(self.hydro_code.gas_particles)
                print ("[HYDRO] Now {a} in hydro gas".format(
                    a=len(self.hydro_code.gas_particles)), flush=True)

        # Remove ejected sinks as they can crash the code
        # Assumption is that box_size is large, so half that is still very far
        # away!
        if len(self.sink_particles) > 0:
            self.sink_particles.remove_particles(self.sink_particles.select_array(
                lambda r: r.lengths() > self.box_size/2., ['position']))
            if len(self.sink_particles) != len(self.hydro_code.dm_particles):
                self.sink_particles.synchronize_to(self.hydro_code.dm_particles)
                print ("[HYDRO] Sink(s) have been ejected!", flush=True)

        self.model_time = end_time


    def form_sinks (self):
        # Convert overdense particles into sinks

        gas_to_sinkify = self.gas_particles.select_array(
            lambda rho: rho > self.density_threshold, ['rho']).copy()

        if len(gas_to_sinkify) > 0:

            print ("[HYDRO] Forming {a} sink(s)".format(a=len(gas_to_sinkify)), flush=True)

            self.gas_particles.remove_particles(gas_to_sinkify)
            self.hydro_code.gas_particles.remove_particles(gas_to_sinkify)

            gas_to_sinkify.time_born = self.model_time
            gas_to_sinkify.Lx = 0 | (units.g * units.m ** 2) / units.s
            gas_to_sinkify.Ly = 0 | (units.g * units.m ** 2) / units.s
            gas_to_sinkify.Lz = 0 | (units.g * units.m ** 2) / units.s

            #gas_to_sinkify.tff = (constants.G * gas_to_sinkify.rho)**-0.5
            gas_to_sinkify.tff = (constants.G * gas_to_sinkify.mass/(4.*np.pi/3.*gas_to_sinkify.radius**3))**-0.5
            gas_to_sinkify.sf_threshold = self.hydro_code.model_time# + gas_to_sinkify.tff * np.exp(-self.hydro_code.model_time/self.delay_time)

            self.hydro_code.dm_particles.add_particles(gas_to_sinkify)
            self.sink_particles.add_particles(gas_to_sinkify)


    def merge_sinks (self):
        # Merge sinks that are too close

        if len(self.sink_particles) > 1 and self.sink_particles.radius.min() > (0|units.AU):

            sinks_to_merge = self.sink_particles.copy().connected_components(
                threshold=self.merge_radius)

            for sinks in sinks_to_merge:
                if len(sinks) > 1:
                    print ("[HYDRO] Merging {a} sinks".format(a=len(sinks)), flush=True)

                    new_sink = self.merge_n_sinks(sinks)

                    self.hydro_code.dm_particles.remove_particles(sinks)
                    self.hydro_code.dm_particles.add_particle(new_sink)

                    self.sink_particles.remove_particles(sinks)
                    self.sink_particles.add_particle(new_sink)


    def merge_n_sinks (self, sinks):
        # Merge a number of sinks

        com_pos = sinks.center_of_mass()
        com_vel = sinks.center_of_mass_velocity()

        new_sink = Particles(1)

        new_sink.time_born = sinks.time_born.min()
        new_sink.mass = sinks.mass.sum()
        new_sink.position = com_pos
        new_sink.velocity = com_vel
        new_sink.radius = sinks.radius.max()

        new_sink.Lx = sinks.Lx.sum()
        new_sink.Ly = sinks.Ly.sum()
        new_sink.Lz = sinks.Lz.sum()

        #new_sink.rho = new_sink.mass / (4./3.*np.pi * new_sink.radius**3.)

        new_sink.tff = sinks.tff.min()#(constants.G * new_sink.rho)**-0.5
        new_sink.sf_threshold = self.hydro_code.model_time

        return new_sink


    def accrete_sinks (self):
        # Let sinks accrete gas

        if len(self.sink_particles):

            sinks = new_sink_particles(self.sink_particles)

            sinks.accrete(self.gas_particles)

            for i in range(len(self.sink_particles)):
                self.sink_particles.Lx += sinks[i].angular_momentum[0]
                self.sink_particles.Ly += sinks[i].angular_momentum[1]
                self.sink_particles.Lz += sinks[i].angular_momentum[2]

            print ("[HYDRO] {a} sink(s) have accreted {b} gas particles".format(
                a=len(self.sink_particles), 
                b=len(self.hydro_code.gas_particles)-len(self.gas_particles)), flush=True)

            self.gas_particles.synchronize_to(self.hydro_code.gas_particles)

            self.channels['sink_to_hydro'].copy()


    def stop (self):
        self.hydro_code.stop()


    def save_particle_set (self, filepath, index):

        if len(self.gas_particles):
            filename = "{a}/hydro_gas_particles_i{b:05}.hdf5".format(a=filepath, b=index)
            write_set_to_file(self.gas_particles, filename, "hdf5", 
                timestamp=self.model_time, append_to_file=False, overwrite_file=True)

        if len(self.sink_particles):
            filename = "{a}/hydro_sink_particles_i{b:05}.hdf5".format(a=filepath, b=index)
            write_set_to_file(self.sink_particles, filename, "hdf5",
                timestamp=self.model_time, append_to_file=False, overwrite_file=True)


    def print_diagnostics (self):

        print ("[HYDRO] Current class time: {a} Myr".format(
            a=self.model_time.value_in(units.Myr)))
        print ("[HYDRO] Hydro code time: {a} Myr".format(
            a=self.hydro_code.model_time.value_in(units.Myr)))
        print ("[HYDRO] #Gas in class: {a}; #Gas in hydro code: {b}".format(
            a=len(self.gas_particles), b=len(self.hydro_code.gas_particles)))
        print ("[HYDRO] #Sinks in class: {a}; #Sinks in hydro code: {b}".format(
            a=len(self.sink_particles), b=len(self.hydro_code.dm_particles)))
        print ("[HYDRO] Gas mass: {a} MSun; Sink mass: {b} MSun".format(
            a=self.gas_particles.mass.sum().value_in(units.MSun),
            b=self.sink_particles.mass.sum().value_in(units.MSun)))
        print ("[HYDRO] Gas center of mass: {a} pc".format(
            a=self.gas_particles.center_of_mass().value_in(units.parsec)))
        print ("[HYDRO] Gas center of mass velocity: {a} pc/Myr".format(
            a=self.gas_particles.center_of_mass_velocity().value_in(units.parsec/units.Myr)))
        print ("[HYDRO] Sink center of mass: {a} pc".format(
            a=self.sink_particles.center_of_mass().value_in(units.parsec)))
        print ("[HYDRO] Sink center of mass velocity: {a} pc/Myr".format(
            a=self.sink_particles.center_of_mass_velocity().value_in(units.parsec/units.Myr)),
            flush=True)
