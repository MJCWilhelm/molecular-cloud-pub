import numpy as np
import os
import pickle
import argparse
import time

from amuse.units import units, nbody_system, constants
from amuse.datamodel import Particles
from amuse.couple.bridge import Bridge
from amuse.io import read_set_from_file
from amuse.ic.brokenimf import MultiplePartIMF

from hydro_class import Hydro
from gravity_class import Gravity
from ppd_population import PPD_population, restart_population

import mass_functions as mf


kroupa_imf = MultiplePartIMF(mass_boundaries=[0.08, 0.5, 150.]|units.MSun,
    alphas=[-1.3, -2.3])


def make_stars_from_sinks (sinks, time, star_counter, IMF_list, spatial='uniform'):

    stars = Particles()

    mass_from_sink = IMF_list[star_counter] * \
        (1. + 0.1*(IMF_list[star_counter] < 1.9|units.MSun))

    for sink in sinks:

        if mass_from_sink < sink.mass and sink.sf_threshold < time:

            new_star = Particles(1)

            new_star.mass = IMF_list[star_counter]

            if spatial == 'plummer':
                converter = nbody_system.nbody_to_si(1.|units.MSun, sink.radius)
                temp_star = new_plummer_model(1, converter)
                new_star.position = sink.position + temp_star.position
            elif spatial == 'gaussian':
                dR = np.random.normal(size=3, scale=sink.radius.value_in(units.parsec))
                new_star.position = sink.position + (dR | units.parsec)
            else:
                dR = np.random.uniform(size=3, low=-1., high=1.) * sink.radius
                new_star.position = sink.position + dR

            new_star.velocity = sink.velocity
            new_star.time_born = time
            stars.add_particles(new_star)


            sink.mass -= mass_from_sink
            sink.sf_threshold = time + sink.tff * np.exp(-time/sys_params['delay_time'])

            star_counter += 1
            # Generate more stellar masses if not enough
            if star_counter >= len(IMF_list):
                IMF_list.extend( kroupa_imf.next_mass(1000) )

            mass_from_sink = IMF_list[star_counter] * \
                (1. + 0.1*(IMF_list[star_counter] < 1.9|units.MSun))
                
    if len(sinks) > 0 and len(stars) == 0:
        print ("[MAIN] No stars formed. Most massive sink: {a} / {b} MSun".format(
            a=sinks.mass.max().value_in(units.MSun),
            b=mass_from_sink.value_in(units.MSun)),
            flush=True)
        if len(sinks.select_array(lambda mass: mass > mass_from_sink, ['mass'])):
            print ("[MAIN] Next formation: {a} Myr".format(a=sinks.select_array(lambda mass: mass > mass_from_sink, ['mass']).sf_threshold.min().value_in(units.Myr)), flush=True)

    return stars, star_counter


def run_simulation (sys_params, gas_particles=None, sink_particles=None, star_particles=None):

    if gas_particles is None and sink_particles is None and star_particles is None:
        print ("[MAIN] No particles, canceling simulation", flush=True)
        return

    target_stellar_mass = sys_params['sfe'] * read_set_from_file(
        sys_params['read_folder']+'hydro_gas_particles_i00000.hdf5', 'hdf5').mass.sum()

    if star_particles is None:
        stage = 1
    elif star_particles.initial_mass.sum() < target_stellar_mass:
        stage = 2
    else:
        stage = 3

    # Current time is that of particle set, or 0
    if gas_particles is not None and hasattr(gas_particles, "get_timestamp"):
        current_time = gas_particles.get_timestamp()
    elif star_particles is not None and hasattr(star_particles, "age"):
        current_time = star_particles.get_timestamp()
    else:
        current_time = 0.|units.Myr

    if stage > 1:
        bridge_offset_time = current_time
    if stage == 3:
        end_time = star_particles.time_born.max() + sys_params['continue_time']
            

    time_of_next_snapshot = current_time + sys_params['dt_diag']
    snapshot_counter = sys_params['read_index']+1


    hydro = None
    gravity = None
    gravhydro = None
    ppd_population = None


    # Pregenerate list of stellar masses
    if star_particles is not None:
        star_counter = len(star_particles)
    else:
        star_counter = 0
    IMF_list = kroupa_imf.next_mass(10000)

    # Add particles
    if stage < 3 and (gas_particles is not None or sink_particles is not None):
        hydro = Hydro(sys_params, begin_time=current_time)
        if gas_particles is not None:
            hydro.add_gas_particles(gas_particles)
        if sink_particles is not None:
            hydro.add_sink_particles(sink_particles)
    if star_particles is not None:
        if hasattr(star_particles,'age') and star_particles.age.max()>0.|units.Myr:
            # Restart
            ppd_population = restart_population(
                sys_params['read_folder'], sys_params['read_index'], 
                sys_params['alpha'], 2.33, sys_params['number_of_disk_cells'], sys_params['r_min'],
                sys_params['r_max'], fried_folder=sys_params['fried_dir'], 
                number_of_workers=sys_params['number_of_vaders'])
        else:
            # Initial conditions have stars
            ppd_population = PPD_population(
                alpha=sys_params['alpha'], 
                number_of_cells=sys_params['number_of_disk_cells'],
                number_of_workers=sys_params['number_of_vaders'],
                r_min=sys_params['r_min'],
                r_max=sys_params['r_max'],
                begin_time=0.|units.Myr,
                fried_folder=sys_params['fried_dir'])
            ppd_population.add_star_particles(star_particles)

        gravity = Gravity(sys_params, stellar_evolution=True, begin_time=current_time,
            ppd_population=ppd_population)
        # if there are evolved stars, add as such
        if hasattr(star_particles,'age') and star_particles.age.max()>0.|units.Myr:
            gravity.add_old_star_particles(star_particles)
            gravity.channels['star_to_ppd'].copy_attributes(['mass', 'fuv_luminosity'])
        else:
            gravity.add_new_star_particles(star_particles)
            gravity.channels['star_to_ppd'].copy_attributes(['fuv_luminosity'])


    if hydro is not None and gravity is not None:
        gravhydro = Bridge(use_threading=False)
        gravhydro.add_system(gravity, (hydro,))
        gravhydro.add_system(hydro, (gravity,))

        gravhydro.timestep = sys_params['dt_bridge']


    while True:

        # Evolve system
        print ("[MAIN] Main time: {a} Myr".format(
            a=current_time.value_in(units.Myr)), flush=True)

        if ppd_population is not None:
            ppd_population.evolve_model( current_time + sys_params['dt_main']/2. )

        current_time += sys_params['dt_main']

        if stage == 1:
            if sys_params['verbosity']:
                print ("[MAIN] Hydro class time: {a} Myr".format(
                    a=hydro.model_time.value_in(units.Myr)), flush=True)
                print ("[MAIN] Hydro code time: {a} Myr".format(
                    a=hydro.hydro_code.model_time.value_in(units.Myr)), flush=True)
                print ("[MAIN] Evolve hydro for {a} Myr".format(
                    a=sys_params['dt_main'].value_in(units.Myr)), flush=True)

            hydro.evolve_model(current_time)

        elif stage == 2:
            if sys_params['verbosity']:
                print ("[MAIN] Hydro class time: {a} Myr".format(
                    a=hydro.model_time.value_in(units.Myr)), flush=True)
                print ("[MAIN] Hydro code time: {b} Myr".format(
                    b=hydro.hydro_code.model_time.value_in(units.Myr)), flush=True)
                print ("[MAIN] Gravity class time: {a} Myr".format(
                    a=gravity.model_time.value_in(units.Myr)), flush=True)
                print ("[MAIN] Gravity code time: {b} Myr".format(
                    b=gravity.grav_code.model_time.value_in(units.Myr)), flush=True)
                print ("[MAIN] Evolve gravhydro for {a} Myr".format(
                    a=sys_params['dt_main'].value_in(units.Myr)), flush=True)

            gravhydro.evolve_model(current_time - bridge_offset_time)

        else:
            if current_time > end_time:
                print ("[MAIN] Reached end time, aborting", flush=True)
                break

            if sys_params['verbosity']:
                print ("[MAIN] Gravity class time: {a} Myr".format(
                    a=gravity.model_time.value_in(units.Myr)), flush=True)
                print ("[MAIN] Gravity code time: {b} Myr".format(
                    b=gravity.grav_code.model_time.value_in(units.Myr)), flush=True)
                print ("[MAIN] Evolve gravity for {a} Myr".format(
                    a=sys_params['dt_main'].value_in(units.Myr)), flush=True)

            gravity.evolve_model(current_time)


        if ppd_population is not None:
            ppd_population.evolve_model( current_time )


        if sys_params['star_formation'] and stage < 3:
            # Make and add stars
            new_stars, star_counter = make_stars_from_sinks(hydro.sink_particles, current_time,
                star_counter, IMF_list)
            hydro.channels['sink_to_hydro'].copy()

            if len(new_stars):
                if stage == 1: # If these are the first stars, initialize gravity
                    print ("[MAIN] First stars have formed at {a} Myr, initiating gravity".format(
                        a=current_time.value_in(units.Myr)), flush=True)

                    ppd_population = PPD_population(
                        alpha=sys_params['alpha'], 
                        number_of_cells=sys_params['number_of_disk_cells'],
                        number_of_workers=sys_params['number_of_vaders'],
                        r_min=sys_params['r_min'],
                        r_max=sys_params['r_max'],
                        begin_time=current_time,
                        fried_folder=sys_params['fried_dir'])

                    gravity = Gravity(sys_params, begin_time=current_time,
                        stellar_evolution=True, ppd_population=ppd_population)
                    if sys_params['stellar_wind']:
                        gravity.channels['star_to_ext'] = \
                      gravity.star_particles.new_channel_to(hydro.stellar_wind_code.particles)

                    gravhydro = Bridge(use_threading=False)
                    gravhydro.add_system(gravity, (hydro,))
                    gravhydro.add_system(hydro, (gravity,))

                    gravhydro.timestep = sys_params['dt_bridge']
                    bridge_offset_time = current_time

                    stage += 1

                if sys_params['verbosity']:
                    print ("[MAIN] Adding {a} stars, with masses:".format(a=len(new_stars)),
                        new_stars.mass.value_in(units.MSun), "MSun", flush=True)

                gravity.add_new_star_particles(new_stars)
                if sys_params['stellar_wind']:
                    gravity.star_particles.synchronize_to(hydro.stellar_wind_code.particles)
                ppd_population.add_star_particles(new_stars)
                gravity.channels['star_to_ppd'].copy_attributes(['mass', 'fuv_luminosity'])

                if stage == 2 and \
                        gravity.star_particles.initial_mass.sum() > target_stellar_mass:
                    print ("[MAIN] Target SFE reached, removing gas", flush=True)
                    hydro.stop()
                    hydro = None
                    stage += 1
                    end_time = current_time + sys_params['continue_time']


        # Write particles
        if current_time >= time_of_next_snapshot:
            if hydro is not None:
                print ("[MAIN] Saving gas particles", flush=True)
                hydro.save_particle_set(sys_params['write_folder'], snapshot_counter)
            if gravity is not None:
                print ("[MAIN] Saving star particles", flush=True)
                gravity.save_particle_set(sys_params['write_folder'], snapshot_counter)
            if ppd_population is not None:
                print ("[MAIN] Saving viscous particles and grids", flush=True)
                ppd_population.output_counter = snapshot_counter
                ppd_population.write_particles(filepath=sys_params['write_folder'])
                ppd_population.write_grids(filepath=sys_params['write_folder'])

            snapshot_counter += 1
            time_of_next_snapshot += sys_params['dt_diag']

    if gravity is not None:
        gravity.stop()
    if hydro is not None:
        hydro.stop()


def main (sys_params):

    # Read initial conditions
    if os.path.isfile("{a}/hydro_gas_particles_i{b:05}.hdf5".format(
            a=sys_params['read_folder'], b=sys_params['read_index'])):
        gas_particles = read_set_from_file(
            "{a}/hydro_gas_particles_i{b:05}.hdf5".format(
                a=sys_params['read_folder'], b=sys_params['read_index']), 'hdf5')
    else:
        gas_particles = None

    if os.path.isfile("{a}/hydro_sink_particles_i{b:05}.hdf5".format(
            a=sys_params['read_folder'], b=sys_params['read_index'])):
        sink_particles = read_set_from_file(
            "{a}/hydro_sink_particles_i{b:05}.hdf5".format(
                a=sys_params['read_folder'], b=sys_params['read_index']), 'hdf5')
    else:
        sink_particles = None

    if os.path.isfile("{a}/grav_star_particles_i{b:05}.hdf5".format(
            a=sys_params['read_folder'], b=sys_params['read_index'])):
        star_particles = read_set_from_file(
            "{a}/grav_star_particles_i{b:05}.hdf5".format(
                a=sys_params['read_folder'], b=sys_params['read_index']), 'hdf5')
    else:
        star_particles = None


    # Write simulation parameters
    param_dump = open(sys_params['write_folder']+'_sys_params.pickle', 'wb')
    pickle.dump(sys_params, param_dump)
    param_dump.close()


    if sys_params['verbosity']:
        if gas_particles is not None:
            N_gas = len(gas_particles)
            if hasattr(gas_particles, "get_timestamp"):
                begin_time = gas_particles.get_timestamp()
            else:
                begin_time = 0.|units.Myr
        else:
            N_gas = 0
            begin_time = 0.|units.Myr

        if sink_particles is not None:
            N_sinks = len(sink_particles)
        else:
            N_sinks = 0

        if star_particles is not None:
            N_stars = len(star_particles)
        else:
            N_stars = 0

        print ("[MAIN] Starting simulation at {a} Myr".format(
            a=begin_time.value_in(units.Myr)), flush=True)
        print ("[MAIN] {a} gas, {b} sinks, {c} stars".format(
            a=N_gas, b=N_sinks, c=N_stars), flush=True)


    # Run simulation
    run_simulation (sys_params, gas_particles, sink_particles, star_particles)


if __name__ == '__main__':

    sys_params = {
        # Timesteps
        'dt_main': 1. | units.kyr,     # Main timestep of loop (star formation, rad/hydro coupling)
        'dt_hydro': 0.01 | units.kyr,     # Internal timestep of hydro code
        'dt_bridge': 1. | units.kyr,   # Bridge timestep between hydro and gravity
        #'dt_wind': 0.1 | units.kyr,      # Stellar wind code timestep
        'dt_diag': 10. | units.kyr,     # Snapshot timescale

        # Simulation domain
        'box_size': 1e6 | units.pc,     # The size of the simulation domain

        # Files to use
        'read_folder': '../output/',
        'read_index': 0,                # Index of data file to read
        'write_folder': '../output/', # Folder of data to write

        'verbosity': 1,                 # Print diagnostics or not

        # Sink settings
        'density_threshold':(1.|units.MSun)/(0.05|units.parsec)**3, # Density at which gas turns into a sink
        'merge_radius': 0.01 | units.pc,# Radius at which sinks merge
        'gas_epsilon': 0.05 | units.pc,

        # Star formation settings
        'sfe': 0.3,
        'continue_time': 2. | units.Myr,
        'delay_time': 1. | units.Myr,

        # Wind settings
        #'M_wind_particles': 1e-5 | units.MSun,   # Stellar wind particle mass

        'seed': 439206237,

        # Mode flags
        'star_formation': True,         # Form stars from sinks
        'stellar_wind': False,           # Do stellar wind feedback from massive stars
        'cooling': True,                # Use thermal model

        # PPD settings
        'fried_dir': '../data/',
        'r_min': 0.05 | units.AU,
        'r_max': 2000. | units.AU,
        'alpha': 5e-3,
        'number_of_vaders': 5,
        'number_of_disk_cells': 100,
    }


    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='directory', type=str, default='')
    parser.add_argument('-s', dest='seed', type=int, default=-1)
    parser.add_argument('-i', dest='read_index', type=int, default=-1)
    parser.add_argument('-t', dest='delay_time', type=float, default=-1.)
    parser.add_argument('-e', dest='sfe', type=float, default=-1.)
    args = parser.parse_args()

    if args.directory != '':
        sys_params['read_folder'] = args.directory
        sys_params['write_folder'] = args.directory

    if args.seed > 0:
        sys_params['seed'] = args.seed

    if args.read_index >= 0:
        sys_params['read_index'] = args.read_index

    if args.delay_time > 0.:
        sys_params['delay_time'] = args.delay_time | units.Myr

    if args.sfe > 0.:
        sys_params['sfe'] = args.sfe


    np.random.seed(sys_params['seed'])

    main(sys_params)
