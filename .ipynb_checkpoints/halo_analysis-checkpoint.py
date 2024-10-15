'''

'''
#-----------------------------------------------------------------------------------
# Import libraries.
#-----------------------------------------------------------------------------------
from astropy.cosmology import FlatLambdaCDM
from astropy.modeling.physical_models import NFW
import h5py
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.signal import argrelextrema
import sys
import struct
import time
#-----------------------------------------------------------------------------------
# Import local modules/libraries/scripts.
#-----------------------------------------------------------------------------------
import halo_utilities
import utilities
#-----------------------------------------------------------------------------------
# Classes
#-----------------------------------------------------------------------------------
'''
Halo dictionary-class properties:
    ***** Need to make sure if distance units are in over-h or non-h units!
    "ID.halo.infall": halo ID at the infall snapshot
        - This will often be used as the halo ID (hID).
        - There could be duplicate hIDs across snapshots.
        - But they are unique within a snapshot, so always use them together.
        - And use the treeID where available as it is unique within the simulation.
    "snapshot.infall": infall snapshot number
    "file.path": full directory/file name for the current halo's particle data
    "coordinates": 3-D positions of particles in simulation units [kpc comoving]
        - Each element contains 3-D coordinates for all of halo's particles within a single snapshot.
    "velocities": 3-D velocities of particles in simulation units [km/s]
        - Each element contains 3-D velocities for all of halo's particles within a single snapshot.
    "ID.particle": IDs of particles
    "snapshot.numbers": snapshot numbers from infall to the end of simulation
    
'''
class halo(dict):
    '''
    * A dictionary-class for handling the data for a single halo.
    '''
    def __init__(self, hID, infall_snapshot, fname, simulation_snapnum_dict, BH_parameters):  # , hID, infall_snap
        '''
        * Method to assign instance attributes.
        - ID.halo.infall: halo ID at the infall snapshot - this will be the halo ID.
          ***** REMEMBER *****
            - There could be duplicate ID.halo.infall values across snapshots.
            - But they are unique within a snapshot, so always use them together.
        - snapshot.infall: infall snapshot number
        - file.path: full directory/file name for the particle data.
        '''
        self["ID.halo.infall"] = hID
        self["snapshot.infall"] = infall_snapshot
        self["file.path"] = fname
        self["simulation.snapshot.number.dict"] = simulation_snapnum_dict
        self["h"] = BH_parameters['h']
        self["most.bound.fraction"] = BH_parameters['most_bound_frac']
        self["most.bound.number.min"] = BH_parameters['most_bound_min']
        self["most.bound.number.max"] = BH_parameters['most_bound_max']
        self["particle.mass"] = BH_parameters['part_mass'] # Assumes equal mass particles, so just one number.
        self["cv.rapid.drop.fraction"] = BH_parameters['cv_rapid_drop_frac']
        self["cv.infall.drop.fraction"] = BH_parameters['cv_infall_frac']
        self["cv.stays.low.fraction"] = BH_parameters['cv_stays_low_frac']
        # Initialize the halo data.
        # I could use individual methods here, but I decided to use a wrapper method "initialize_halo".
        self.initialize_halo()
    #
    def initialize_halo(self):
        '''
        * This function initializes the halo object by setting the particle data.
        '''
        # Set the halo particle data.
        self.set_particle_data()
        # Determine number of most bound particles to use.
        self.most_bound_particle_number()
        # Compute and store the center of mass of halo particles at each snapshot.
        # These initial COMs will be replaced later with more accurate ones.
        self.compute_com()
    #
    def set_particle_data(self):
        '''
        * Method to read in the particle data for the current halo and set the particle data as attributes for the current halo object.
        - utilities.open_halo_particles_file reads in the coordinates in over-h units.
        - So, convert it to non-h units here.
        - Coordinates and velocities will be stored in co-moving units.
        '''
        # Use function "open_halo_particles_file" from utilities.py to read in halo particles.
        # "open_halo_particles_file" needs the full file path.
        snapnums, coordinates, velocities, particleIDs = utilities.open_halo_particles_file(self["file.path"])
        # Assign the particle data as attributes.
        self["coordinates"] = coordinates / self["h"] # ckpc
        self["velocities"] = velocities # km/s / sqrt(a), co-moving
        self["ID.particle"] = particleIDs
        self["snapshot.numbers"] = snapnums
        self["number.of.particles"] = len(self["ID.particle"])
        # Assign scale factors and redshifts corresponding to the snapshot numbers.
        intersecting_snapnums, intersecting_idx_sim, intersecting_idx_halo = np.intersect1d(self["simulation.snapshot.number.dict"]["snapshot_numbers"], self["snapshot.numbers"], assume_unique=True, return_indices=True)
        self["scale.factor"] = self["simulation.snapshot.number.dict"]["scale_factors"][intersecting_idx_sim]
        self["redshift"] = self["simulation.snapshot.number.dict"]["redshifts"][intersecting_idx_sim]
    #
    def most_bound_particle_number(self):
        '''
        * Method to determine the number of most bound particles to use.
        - Cases:
            1. num_total < most_bound_number_min: num_most_bound = num_total
            2. num_most_bound < most_bound_number_min: num_most_bound = most_bound_number_min
            3. most_bound_number_min <= num_most_bound <= most_bound_number_max: num_most_bound = num_part * most_bound_fraction
            4. num_most_bound > most_bound_number_max: num_most_bound = most_bound_number_max
        '''
        # Multiply the number of particles by the most bound fraction parameter.
        num_most_bound = int(self["number.of.particles"] * self["most.bound.fraction"])
        if num_most_bound < self["most.bound.number.min"]:
            # When num_most_bound is smaller than the minimum most bound particle number limit:
            if self["number.of.particles"] < self["most.bound.number.min"]:
                # If the total number of halo particles is smaller than the minimum most bound particle number limit, use all of the particles.
                num_most_bound = self["number.of.particles"]
            else:
                # If the total number of halo particles is equal to or greater than the minimum most bound particle number limit, use the limit itself.
                num_most_bound = self["most.bound.number.min"]
        elif num_most_bound > self["most.bound.number.max"]:
            num_most_bound = self["most.bound.number.max"]
        # Set the computed number as an attribute of the halo object.
        self["most.bound.number"] = int(num_most_bound)
    #
    def compute_com(self):
        '''
        * Method to compute the center of mass of the particles recursively.
            - All snapshots
            - in comoving kpc
        '''
        com_list = []
        #
        for i in range(len(self["snapshot.numbers"])):
            # Compute the COM for each snapshot, using the cm function from halo_utilities.py.
            # I think: num_part is the minimum number of particles required to do the computation iteratively. If the halo has fewer particles than num_part, COM will be computed using all particles directly.
            # I think: nel_lim sets down to how many particles the iteration will be done.
            com = halo_utilities.cm(self["coordinates"][i], nofile=True, num_part=self["most.bound.number.min"], nel_lim=int(self["most.bound.number.min"] * 0.9), print_statement=False)
            # Append the result to the result list.
            com_list.append(com)
        # Set the center array as an attribute.
        self["com"] = np.array(com_list)
    #
    def compute_particle_energies_at_infall(self):
        '''
        * Method to compute kinetic, potential, and binding energies of halo particles at the halo's infall snapshot.
        - All computations are done in physical units.
        - Cosmological simulations using GIZMO use co-moving units.
        - To convert them to physical units:
            - length_physical = length_comoving * scale_factor
            - velocity_physical = velocity_comoving * sqrt(scale_factor)
        - Velocity units do not have a factor of h.

        - Particle energies at the infall snapshot are computed mainly to find the n% most bound particles at the infall snapshot. n will usually be small with the default value 2. For a massive halo, even 2% may be a large number and particle energy calculations for a large number of particles could be time consuming. So this method sets a minimum and maximum numbers of (inner-most) particles to compute the energies for.
        * Cases:
            - think about it!
        '''
        '''
compute_particle_energies_at_infall(current_halo.dmo_com[0], current_halo.dmo_coords[0], current_halo.dmo_vels[0],   current_halo.scale_factors[0], part_mass)
        '''
        scale_at_infall = self["scale.factor"][0]
        com_at_infall = self["com"][0] * scale_at_infall # non-h pkpc
        particle_coords_at_infall = self["coordinates"][0] * scale_at_infall # non-h pkpc
        particle_vels_at_infall = self["velocities"][0] * np.sqrt(scale_at_infall) # physical
        particle_mass = self["particle.mass"]
        # Number of particles to compute energies for.
        # Assuming most of the most bound particles are in the inner regions, use up to 10x more particles than the number of most bound particles used.
        # Default most.bound.number should be capped at 500, so the maximum value of n_use should be 5000.
        # n_use will be used to slice arrays that are sorted by the particle distance, so if the total number of particles is below n_use, the total number of particles will be used.
        n_use = int(self["most.bound.number"] * 10)
        # Use the function compute_particle_energies to compute the binding energies.
        _, _, self["particle.binding.energies.infall"], self["distance.sort.idx.infall"] = compute_particle_energies(com_at_infall, particle_coords_at_infall, particle_vels_at_infall, n_use, particle_mass)
    #
    def most_bound_particle_com(self):
        '''
        * This method uses the N% most bound particle subset at the infall snapshot to compute more accurate COMs for the halo at all following snapshots.
        - Note:
            - 
        '''
        # List to store the COM results.
        new_com_list = []
        # Particle coordinate data for all snapshots.
        coords_arr = self["coordinates"]
        # Argsort the infall snapshot binding energy array.
        bind_energy_sort_idx = np.argsort(self["particle.binding.energies.infall"])
        # Get N% most bound particles at infall. The exact number is already computed and stored as self["most.bound.number"].
        num_part = self["most.bound.number"]
        # Get indices of particles to use.
        most_bound_sort_idx = bind_energy_sort_idx[:num_part]
        track_idx = self["distance.sort.idx.infall"][most_bound_sort_idx]
        # Store the indices that give the most bound particles at infall.
        self["most.bound.indices"] = track_idx
        # Compute new COMs using most bound particles at infall.
        for i in range(len(coords_arr)):
            # Coordinates of the most bound particles at infall at the ith snapshot.
            current_snap_coords = coords_arr[i][track_idx]
            # Compute COM.
            com = halo_utilities.cm(current_snap_coords, nofile=True, num_part=self["most.bound.number.min"], nel_lim=int(self["most.bound.number.min"] * 0.9), print_statement=False)
            # Append the COM result to the result list.
            new_com_list.append(com)
        # Store the COM result.
        self["com.most.bound"] = np.array(new_com_list)
    #
    def compute_halo_properties(self, host_main_branch_dict):
        '''
        * Method to compute various halo properties.
        * Properties:
            Vmax -
            Rmax -
            cv -
            Rbound - 
            halo_velocity - 
            distance.from.host - 
            scale.disrupt
        '''
        scales = self["scale.factor"]
        # Create empty arrays to store results.
        #vcirc_results = np.zeros(len(scales), dtype=object)
        #sorted_pdist_results = np.zeros(len(scales), dtype=object)
        #density_results = np.zeros(len(scales), dtype=object)
        #bin_mid_results = np.zeros(len(scales), dtype=object)
        #rho_r_squared_results = np.zeros(len(scales), dtype=object)
        #local_min_idx_results_all = np.zeros(len(scales), dtype=object)
        vmax_array = np.zeros(len(scales))
        rmax_array = np.zeros(len(scales))
        cv_array = np.zeros(len(scales))
        rboundary_array = np.zeros(len(scales))
        halo_velocity_list = []
        # Loop through each snapshot.
        print(self["ID.halo.infall"])
        for i in range(len(scales)):
            current_scale = scales[i]
            current_com = self["com.most.bound"][i] * current_scale # pkpc
            current_coords = self["coordinates"][i] * current_scale # pkpc
            current_velocities = self["velocities"][i] * np.sqrt(current_scale) # physical
            # Use the function vcirc_particle_single_halo() from halo_utilities.py to compute Vcirc.
            vcirc_arr, sorted_pdist_arr, sort_idx = halo_utilities.vcirc_particle_single_halo(current_com, current_coords)
            # Use the function compute_density_profile_sorted_pdist() from halo_utilities.py to compute the density profile.
            # The first few bins often contain very few particles if the first bin starts from the first index, so start from a few indices back.
            num_part = len(sorted_pdist_arr)
            if num_part >= 100:
                first_idx = 10
            elif num_part < 100:
                first_idx = int(np.rint(0.1 * num_part))
            ### The last few particles often are outliers, so don't use them: this criterion has been removed.
            last_idx = -1
            density_arr, bin_mid_arr, num_enclosed_within_mid_arr = halo_utilities.compute_density_profile_sorted_pdist(sorted_pdist_arr, self["particle.mass"], first_idx, last_idx)
            # Compute rho(r) * r^2: this will be used to find the boundary of the halo.
            rho_r_squared = density_arr * bin_mid_arr * bin_mid_arr
            # Use the function concentration_one_halo() from halo_utilities.py to compute Vmax, Rmax, and cv, using the Vcirc profile just computed.
            ### R_boundary at infall: the distance that the density profile was computed out to. last_idx is currently set as -1. This part is removed.
            # r_boundary at infall: use the third to last particle's distance.
            if i == 0:
                r_boundary = sorted_pdist_arr[-3]
                #r_boundary = bin_mid_arr[-1]
                # Save some of the properties at infall.
                r_boundary_infall = r_boundary
                infall_min_density = np.min(density_arr)
                # At the infall snapshot, simply compute Vmax, rmax, and cv using all particles.
                cv, vmax, rmax = halo_utilities.concentration_one_halo(vcirc_arr, sorted_pdist_arr)
                # Compute the halo velocity using 10% inner-most particles.
                inner_most_frac = 0.1
                vels_sorted_by_distance = current_velocities[sort_idx]
                halo_velocity = compute_halo_velocity(inner_most_frac, vels_sorted_by_distance)
                #
                local_min_idx = [None]
            else:
                '''
                * Finding the turn-over point of the rho * r^2 profile to find the "edge" of the subhalo.
                '''
                r_boundary = find_r_boundary(density_arr, bin_mid_arr, sorted_pdist_arr, sort_idx, last_idx, r_boundary_infall, r_boundary_prev, infall_min_density, previous_min_density, current_scale, previous_scale, scales[0])
                # Compute Vmax, Rmax, and cv using only particles within r_boundary.
                slice_idx = np.searchsorted(sorted_pdist_arr, r_boundary)
                if slice_idx < 10:
                    # Anything that has a small slice_idx value probably are not to be trusted. But that "small" number also probably is much much larger than 10. But here, just use 10 so the code doesn't break.
                    # I think it will be really easy and obvious to tell which subhalos to remove from the analysis anyway.
                    slice_idx = 10
                cv, vmax, rmax = halo_utilities.concentration_one_halo(vcirc_arr[:slice_idx], sorted_pdist_arr[:slice_idx])
                # Compute the halo velocity using 10% inner-most particles.
                inner_most_frac = 0.1
                vels_sorted_by_distance = current_velocities[sort_idx[:slice_idx]]
                halo_velocity = compute_halo_velocity(inner_most_frac, vels_sorted_by_distance)
            # Some of the properties will be used at the next snapshot.
            r_boundary_prev = r_boundary
            v_halo_prev = halo_velocity
            previous_scale = current_scale
            previous_min_density = np.min(density_arr)
            # Add the result from the current scale factor to the result arrays.
            vmax_array[i] = vmax
            rmax_array[i] = rmax
            cv_array[i] = cv
            rboundary_array[i] = r_boundary
            halo_velocity_list.append(halo_velocity)
        # Compute halo's distance from the host halo at all snapshots.
        self['distance.from.host'] = compute_distance_from_host(self["com.most.bound"], scales, host_main_branch_dict)
        # Find the disruption scale factor.
        self['scale.disrupt'] = find_disruption(cv_array, scales, self["cv.rapid.drop.fraction"], self["cv.infall.drop.fraction"], self["cv.stays.low.fraction"])
        # Store results.
        self['vmax'] = vmax_array
        self['rmax'] = rmax_array
        self['cv'] = cv_array
        self['r.boundary'] = rboundary_array
        self['halo.velocity'] = np.array(halo_velocity_list)
#-----------------------------------------------------------------------------------
# Fuctions
#-----------------------------------------------------------------------------------
def analyze_halo(halo_ID, infall_snapshot, particle_fname, host_main_branch_dict, snapnum_info_dict, BH_parameters, out_f):
    '''
    * This is a wrapper function for performing subhalo analysis for ONE halo.
    - Files/data required:
      - host tree main branch file: DMO and Disk
      - DMO surviving halo main branch file
      - DMO subtree main branch file
      - Disk subtree file with subhalos found by using the infall criteria
    '''
    # Initialize and set halo data for the current subhalo.
    halo_obj = halo(halo_ID, infall_snapshot, particle_fname, snapnum_info_dict, BH_parameters)
    # Compute particle energies at the infall snapshot.
    halo_obj.compute_particle_energies_at_infall()
    # Compute more accurate COMs using most bound particles at infall.
    halo_obj.most_bound_particle_com()
    # Compute halo properties at each snapshot.
    halo_obj.compute_halo_properties(host_main_branch_dict)
    print(halo_obj["ID.halo.infall"], halo_obj["snapshot.infall"], flush=True, file=out_f)
    print(halo_obj["file.path"], flush=True, file=out_f)
    print(halo_obj["number.of.particles"], flush=True, file=out_f)
    print(halo_obj["most.bound.number"], flush=True, file=out_f)
    print(halo_obj["com"][:3], flush=True, file=out_f)
    print(halo_obj["com.most.bound"][:3], flush=True, file=out_f)
    print("", flush=True, file=out_f)
    return(halo_obj)
#
def compute_distance_from_host(halo_com_arr, halo_scale_arr, host_main_branch_dict):
    host_scale_arr = host_main_branch_dict['scale']
    host_com_arr = np.vstack((host_main_branch_dict['x'], host_main_branch_dict['y'], host_main_branch_dict['z'])).T * 1000.
    # Find, for host's arrays, the index that corresponds to the infall scale factor.
    start_scale = halo_scale_arr[0]
    host_first_idx = np.where(np.isclose(host_scale_arr, start_scale, atol=1e-4))[0][0]
    host_com_arr_use = host_com_arr[host_first_idx:]
    # Compute halo's distances.
    halo_dist_comov = np.linalg.norm(halo_com_arr - host_com_arr_use, None, 1)
    return(halo_dist_comov)
#
def compute_halo_velocity(inner_most_frac, sorted_vels):
    '''
    * Function to compute the (COM) velocity of a halo at a given snapshot.
    - It uses N% innermost particles and their velocities to compute the bulk velocity.
    - It assumes that the input velocity array, sorted_vels, is already sorted by particle distance.
    - Rockstar uses 10%.
    * Input:
    - inner_most_frac: fraction of the innermost particles to use (N / 100).
    - sorted_vels: particle velocity array sorted by particle distance.
    '''
    # Number of particles to compute the velocity with.
    tot_numpart = int(len(sorted_vels))
    numpart_use = int(tot_numpart * inner_most_frac)
    # Velocity array to use: the array is already assumed to be sorted.
    vels_use = sorted_vels[:numpart_use]
    # Compute the bulk velocity.
    halo_vel = halo_utilities.bulk_velocity(vels_use)
    # Return the result.
    return(halo_vel)
#
def compute_particle_energies(halo_com, particle_coords, particle_vels, n_use, particle_mass):
    '''
    * Function to compute the kinetic, gravitational potential, and binding energies for a given set of particles.
    - I assume everything is already in physical units, but it doesn't matter as long as everything is consistent.
    - Use 10% inner-most particles to compute the halo velocity, then compute the kinetic energies of particles: similar to how Rockstar does it.
    - Returns:
        - kinetic energy array
        - potential energy array,
        - binding energy array,
        - index array that would sort the distance array.
    '''
    # Convert coordinates into halocentric units: with respect to the center of the halo.
    rel_coords = particle_coords - halo_com
    # Compute particles' distance and get the sorting indices.
    dist_arr = np.linalg.norm(rel_coords, None, 1)
    dist_sort_idx_arr = np.argsort(dist_arr) # This takes ~0.1s for 1,000,000 elements on my laptop, so it should not cause issues.
    # Distance, coordinates, and velocity arrays sorted by particle distance.
    sorted_dist_arr = dist_arr[dist_sort_idx_arr]
    sorted_coords = rel_coords[dist_sort_idx_arr]
    sorted_vels = particle_vels[dist_sort_idx_arr]
    # Compute the halo velocity.
    inner_most_frac = 0.1
    halo_velocity = compute_halo_velocity(inner_most_frac, sorted_vels)
    # Convert sorted particle velocities into halocentric units: with respect to the halo velocity.
    rel_sorted_vels = sorted_vels - halo_velocity
    # Compute kinetic energies: square the velocity magnitude.
    kin_energy_arr = 0.5 * particle_mass * np.sum((rel_sorted_vels[:n_use] * rel_sorted_vels[:n_use]), axis=1)
    # Compute potential energies.
    # For now, compute the potential energy of each particle due to ALL other halo particles, directly: might want to implement Barnes-Hut or something.
    pot_energy_arr = halo_utilities.particle_PE_direct_sum(sorted_coords[:n_use], sorted_coords, particle_mass)
    # Compute binding energies: It's actually the total energy instead of the negative of it, so the most bound particle has the smallest/most negative value.
    bind_energy_arr = kin_energy_arr + pot_energy_arr
    # Return results.
    return(kin_energy_arr, pot_energy_arr, bind_energy_arr, dist_sort_idx_arr)
#
def find_disruption(cv_array, scale_array, cv_rapid_drop_fraction, cv_infall_drop_fraction, cv_stays_low_fraction):
    '''
    * This function determines when the given subhalo disrupts.
    * How I determine disruption:
      1) First, find snapshots where cv decreases to below 30% of the previous snapshot.
      2) Of these, check if cv is below 20% of cv at infall:
          - Often, a subhalo's cv increases a little when the subhalo crosses the host boundary. It might make more sense use that value than the cv value at infall as a reference point?
      3) Also check if cv does not back back to above 40% of cv at infall in the next 5 snapshots.
    *** Disruption time is defined as the snapshot immediately before the subhalo is found to be completely
        disrupted: so it's the last snapshot the subhalo is found to be a halo.
    * Input:
      - cv_array: cv values for all snapshots
      - scale_array: scale factor array
    * Output:
      - 
    * Things to consider, for the future:
      - Should I add a minimum cv value requirement as well?
        - E.g. cv below 1000 or something like that
      - Using the maximum cv value within the first 3-5 snapshots as the "infall" reference value?
    '''
    '''
    * Rapid drop criteria:
    '''
    # Cv at infall
    cv_infall = cv_array[0]
    # Cv normalized by Cv at infall: infall is the first element of the array.
    cv_norm = cv_array / cv_infall
    # Compute how much the value of cv changes by at each snapshot.
    cv_change = cv_norm[1:] / cv_norm[:-1]
    # Find where cv_change decreases down to the amount set by cv_rapid_drop_fraction.
    cv_rapid_drop_idx = np.where(cv_change <= cv_rapid_drop_fraction)[0]
    # Cv decrease: Cv at the current snapshot divided by Cv at the previous snapshot
    #print(cv_norm[0], np.max(cv_norm))
    print(cv_rapid_drop_fraction, cv_rapid_drop_idx)
#
def find_r_boundary(density_arr, bin_mid_arr, sorted_pdist_arr, sort_idx, last_idx, r_boundary_infall, r_boundary_prev, infall_min_density, previous_min_density, current_scale, previous_scale, infall_scale):
    '''
    * This function finds the turn-over point (r_boundary) in the density profile and separates the "true" density profile part from the added contribution from stripped particles.
    * rho(r)*r^2 turn-over point criteria:
      - First, use scipy's argrelextrema to find local minima, then use the following criteria to find a turn-over point.
        - not the first nor last element of the array
        - smaller than N% of the first value of rho? rho*r^2?
        - located within 300% of r_boundary_infall (comparison must be done in comoving units!)
        - located between 40-200% of r_boundary_prev (comparison must be done in comoving units!)
        - then choose the one with the smallest density value.
      - If there are there are no local minima, or none of the minima meet the criteria:
        - look for zero values of rho*r^2
        - places where the density is below N% of the smallest value of density at infall (comparison must be done in comoving units!): 
          - this aims to take care of cases where there is no clear turn-over point,
          - but the density gradually decreases to an 'unphysical' values.
          - E.g. N_p_infall=400, r_boundary_infall=10kpc subhalo having the density profile extend out to 50kpc surely has the real boundary well within 50kpc.
        - Figuring out a place where there is a big jump in the distance of particles in a sorted array of particles could also be a useful check for r_boundary!
    * r_boundary is important for two reasons:
      - 1. this will basically be used as the radius of the halo,
      - 2. halo properties will be computed only within r_boundary.
    '''
    # Convert physical units to comoving units.
    r_boundary_infall_comoving = r_boundary_infall / infall_scale
    r_boundary_prev_comoving = r_boundary_prev / previous_scale
    infall_min_density_comoving = infall_min_density * infall_scale * infall_scale * infall_scale
    previous_min_density_comoving = previous_min_density * previous_scale * previous_scale * previous_scale
    density_arr_comoving = density_arr * current_scale * current_scale * current_scale
    # Compute rho(r) * r^2: this will be used to find the boundary of the halo.
    rho_r_squared = density_arr * bin_mid_arr * bin_mid_arr
    first_rho_r_squared = rho_r_squared[0]
    last_rho_r_squared = rho_r_squared[-1]
    last_rho_r_squared_idx = int(len(rho_r_squared)-1)
    # Find local minima of the rho(r)*r^2 profile.
    local_min_idx = argrelextrema(rho_r_squared, np.less)
    # List for local minimum indices that meet the criteria: for one snapshot.
    min_idx_use = []
    # r_boundary_found tracks whether a density turn-over point (or a reasonable halo boundary) has been found or not.
    r_boundary_found = False
    # Check if local minima exist.
    if len(local_min_idx) > 0:
        # Local minima exist, so check if they meet the density turn-over criteria.
        for j in range(len(local_min_idx[0])):
            # Local minimum index to use
            current_min_idx = local_min_idx[0][j]
            # Value of rho*r^2 at the current index
            current_min_rho_r_squared = rho_r_squared[current_min_idx]
            # Value of distance at the current index, in comoving units
            current_min_r_comoving = bin_mid_arr[current_min_idx] / current_scale
            # Check if the current minimum satisfies the criteria.
            if (current_min_idx!=0) and (current_min_idx!=last_rho_r_squared_idx) and \
            (current_min_rho_r_squared < 0.02*first_rho_r_squared) and \
            (current_min_r_comoving < 3.*r_boundary_infall_comoving) and \
            (0.4*r_boundary_prev_comoving < current_min_r_comoving < 2.*r_boundary_prev_comoving):
                min_idx_use.append(current_min_idx)
    if len(min_idx_use) == 1:
        # There is exactly one minimum that meets all criteria: use this as the turn-over point.
        turn_over_idx = min_idx_use[0]
        r_boundary = bin_mid_arr[turn_over_idx]
        r_boundary_found = True
        test = 'minimum found'
    elif len(min_idx_use) > 1:
        # There are multiple minima that meet all criteria.
        # Choose the first occurance of these minima as the true minimum.
        turn_over_idx = min_idx_use[0]
        r_boundary = bin_mid_arr[turn_over_idx]
        test = 'minimum found'
        r_boundary_found = True
    if r_boundary_found == False:
        # No suitable r_boundary found from the local minimum test.
        # This automatically takes care of both len(min_idx_use)==0 and len(local_min_idx)==0.
        # Do the next r_boundary test: look for density bins with zero values.
        zero_value_bin_idx = np.where(rho_r_squared == 0.)[0]
        zero_idx_use = []
        if len(zero_value_bin_idx) > 0:
            # There are bins with a zero density value.
            # Check if these look like the "edge" of the halo.
            for j in range(len(zero_value_bin_idx)):
                current_zero_idx = zero_value_bin_idx[j]
                current_zero_r_comoving = bin_mid_arr[current_zero_idx] / current_scale
                # Check if these zero value points satisfy the distance criteria.
                if (current_zero_r_comoving < 3.*r_boundary_infall_comoving) and \
                (0.4*r_boundary_prev_comoving < current_zero_r_comoving < 2.*r_boundary_prev_comoving):
                    zero_idx_use.append(current_zero_idx)
            if len(zero_idx_use) > 0:
                # At least one reasonable zero-value point is found: use the first occurance of these as the turn-over point.
                turn_over_idx = zero_idx_use[0]
                r_boundary = bin_mid_arr[turn_over_idx]
                r_boundary_found = True
                test = 'r_boundary found from zero density value check'
    if r_boundary_found == False:
        # No suitable r_boundary found from zero density bin check.
        # Do the next r_boundary test: look for a place in the sorted particle distance array where the difference between adjacent particles is over 1kpc.
        pdist_difference_arr = sorted_pdist_arr[1:] - sorted_pdist_arr[:-1]
        pdist_big_jump_idx = np.where(pdist_difference_arr>1.)[0]
        big_jump_idx_use = []
        if len(pdist_big_jump_idx) > 0:
            # There are big jumps in particle distances.
            # Check if these are located in a reasonable place that makes it a reasonable "edge" of the halo.
            for j in range(len(pdist_big_jump_idx)):
                current_big_jump_idx = pdist_big_jump_idx[j]
                # pdist_at_idx_comoving would be r_boundary if it satisfies the distance criteria.
                pdist_at_idx_comoving = sorted_pdist_arr[current_big_jump_idx] / current_scale
                if (pdist_at_idx_comoving < 3.*r_boundary_infall_comoving) and \
                (0.4*r_boundary_prev_comoving < pdist_at_idx_comoving < 2.*r_boundary_prev_comoving):
                    big_jump_idx_use.append(current_big_jump_idx)
            if len(big_jump_idx_use) > 0:
                # At least one reasonable particle distance jump is identified: use the first occurance as the turn-over point (r_boundary).
                r_boundary = sorted_pdist_arr[big_jump_idx_use[0]]
                r_boundary_found = True
                test = 'r_boundary found from particle distance jump check'
    if r_boundary_found == False:
        # No suitable r_boundary found from particle distance jump check.
        # Do the next r_boundary test: look for a place where the density is below N% of the smallest value at infall or N'% of the smallest value at the previous snapshot.
        small_density_idx = np.where((density_arr_comoving < 0.1*infall_min_density_comoving) | (density_arr_comoving < 0.2*previous_min_density_comoving))[0]
        if len(small_density_idx) > 0:
            # Unusually small density values are identified.
            # Check if any of these meet the criteria.
            r_at_small_density_comoving = bin_mid_arr[small_density_idx] / current_scale
            small_density_idx_use = np.where((r_at_small_density_comoving < 3.*r_boundary_infall_comoving) & (r_at_small_density_comoving > 0.4*r_boundary_prev_comoving) & (r_at_small_density_comoving < 2.*r_boundary_prev_comoving))[0]
            if len(small_density_idx_use) > 0:
                # There are density values that satisfy the small density criteria.
                # Use the first occurance as the "turn-over" point.
                small_density_idx_use = small_density_idx_use[0]
                r_boundary = bin_mid_arr[small_density_idx[small_density_idx_use]]
                r_boundary_found = True
                test = 'r_boundary found from small density check'

    if r_boundary_found == False:
        # Absolutely no reasonable density turn-over point or halo boundary found.
        r_boundary = sorted_pdist_arr[-1]
        test = 'no r_boundary found: using the farthese particle distance as r_boundary'
    #
    return(r_boundary)

