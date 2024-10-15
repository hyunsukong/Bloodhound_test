###################################################################################################
#
# Modules and libraries
#
###################################################################################################
import os
import sys
import pandas as pd
import numpy as np
import h5py
import time
import pathlib
from astropy.cosmology import FlatLambdaCDM
from scipy.signal import argrelextrema
#
# Define useful global constants.
omega_l=0.6879
omega_m=0.3121
h=0.6751
H0=0.1*h  # km / s / kpc
part_mass=1.9693723*1e-06*1e10/h
G=4.30091*1e-6 # kpc / Msolar (km/s)^2
cosmo = FlatLambdaCDM(H0=67.51, Om0=0.3121, Tcmb0=2.725)
t0 = cosmo.age(0).value
#
# Modules I made:
# Import my rockstar analysis module.
import rockstar_handling as rh
'''
* Import my halo utility module:
- Functions
- Halo class
'''
import halo_utilities as halo_util
#from hyunsu.halo_utilities import halo
print(f"* halo_utilities.py is read in: \n {halo_util}")
###################################################################################################
#
# Functions
#
###################################################################################################
'''
* This function finds repeated halo IDs in an array and replaces the repeated ones by attaching trailing zeros.
- Returns a modified array.
'''
def replace_repeated_halo_ID(halo_ID_arr):
    # Find repeated IDs.
    uniq_ids, counts = np.unique(halo_ID_arr, return_counts=True)
    repeat_idx = np.where(counts > 1)[0]
    #
    if len(repeat_idx) == 0:
        # If there's no repeated ID, return the original array.
        return(halo_ID_arr)
    #
    else:
        repeat_id = uniq_ids[repeat_idx]
        print(repeat_id)
        #
        # Loop through each repeated ID.
        for i in range(len(repeat_id)):
            current_repeat_id = repeat_id[i]
            # Find where the repeated ID is in the original array.
            repeat_id_idx = np.where(halo_ID_arr == current_repeat_id)[0]
            #
            print(repeat_id_idx)
            #
            # Loop through the found indices and replace them.
            for i in range(len(repeat_id_idx)):
                current_idx = repeat_id_idx[i]
                current_id = halo_ID_arr[current_idx]
                #
                if i == 0:
                    # Do nothing if it's the first time: it's not "repeated".
                    continue
                #
                # Make a new ID to replace the repeated ones.
                new_id = current_id * int(10000 * 10**(i))
                #
                # Replace the old, repeated ID with the new one.
                halo_ID_arr[current_idx] = new_id
                #
                print(f"* {current_id} replaced by {new_id} at index {current_idx}")
        #    
        return(halo_ID_arr)
'''
* This function finds repeated halo IDs in an array and replaces the repeated ones by attaching trailing zeros.
- Returns a modified array.
'''
def replace_repeated_halo_ID_old(halo_ID_arr):
    # Find repeated IDs.
    uniq_ids, counts = np.unique(halo_ID_arr, return_counts=True)
    repeat_idx = np.where(counts > 1)[0]
    #
    if len(repeat_idx) == 0:
        # If there's no repeated ID, return the original array.
        return(halo_ID_arr)
    #
    else:
        repeat_id = uniq_ids[repeat_idx]
        #
        # Find where the repeated IDs are in the original array.
        repeat_id_idx = np.where(halo_ID_arr == repeat_id)[0]
        #
        # Loop through the found indices and replace them.
        for i in range(len(repeat_id_idx)):
            current_idx = repeat_id_idx[i]
            current_id = halo_ID_arr[current_idx]
            #
            if i == 0:
                # Do nothing if it's the first time: it's not "repeated".
                continue
            #
            # Make a new ID to replace the repeated ones.
            new_id = current_id * int(10000 * 10**(i))
            #
            # Replace the old, repeated ID with the new one.
            halo_ID_arr[current_idx] = new_id
            #
            print(f"* {current_id} replaced by {new_id} at index {current_idx}")
        #    
        return(halo_ID_arr)
'''
* This function takes an analyzed halo object and makes a (tree-like) halo property catalog.
- Returns a pandas dataframe.
- Does either DMO or Disk, only one set at a time.
'''
def make_halo_property_catalog(halo_obj, attr_names, run_type):
    # Property names that have different values for DMO and Disk runs.
    check_names = ['halo_vel', 'vmax', 'rmax', 'com', 'disrupt_scale', 'dist', 'rvir_infall', 'subtree_id', 'subtree_type']
    #
    # An empty dictionary to append key-value pairs - will be converted to a dataframe at the end.
    result_dict = {}
    #
    if run_type == "DMO":
        run_type = "dmo"
    #    
    elif run_type == "Disk":
        run_type = "disk"
    #    
    for i in range(len(attr_names)):
        # Make the attribute name.
        attr_name = attr_names[i]
        #
        # Attach "dmo" or "disk"
        if attr_name in check_names:
            attr_name = f"{run_type}_{attr_names[i]}"
        #
        # Get the attribute - value pairs 
        # Note: some attributes (e.g.) have one value instead of an array, but turning it into a dataframe
        #       seems to take care of that.
        value = np.array(getattr(halo_obj, attr_name))
        #
        if attr_names[i] == 'com':
            # COM should be stored as ['x', 'y', 'z'].
            result_dict['x'] = value[:,0]
            result_dict['y'] = value[:,1]
            result_dict['z'] = value[:,2]
        #
        else:
            # Append to key-value pair to the dictionary
            # Using attr_names[i] because I don't want "dmo" or "disk".
            result_dict[attr_names[i]] = value
    #    
    # Convert the final dictionary to a Pandas dataframe.
    result_df = pd.DataFrame(index=None).from_dict(result_dict)
    return(result_df)
'''
* This function looks for merger tree data corresponding to a given subhalo.
- This should be used when:
  1) I have subhalo properties at the infall snapshot.
  2) I don't have matching merger tree data already.
- Looks for the match in both halo and subhalo tree data.
- Uses COM and Vmax criteria at the infall scale factor.
  - I'm using infall scale factor for now, but I think it can be used at any scale factor?
*** This function is very specific: it needs both subtree and tree files and they are assumed to be
    main branch files.
* Flow:
1) Look for the corresponding tree in both the surviving and destroyed tree files.
2) If only one file has found trees:
3) Check whether the Vmax is within a provided range (vmax_range) or not.
4) If both have found trees:
5) Using Vmax, determine which has a better match.
*** I need to make sure I'm not skipping just because I have a found tree from one set (surv/dest).
    There might be a better match in the other set.
* Input:
  - halo_com: COM x, y, z for the subhalo to match at the given scale factor
              Make sure units are already converted to match those in the tree file.
  - halo_vmax: Vmax of the subhalo from particles, at the given scale factor
  - scale_factor: scale factor to use, assumed to be rounded to the same decimal places.
  - surv_tree: tree data for surviving halos
  - dest_tree: tree data for destroyed (sub)halos
  - coord_query_range: range of coordinates to use to query - e.g. +- 0.002 Mpc  
  - vmax_range: within x%
* Output:
  - Tree ID of the identified tree: int
  - Which type of file the tree came from (dest vs. surv): string
'''
def match_halo_to_tree(halo_com, halo_vmax, scale_factor, surv_tree, dest_tree, coord_query_range, vmax_range):
    # found checks whether a correct match has been found or not.
    found=False
    #
    '''
    * First, identify the subhalo in both the destroyed and surviving main branch files, at the given scale factor.
    * Various querying methods:
    - df.query()
    - df[df.A > df.B] with df.scale.round(3).values and np.around()
    - np.isclose(t1, t2, atol=1e-03) 
    '''
    # Query subtree and tree files by the given scale factor.
    dest_scale_query = dest_tree[np.isclose(dest_tree.scale.values, scale_factor, atol=1e-04)]
    surv_scale_query = surv_tree[np.isclose(surv_tree.scale.values, scale_factor, atol=1e-04)]
    #
    # Query the scale factor queried dataframes by coordinates.
    dest_com_query = halo_util.match_halo_to_catalog_com(halo_com, dest_scale_query, coord_query_range)
    surv_com_query = halo_util.match_halo_to_catalog_com(halo_com, surv_scale_query, coord_query_range)
    #
    '''
    * Possible cases:
    - No tree is identified in both files.
    - One file has trees identified.
    - Both files have trees identified.
    '''
    # When neither file has identified trees.
    if len(dest_com_query)==0 and len(surv_com_query)==0:
        #print(f"* No tree is identified at scale factor {scale_factor}")
        #print(halo_vmax)
        match_tree_id = -1
        ftype="None"
        return([match_tree_id, ftype])
    #
    # When one of the files has identified trees.
    elif len(dest_com_query)==0 or len(surv_com_query)==0:
        # Subtree file (destroyed halos) has identified trees.
        if len(dest_com_query) > 0:
            found_tree = dest_com_query
            match_tree_ids = found_tree.subtree_id.values
            ftype="dest"
        #
        # Tree file (surviving halos) has identified trees.
        else:
            found_tree = surv_com_query
            match_tree_ids = found_tree.tree.values
            ftype="surv"
        #    
        # When there is exactly one tree found:
        if len(found_tree) == 1:
            vmax_iden = found_tree.vmax.values[0]
            #
            # Compare vmax_iden with Vmax from particles, using vmax_range.
            if np.absolute(halo_vmax - vmax_iden)/halo_vmax < vmax_range:
                #final_tree = found_tree
                match_tree_id = match_tree_ids[0]
            #
            else:
                # Identified tree didn't pass the Vmax criterion, so there is no final tree.
                match_tree_id = -1
                #print(vmax_iden, halo_vmax)
                ftype="None"
        #  
        # When there are more than one tree found:
        else:
            vmax_iden = found_tree.vmax.values
            #
            # Find the tree with the closest Vmax value to halo_vmax.
            closest_idx = np.argmin(np.absolute(halo_vmax - vmax_iden))
            closest_vmax = vmax_iden[closest_idx]
            #
            # Check if the closest Vmax is within vmax_range of the Vmax from particles.
            if np.absolute(halo_vmax - closest_vmax)/halo_vmax < vmax_range:
                match_tree_id = match_tree_ids[closest_idx]
            else:
                # Identified trees didn't pass the Vmax criterion, so there is no final match.
                match_tree_id = -1
                ftype="None"
    else:
        '''   
        * When both files have identified trees.
        * Possible cases:
          - Both have exactly one tree identified: compare their Vmax right away.
          - More than one tree identified in either files: find the closest Vmax first than compare the 
        closest Vmax from one file to the closest Vmax from the other file.
        '''
        dest_vmaxs = dest_com_query.vmax.values
        surv_vmaxs = surv_com_query.vmax.values
        # When both files have exactly one tree identified each:
        if len(dest_com_query)==1 and len(surv_com_query)==1:
            # Compute the absolute Vmax difference between identified Vmaxs and halo_vmax.
            dest_vmax_diff = np.absolute(halo_vmax - dest_vmaxs[0])
            surv_vmax_diff = np.absolute(halo_vmax - surv_vmaxs[0])
            #
            # Choose a Vmax that's closer to halo_vmax.
            if dest_vmax_diff > surv_vmax_diff:
                # Vmax from the surviving halo file is closer to Vmax from particles.
                found_tree = surv_com_query
                match_tree_ids = found_tree.tree.values
                closer_vmax_diff = surv_vmax_diff
                ftype="surv"
            else:
                # Vmax from the destroyed halo file is closer to Vmax from particles.
                found_tree = dest_com_query
                match_tree_ids = found_tree.subtree_id.values
                closer_vmax_diff = dest_vmax_diff
                ftype="dest"
        else:
            '''
            * When either has more than one tree (including when both has more than one).
              - I don't think this cases exists for the set of 320 subhalos in sim 493.
            * Get closest_vmax from the one with more than one tree: 
            * min/max for length 1 array is just the element itself.
            '''
            print("  * Both surviving and destroyed tree files have more than one matching trees.")
            # Get the smallest vmax_diff values in each set.
            dest_vmax_diff = np.absolute(halo_vmax - dest_vmaxs)
            surv_vmax_diff = np.absolute(halo_vmax - surv_vmaxs)
            smallest_idx_dest = np.argmin(dest_vmax_diff)
            smallest_idx_surv = np.argmin(surv_vmax_diff)
            smallest_diff_dest = dest_vmax_diff[smallest_idx_dest]
            smallest_diff_surv = surv_vmax_diff[smallest_idx_surv]
            #
            # From chosen Vmax values from the two sets, choose a Vmax that's closer to halo_vmax.
            if smallest_diff_dest > smallest_diff_surv:
                # Vmax from the surviving halo file is closer to Vmax from particles.
                found_tree = surv_com_query.iloc[[smallest_idx_surv]]
                match_tree_ids = found_tree.tree.values
                closer_vmax_diff = smallest_diff_surv
                ftype="surv"
            else:
                # Vmax from the destroyed halo file is closer to Vmax from particles.
                found_tree = dest_com_query.iloc[[smallest_idx_dest]]
                match_tree_ids = found_tree.subtree_id.values
                closer_vmax_diff = smallest_diff_dest
                ftype="dest"
        # Check if the closer Vmax is within vmax_range of the Vmax from particles.
        if closer_vmax_diff/halo_vmax < vmax_range:
            match_tree_id = match_tree_ids[0]
        else:
            # Identified tree didn't pass the Vmax criterion, so there is no final tree.
            match_tree_id = -1
            ftype="None"
    return([match_tree_id, ftype])
'''
* This function computes subhalo's distance from the host halo's COM.
* Input:
  - halo_com
  - halo_scale
  - host_com
  - host_scale
*** Input data must be in matching units.
'''
def compute_dist_from_host(halo_com, halo_scale, host_com, host_scale):
    # Find, for host's arrays, the index that corresponds to the infall scale factor.
    start_scale = halo_scale[0]
    host_first_idx = np.where(np.isclose(host_scale, start_scale, atol=1e-4))[0][0]
    #host_first_idx = np.where(np.round(host_scale, decimals=3) == np.round(start_scale, decimals=3))[0][0]
    host_com_matched = host_com[host_first_idx:]
    # Compute halo distances.
    halo_dist_comov = np.linalg.norm(halo_com - host_com_matched, None, 1)
    halo_dist_phys = np.multiply(halo_dist_comov, halo_scale)
    #
    # Return the result in comoving units.
    return(halo_dist_comov)
'''
* This function determines when the given subhalo disrupts.
* How I determine disruption:
  1) First, find snapshots where cv decreases to below 30% of the previous snapshot.
  2) Of these, check if cv is below 20% of cv at infall.
  3) Also check if cv does not back back to above 40% of cv at infall in the next 5 snapshots.
*** Disruption time is defined as the snapshot immediately before the subhalo is found to be completely
    disrupted: so it's the last snapshot the subhalo is found to be a halo.
* Input:
  - cv_array: cv values for all snapshots
  - scale_array: scale factor array
* Output:
  - 
'''
def find_disruption(cv_array, scale_array):
    # Cv normalized by Cv at infall: infall is the first element of the array.
    cv_norm = cv_array / cv_array[0]
    #
    # cv decrease: cv at the current snapshot divided by cv at the previous snapshot
    cv_decrease = cv_norm[1:] / cv_norm[:-1]
    #
    # Find where cv_decrease is below 0.3: where cv drops down to 30% in one snapshot.
    cv_severe_drop_idx = np.where(cv_decrease <= 0.3)[0]
    #
    # disrupt_found checks if an instance that meets all disruption criteria is found.
    disrupt_found = False
    #
    for i in range(len(cv_severe_drop_idx)):
        idx = cv_severe_drop_idx[i]
        #
        # Of these, find those that have cv below 20 % of cv at infall.
        # idx + 1 because cv_decrease is computed from the second element of cv_norm.
        # Check cv doesn't come back up to 40 % of cv at infall in the next 5 snapshots.
        if cv_norm[idx+1] < 0.2 and np.max(cv_norm[idx+1:idx+6]) < 0.4:
            '''
            * Disruption time is defined as the snapshot immediately before the subhalo is found to be completely
              disrupted: so it's the last snapshot the subhalo is found to be a halo.
              - Then, idx instead of idx+1.
            '''
            disrupt_scale = scale_array[idx]
            disrupt_found = True
        #
        # If disruption is found, return result: returning terminates the loop.
        if disrupt_found:
            return(disrupt_scale)
    # If it loops all the way to the end, the halo does not disrupt: return -1.
    return(-1)

'''
* A simple function to assign halo properties as halo class attributes.
- Attributes names and attribute values need to be in the same order!
- Uses function set_attribute(obj, attr_name, value)
* Input:
  - halo_obj: halo class object
  - attr_names: names of attributes to set.
  - properties: attribute values to set - must be put in a list
    - if only one value, use [value]
  - run_type: "DMO" or "Disk" - attaches "dmo" or "disk" in front of attribute names.
'''
def assign_halo_properties(halo_obj, attr_names, properties, run_type):
    if run_type == "DMO":
        run_type = "dmo"
    elif run_type == "Disk":
        run_type = "disk"
    for i in range(len(attr_names)):
        # Make the attribute name.
        attr_name = f"{run_type}_{attr_names[i]}"
        #
        # Set the attribute using halo properties.
        setattr(halo_obj, attr_name, properties[i])
'''
* This function computes Density and Vcirc profiles of halo particles then computes Vmax, Rmax, and Cv.
- Return results as arrays.
- Everything will be computed in physical units, then distance units will be converted to comoving units
  when returning results.
- For some snapshots, especially for the earlier ones where the density profile is still somewhat smooth,
  there won't be a "local" minimum for the rho * r^2 profile. In that case, the minimum index will be zero.
  Later, when I actually use these indices, I need to remember to skip when the index is zero.
* Output:
  - [vcirc_list, pdist_list, dens_list, mid_r_list, vmax_array, rmax_array, cv_array]
  - I want: [halo_vel_list, vcirc_list, pdist_list, dens_list, mid_r_list, vmax_array, rmax_array, cv_array]
'''
def compute_halo_properties(coords, vels, coms, scales, part_mass, ):
    # Result arrays and lists to return.
    #halo_vel_list = []
    vcirc_list = []
    pdist_list = []
    dens_list = []
    mid_r_list = []
    vmax_array = np.zeros(len(scales))
    rmax_array = np.zeros(len(scales))
    cv_array = np.zeros(len(scales))
    #halo_dist_array = np.zeros(len(scales))
    #
    # Loop through each snapshot.
    for i in range(len(scales)):
        scale_now = scales[i]
        coord_now_all = coords[i] * scale_now # pkpc
        com_now = coms[i] * scale_now # pkpc
        vels_now = vels[i] * np.sqrt(scale_now) # physical
        #
        # Use function vcirc_particle_single_halo_snap(com, coords) to compute Vcirc.
        vcirc, sorted_pdist, sort_idx = halo_util.vcirc_particle_single_halo(com_now, coord_now_all) # physical, non-h
        #
        # Compute the halo velocity using my function, compute_halo_velocity(inner_most_frac, sorted_vels):
        #sorted_vels = vels_now[sort_idx]
        #halo_vel = compute_halo_velocity(0.1, sorted_vels)
        #
        # Compute Vmax, Rmax, and Cv using the Vcirc profile just computed.
        cv, vmax, rmax = halo_util.concentration_one_halo(vcirc, sorted_pdist)
        #
        # Use function compute_density_profile(coord_now_all, com_now, r_last, r_min_idx, num_bins, p_mass)
        # to compute the density profile
        r_last = sorted_pdist[-100]
        r_min_idx = 20
        num_bins = 20
        p_mass = part_mass
        #
        dens, num_enclosed, inner_r, mid_r, outer_r, shell_width = halo_util.compute_density_profile(coord_now_all, com_now, 
                                                                                                   r_last, r_min_idx, num_bins, p_mass)
        rho_r_squared = dens * mid_r * mid_r
        #
        # Rvir at infall: 100th farthest particle at the first snapshot.
        if i == 0:
            rvir_infall = r_last # pkpc
        '''
        * Find the "local" minimum of the rho * r^2 profile.
        - This minimum will be the rough "edge" of the subhalo.
        '''
        local_min_idx = argrelextrema(rho_r_squared, np.less)
        #
        # If there is no minimum, vmax_new = vmax etc.
        vmax_new = vmax
        rmax_new = rmax # pkpc
        cv_new = cv
        #
        # First check if a local minimum exists.
        if len(local_min_idx[0]) > 0:
            '''
            * Select the smallest local minimum that:
              - is not last element of the array,
              - is not the first element of the array,
              - has value smaller than the first value of rho * r^2.
              - is located at larger than 0.5 pkpc and within Rvir at infall.
            '''
            # List for local minimum indices that meet the criteria: for one snapshot
            min_idx_use = []
            #
            # Loop through each local minimum index.
            for j in range(len(local_min_idx[0])):
                current_min_idx = local_min_idx[0][j]
                current_min_rho_r2 = rho_r_squared[current_min_idx]
                first_rho_r2 = rho_r_squared[0]
                current_min_r = mid_r[current_min_idx]
                #
                # Check if the current local minimum meets my criteria.
                if (current_min_idx < len(rho_r_squared) - 1) and (current_min_idx != 0) and (current_min_r >= 0.5) and \
                    (current_min_rho_r2 < first_rho_r2) and (current_min_r < rvir_infall):
                    #
                    # Append index to the index list.
                    min_idx_use.append(current_min_idx) 
            '''
            * If there is no minimum that meets all criteria, it probably is because
              the density profile is still somewhat smooth.
              - so use the original vmax and rmax.
            '''
            if len(min_idx_use) == 0:
                # I probably don't need to do this because vmax_new = vmax is already done above,
                # but I do it so I can follow the logic.
                vmax_new = vmax
                rmax_new = rmax # pkpc
                cv_new = cv
            elif len(min_idx_use) == 1:
                # There is only one minimum that meets all criteria.
                # The r value corresponding to the local minimum of rho * r^2 profile.
                rho_min_r = mid_r[min_idx_use[0]]
                #
                # Re-compute Vmax and Rmax using the particles within rho_min_r.
                # Since vcirc and sorted_pdist arrays are sorted by pdist, I can simply use np.searchsorted(pdist, rho_min_r).
                slice_idx = np.searchsorted(sorted_pdist, rho_min_r)
                vcirc = vcirc[:slice_idx]
                sorted_pdist = sorted_pdist[:slice_idx]
                cv_new, vmax_new, rmax_new = halo_util.concentration_one_halo(vcirc, 
                                                                            sorted_pdist)
            elif len(min_idx_use) > 1:
                # There are more than one minimum that meet the criteria - choose the smallest rho*r^2 value.
                rho_minima = rho_r_squared[min_idx_use]
                rho_argmin = np.argmin(rho_minima)
                rho_min_idx = min_idx_use[rho_argmin]
                # The r value corresponding to the local minimum of rho * r^2 profile.
                rho_min_r = mid_r[rho_min_idx]
                #
                # Re-compute Vmax and Rmax using the particles within rho_min_r.
                # Since vcirc and sorted_pdist arrays are sorted by pdist, I can simply use np.searchsorted(pdist, rho_min_r).
                slice_idx = np.searchsorted(sorted_pdist, rho_min_r)
                vcirc = vcirc[:slice_idx]
                sorted_pdist = sorted_pdist[:slice_idx]
                cv_new, vmax_new, rmax_new = halo_util.concentration_one_halo(vcirc, 
                                                                            sorted_pdist)
        # Add results from current snapshot.
        # For snapshots where vcirc and pdist were sliced using rho*r^2 minimum, the sliced list is returned.
        #halo_vel_list.append(halo_vel)
        vcirc_list.append(vcirc)
        #pdist_list.append(sorted_pdist) # pkpc
        pdist_list.append(sorted_pdist / scale_now) # ckpc
        dens_list.append(dens)
        #mid_r_list.append(mid_r) # pkpc
        mid_r_list.append(mid_r / scale_now) # ckpc
        # Although it says x_new, it holds the old value if a new value wasn't computed.
        vmax_array[i] = vmax_new
        #rmax_array[i] = rmax_new # pkpc
        rmax_array[i] = rmax_new / scale_now # ckpc
        cv_array[i] = cv_new
    # Return results as a list of lists and arrays:
    # [lists and arrays]
    return([vcirc_list, pdist_list, dens_list, mid_r_list, vmax_array, 
            rmax_array, cv_array])
    #return([halo_vel_list, vcirc_list, pdist_list, dens_list, mid_r_list, vmax_array, 
    #        rmax_array, cv_array])
'''
* Get N% most bound particles and compute their COMs.
- This function computes new COMs for halo particles using the most bound particles at the first snapshot.
- COM at infall will not be re-computed but will be appended to the final list.
'''
def most_bound_particle_com(coords_to_track, bind_energy, dist_sort_idx, N_frac, infall_com):    
    # Argsort the binding energy array.
    bind_sorted_idx = np.argsort(bind_energy)
    #
    # Get N% most bound particles at infall.
    # If N% is below 100 particles, use 100 particles.
    num_part = np.int32(np.rint(len(coords_to_track[0]) * N_frac))
    if num_part < 100:
        num_part = 100
    #
    '''
    * Track selected particles forward in time.
    * Take the indices to track the particle coordinates with: up to num_part.
      Remember: bind_sorted is sorted twice - once by distance from center then by binding energy.
      so need something like coords[first_argsort[second_argsort[:num_part]]]
      Also, I can use the initial value of subhalo COM for the infall snapshot because I expect 
      the COM to be correct at early snapshots.
    '''
    most_bound_argsort = bind_sorted_idx[:num_part]
    track_idx = dist_sort_idx[most_bound_argsort]
    #
    # Compute new COMs using most bound particles.
    new_com_list = []
    #
    # Append COM at infall as the first element.
    new_com_list.append(infall_com)
    #
    for i in range(len(coords_to_track)):
        tracked_bound_coords = coords_to_track[i][track_idx]
        #
        # Compute COM using tracked most bound particles.
        new_com = halo_util.cm(tracked_bound_coords, nofile=True, num_part=100, nel_lim=100)
        #
        # Append COM result for one subhalo.
        new_com_list.append(new_com) # ckpc
    #
    # Return the result: new COMs for all snapshots
    return(new_com_list)
'''
* This function computes the (COM) velocity of a halo at a given snapshot.
- It uses N% innermost particles' velocity to compute the bulk velocity.
- N is given by the variable inner_most_frac.

* Input:
- inner_most_frac:
- sorted_vels: velocity array sorted by particle's distance.

* Output:
- halo_vel: halo's velocities, vx, vy, vz

'''
def compute_halo_velocity(inner_most_frac, sorted_vels):
    # Number of particles to compute the velocity with.
    tot_numpart = int(len(sorted_vels))
    numpart_use = int(tot_numpart * inner_most_frac)
    #
    # Velocity array to use: the array is already assumed to be sorted.
    vels_use = sorted_vels[:numpart_use]
    #
    # Compute the bulk velocity.
    halo_vel = halo_util.bulk_velocity(vels_use)
    #
    return(halo_vel)
'''
* This function computes PE, KE, and binding energy for given particles.
- It does not care whether the halo is from DMO or Disk runs. It just takes whatever particles are given.
* Everything in this function will already assume to be in physical units.
* Use 10% inner-most particles to compute the halo velocity, then compute the kinetic energies of particles.

* Output:
  - A list of arrays:
    - kinetic energy
    - potential energy
    - binding energy
    - argsort indices for the distance array
'''
def compute_particle_energies(halo_com, coords, vels, n_use, part_mass):
    # Put the coordinates in halocentric units.
    rel_coords = coords - halo_com # pkpc
    #
    # Compute particles' distance and argsort.
    part_dist_phys = np.linalg.norm(rel_coords, None, 1) # pkpc
    dist_sort_idx = np.argsort(part_dist_phys)
    #
    # dist, coords, and velocity arrays sorted by particles' distance.
    sorted_dist = part_dist_phys[dist_sort_idx] # pkpc
    sorted_coords = rel_coords[dist_sort_idx]
    sorted_vels = vels[dist_sort_idx]
    #
    # Compute the halo velocity.
    inner_most_frac = 0.1
    halo_vel = compute_halo_velocity(inner_most_frac, sorted_vels)
    #print(f"  * Halo velocity computed: {halo_vel}")
    #
    # Put the velocities in halocentric units.
    rel_sorted_vels = vels - halo_vel # physical
    #
    # Compute the kinetic energy in physical units.
    kin_energy = (0.5 * part_mass * np.sum((rel_sorted_vels[:n_use] * rel_sorted_vels[:n_use]), axis=1))
    #
    # Compute the potential energy using my particle_PE_direct_sum function.
    pot_energy = halo_util.particle_PE_direct_sum(sorted_coords[:n_use], sorted_coords, part_mass)
    #
    # Compute the binding energy.
    bind_energy = kin_energy + pot_energy
    #
    # Return results.
    return([kin_energy, pot_energy, bind_energy, dist_sort_idx])
'''
* This function computes PE, KE, and binding energy for a halo at its infall snapshot, for both DMO and Disk.
* It uses function compute_particle_energies() to compute the energies.
* Tree's velocities are in physical units: no hubble flow.
- convert it to comoving units by dividing it by sqrt(a)

* Input:
  - infall_com: COM at the infall snapshot, ckpc
  - infall_coords: ckpc
  - infall_a: Use it to convert to pkpc
  - infall_vels: ckpc
'''
def compute_particle_energies_at_infall(infall_com, infall_coords, infall_vels, infall_a, part_mass):
    # Convert comoving units to physical units.
    infall_com = infall_com * infall_a # Non-h pkpc
    infall_coords = infall_coords * infall_a # Non-h pkpc
    infall_vels = infall_vels * np.sqrt(infall_a) # physical
    #
    # Number of particles to compute energies for.
    n_use = len(infall_coords) // 4
    if n_use < 3000:
        if len(infall_coords) < 3000:
            n_use = len(infall_coords)
        else:
            n_use = 3000
    #
    energies = compute_particle_energies(infall_com, infall_coords, infall_vels, n_use, part_mass)
    #
    # Return results: sorted particle lists, kinetic, potential, and binding energies
    return(energies)
'''
* This function sets the initial halo data using my halo class.
* Input:
  - halo_ID
  - dmo_fname
  - disk_fname
  - full_a_arr: full scale factor array for the simulation
  - full_z_arr: full redshift array for the simulation
  
* Output:
  - halo class with:
    - particle coordinates, velocities, and IDs
    - snapshot number, scale factor, and redshift
    - initial computation of halo COM
    at each snapshot
'''
def set_initial_halo_data(halo_ID, dmo_fname, disk_fname, full_a_arr, full_z_arr):
    # Initialize the halo object with the halo ID.
    current_halo = halo_util.halo(halo_ID)
    #
    # Set DMO and Disk halo data - need file paths.
    current_halo.set_dmo(dmo_fname, full_a_arr, full_z_arr)
    current_halo.set_disk(disk_fname)
    #
    # Set the number of particles as an attribute.
    current_halo.num_part = len(current_halo.disk_coords[0])
    #
    # Compute snapshots' lookback time and save as class attribute.
    lb_time = cosmo.lookback_time(current_halo.redshifts).value
    current_halo.lookback_times = lb_time
    #
    # Compute COMs: these initial computations of COMs will be replaced later with more accurate ones.
    current_halo.compute_com()
    #
    # Return the halo object.
    return(current_halo)
'''
* This function runs my subhalo particle analysis for one halo.
* The full analysis will need:
  1) Subtree data satisfying my infall criteria from one set (DMO or Disk) of merger tree data.
  2) DMO and Disk subhalo particle data corresponding to 1)
  3) Subtree data from the other set (Disk of DMO) corresponding to 2). 
     - I think I want to do it after the use of this function.
'''
#def subhalo_analysis(hID, fname_dmo, fname_disk, disk_subtree, full_a_arr, full_z_arr, part_mass, matching_tree_found):
def subhalo_analysis(hID, fname_dmo, fname_disk, disk_subtree, full_a_arr, full_z_arr, part_mass, host_coords_dmo, host_scale_dmo, host_coords_disk, host_scale_disk):
    # Initialize the halo data for the current subhalo.
    print("  * Initializing the particle data for the current halo...")
    current_halo = set_initial_halo_data(hID, fname_dmo, fname_disk, full_a_arr, full_z_arr)
    #print(f"  * Halo's initial data is set: coordinates, velocities, pIDs, COM, snapnum, scale factor, redshift, lb time")
    
    # Set the subtree_id as a class attribute.
    current_halo.disk_subtree_id = disk_subtree.subtree_id.values[0]   

    '''
    * Compute particle energies at the infall snapshot.
    - Computing the potential energy can be done just using the particles.
      - Can't use the same COMs for Disk and DMO - they can be significantly different - checked!
    - Computing the kinetic energy can now be done just using the particles.
    - Result list: kinetic, potential, binding energies and argsort indices for particle distance for DMO and Disk.
    '''
    #print(f"  * Computing particle energies at the infall snapshot and updating COMs...")
    dmo_energies = compute_particle_energies_at_infall(current_halo.dmo_com[0], current_halo.dmo_coords[0], current_halo.dmo_vels[0], current_halo.scale_factors[0], part_mass)
    #print(f"    * DMO done.")
    disk_energies = compute_particle_energies_at_infall(current_halo.disk_com[0], current_halo.disk_coords[0], current_halo.disk_vels[0],
                                                      current_halo.scale_factors[0], part_mass)
    #print(f"    * Disk done.")
    
    # Get 2% most bound particles at infall, then track them and compute COMs.
    N_frac = 0.02
    dmo_coms = most_bound_particle_com(current_halo.dmo_coords[1:], dmo_energies[2],
                                       dmo_energies[3], N_frac, current_halo.dmo_com[0])
    disk_coms = most_bound_particle_com(current_halo.disk_coords[1:], disk_energies[2],
                                        disk_energies[3], N_frac, current_halo.disk_com[0])
    
    # Update halo's COMs with new COMs.
    current_halo.dmo_com = dmo_coms
    current_halo.disk_com = disk_coms
    
    '''
    * Compute subhalo properties.
    - Vmax, Rmax, Cv
    - Computed using "bound" particles determined at each snapshot using the minima of rho(r)*r^2.
    - Returned results are: [vcirc_list, pdist_list, dens_list, mid_r_list, vmax_array, 
                             rmax_array, cv_array, rvir_infall]
    - Set results as class attributes.
    '''
    # DMO
    dmo_properties = compute_halo_properties(current_halo.dmo_coords, current_halo.dmo_vels,
                                             current_halo.dmo_com, current_halo.scale_factors, part_mass)
    
    # Disk
    disk_properties = compute_halo_properties(current_halo.disk_coords, current_halo.disk_vels,
                                              current_halo.disk_com, current_halo.scale_factors, part_mass)

    # Set computed subhalo properties as class attributes.
    # Names of subhalo properties to set.
    # Don't use 'halo_vel' yet: the method I have right now uses a fixed number of particles
    # and is inaccurate at later times.
    attr_names = ['vcirc', 'pdist', 'density', 'dens_r_bin', 'vmax', 'rmax', 'cv']
    assign_halo_properties(current_halo, attr_names, dmo_properties, "DMO")
    assign_halo_properties(current_halo, attr_names, disk_properties, "Disk")
    
    # Compute subhalo's distance from the host halo's tree: only for Disk now.
    # Comoving units
    # Save the result in ckpc: When I'm done with the DMO part, this process can be moved into compute_halo_properties.
    # Worry about this later.
    halo_dist_dmo = compute_dist_from_host(current_halo.dmo_com, current_halo.scale_factors,
                                           host_coords_dmo*1000., host_scale_dmo)
    halo_dist_disk = compute_dist_from_host(current_halo.disk_com, current_halo.scale_factors,
                                            host_coords_disk*1000., host_scale_disk)
    current_halo.dmo_dist = halo_dist_dmo
    current_halo.disk_dist = halo_dist_disk
    
    # Find when the subhalo disrupts: disruption scale factor or -1
    dmo_disrupt_scale = find_disruption(current_halo.dmo_cv, current_halo.scale_factors)
    disk_disrupt_scale = find_disruption(current_halo.disk_cv, current_halo.scale_factors)
    
    # Set disruption scale factor as a class attribute: non-disrupted takes -1.
    attr_names = ['disrupt_scale']
    assign_halo_properties(current_halo, attr_names, [dmo_disrupt_scale], "DMO")
    assign_halo_properties(current_halo, attr_names, [disk_disrupt_scale], "disk")
    
    # Return the final halo object.
    return(current_halo)
