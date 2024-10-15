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
'''
* Import my subhalo analysis functions.
'''
import subhalo_analysis_functions as sub_analysis
#from hyunsu.halo_utilities import halo
#
# Open an output statement file.
out_f = open("result_text.txt", 'w')
# 
print(f"* halo_utilities.py is read in: \n {halo_util}", file=out_f)
###################################################################################################
#
# File names for input data.
#
###################################################################################################
t_s_overall = time.time()
# Simulation number and type to use.
simnum=1107
simtype='disk'
simtype_dmo = 'dmo'
base_dir = f"/scratch/05097/hk9457/hyunsu/pELVIS/z13/disk_from_ranch/halo_{simnum}/subhalo_analysis"
print("", file=out_f)
print("*********************************************************************************", file=out_f)
print(f"* Beginning tracked subhalo analysis for simulation {simnum}!", file=out_f)
print("*********************************************************************************", file=out_f)
print("", file=out_f)
#
# Simulation snapshot header file that contains number/redshift/scale factor information.
snap_header_name = f'{base_dir}/snapshot_header.txt'
#
# File path for the host tree main branch file.
host_main_branch_path_dmo = f'{base_dir}/{simnum}_{simtype_dmo}_host_main_branch_new.csv'
host_main_branch_path = f'{base_dir}/{simnum}_{simtype}_host_main_branch_new.csv'
#
# File path for the Disk subtree main branch file.
subtree_fpath = f'{base_dir}/{simnum}_{simtype}_host_subtree_main_branch.hdf5'
#
# File paths for DMO surviving halo main branch trees.
if simnum==493:
    surv_tree_fpath = f'{base_dir}/main_branches_dmo_{simnum}_tyler.csv'
else:
    surv_tree_fpath = f'{base_dir}/main_branches_dmo_{simnum}_new.csv'
#
# File paths for DMO subtree main branch file.
dest_tree_fpath = f'{base_dir}/{simnum}_dmo_host_subtrees_main_branch_new.hdf5'
#
# File path for the Disk subtree file with subhalos found by using the infall criteria.
infall_sub_fpath = f'{base_dir}/{simnum}_{simtype}_subtrees_infall_criteria.hdf5'
#
# File path for the identified DMO subtree/trees.
iden_dmo_tree_fpath = f'{base_dir}/{simnum}_{simtype_dmo}_subtrees_matched_stampede.hdf5'
#
# File paths for tracked subhalo particles.
disk_part_dir = f'{base_dir}/massive_subs/{simtype}'
dmo_part_dir = f'{base_dir}/massive_subs/{simtype_dmo}'
#
# File paths for the subid-hid pair path
subid_hid_pair_path = f'/scratch/05097/hk9457/hyunsu/pELVIS/z13/disk_from_ranch/halo_{simnum}/particle_tracking/{simnum}_disk_infall_criteria_tid_hid_pairs.csv'
#
# File path for the halo property catalog I make
# Save the final dataframe as a .hdf5 file.
out_df_path_dmo = f'{base_dir}/{simnum}_tree_from_tracked_particles_dmo_stampede.hdf5'
out_df_path_disk = f'{base_dir}/{simnum}_tree_from_tracked_particles_disk_stampede.hdf5'
###################################################################################################
#
# Get snapshot number - redshift - scale factor information from snapshot_header.txt I made.
#
###################################################################################################
full_snap_list = []
full_z_list = []
full_a_list = []
with open(snap_header_name) as f:
    f.readline()
    for line in f:
        line_list = line.split()
        full_snap_list.append(int(line_list[0]))
        full_z_list.append(float(line_list[1]))
        full_a_list.append(float(line_list[2]))
        #print(line_list)
full_snap_arr = np.array(full_snap_list)
full_z_arr = np.array(full_z_list)
full_a_arr = np.around(np.array(full_a_list), decimals = 7)
print(f"  * Snapshot number - redshift - scale factor information read in from {snap_header_name}", file=out_f)
print("*********************************************************************************", file=out_f)
print("", file=out_f)
###################################################################################################
#
# Read in host tree's main branch file (.csv)
#
###################################################################################################
print(f"  * Reading in host tree main branch data...", file=out_f)
print("", file=out_f)
# Disk
host_main_branch = pd.read_csv(host_main_branch_path)
#
# Get various properties of the host tree as arrays.
# Flip the arrays so they are ordered from early time time to late time.
host_x = np.flip(host_main_branch.x.values/h)
host_y = np.flip(host_main_branch.y.values/h)
host_z = np.flip(host_main_branch.z.values/h)
host_coords = np.vstack((host_x, host_y, host_z)).T # comoving non-h Mpc.
host_scale = np.flip(host_main_branch.scale.values)
host_redshift = 1. / host_scale - 1.
host_t_cosmic = cosmo.age(host_redshift).value
host_t_lookback = cosmo.lookback_time(host_redshift).value
host_vmax = np.flip(host_main_branch.vmax.values)
host_rvir = np.flip(host_main_branch.rvir.values/h)
host_rvir_pkpc = np.multiply(host_rvir, host_scale)
#
print(f"* Disk host tree main branch:", file=out_f)
print(f"* File path: {host_main_branch_path}", file=out_f)
print(f"* Number of snapshots: {len(host_main_branch)}", file=out_f)
print("* Scale factors:", file=out_f)
print(host_scale[:10], file=out_f)
print(host_rvir_pkpc[-10:], file=out_f)
print(host_vmax[-10:], file=out_f)
print("", file=out_f)
#
# DMO
host_main_branch_dmo = pd.read_csv(host_main_branch_path_dmo)
print(host_main_branch_dmo.columns, file=out_f)
#
# Get various properties of the host tree as arrays.
# Flip the arrays so they are ordered from early time time to late time.
host_x_dmo = np.flip(host_main_branch_dmo.x.values/h)
host_y_dmo = np.flip(host_main_branch_dmo.y.values/h)
host_z_dmo = np.flip(host_main_branch_dmo.z.values/h)
host_coords_dmo = np.vstack((host_x_dmo, host_y_dmo, host_z_dmo)).T # comoving non-h Mpc.
host_scale_dmo = np.flip(host_main_branch_dmo.scale.values)
host_redshift_dmo = 1. / host_scale_dmo - 1.
host_t_cosmic_dmo = cosmo.age(host_redshift_dmo).value
host_t_lookback_dmo = cosmo.lookback_time(host_redshift_dmo).value
host_vmax_dmo = np.flip(host_main_branch_dmo.vmax.values)
if simnum==493:
    host_rvir_dmo = np.flip(host_main_branch_dmo.Rvir.values/h)
else:
    host_rvir_dmo = np.flip(host_main_branch_dmo.rvir.values/h)
host_rvir_pkpc_dmo = np.multiply(host_rvir_dmo, host_scale_dmo)
#
print(f"* DMO host tree main branch:", file=out_f)
print(f"* File path: {host_main_branch_path_dmo}", file=out_f)
print(f"* Number of snapshots: {len(host_main_branch_dmo)}", file=out_f)
print("* Scale factors:", file=out_f)
print(host_scale_dmo[:10], file=out_f)
print(host_rvir_pkpc_dmo[-10:], file=out_f)
print(host_vmax_dmo[-10:], file=out_f)
print("*********************************************************************************", file=out_f)
print("", file=out_f)
###################################################################################################
#
# Open merger tree and subtree files for DMO.
#
###################################################################################################
print(f"  * Reading in DMO merger tree and subtree data...", file=out_f)
print("", file=out_f)
t_s = time.time()
#
# Open files.
t_s_open = time.time()
dest_dmo_tree_file = pd.read_hdf(dest_tree_fpath)
t_e_open = time.time()
print(f"* Time taken to open destroyed subtree files: {t_e_open - t_s_open:.3f} s", file=out_f)
#
t_s_open = time.time()
surv_dmo_tree_file = pd.read_csv(surv_tree_fpath)
t_e_open = time.time()
print(f"* Time taken to open surviving tree files: {t_e_open - t_s_open:.3f} s", file=out_f)
#
t_e = time.time()
print("", file=out_f)
print(f"* Time taken: {t_e - t_s:.3f} s", file=out_f)
print("*********************************************************************************", file=out_f)
print("", file=out_f)
###################################################################################################
#
# Read in the .hdf5 file that contains Disk subtrees found using my infall criteria.
# - Since these subtrees already satisfy my infall criteria, all I need to do is to get their infall time.
#
###################################################################################################
print(f"  * Read in the .hdf5 file that contains Disk subtrees found using my infall criteria... ", file=out_f)
print("", file=out_f)
t_s = time.time()
#
infall_subtree_df = pd.read_hdf(infall_sub_fpath)
infall_subids = np.unique(infall_subtree_df.subtree_id.values)
#
# Group the dataframe by subtree ids.
infall_subtree_grouped = infall_subtree_df.groupby('subtree_id')
#
# List to contain the infall scale factor and infall index of these subtrees.
subtree_infall_scales = []
subtree_infall_idx = []
#
for i in range(len(infall_subids)):
    current_subid = infall_subids[i]
    current_subtree = infall_subtree_grouped.get_group(current_subid)
    #
    # Find first infall scale factor.
    infall_a, infall_idx = halo_util.find_first_infall_from_subtree(current_subtree, 
                        np.around(host_scale, decimals=5), np.around(host_rvir, decimals=5))
    #
    # Append to the final list.
    subtree_infall_scales.append(infall_a)
    subtree_infall_idx.append(infall_idx)
# 
subtree_infall_scales = np.array(subtree_infall_scales)
subtree_infall_idx = np.array(subtree_infall_idx)
#
# Get the indices of subtrees corresponding to infall redshifts: 0.1<z<=1, 1<z<=2, 2<z<=3.
# -> high_z_scale <= a < low_z_scale
subtree_idx_0_z_1 = np.where(subtree_infall_scales >= (1./(1.+1)))[0]
subtree_idx_1_z_2 = np.where((subtree_infall_scales >= (1./(2.+1))) 
                                 & (subtree_infall_scales < (1./(1.+1))))[0]
subtree_idx_2_z_3 = np.where((subtree_infall_scales >= (1./(3.+1))) 
                                 & (subtree_infall_scales < (1./(2.+1))))[0]
#
# Subtree ids for these subsets.
#subid_0_z_1 = infall_subids[subtree_idx_0_z_1]
#subid_1_z_2 = infall_subids[subtree_idx_1_z_2]
#subid_2_z_3 = infall_subids[subtree_idx_2_z_3]
#
print(f"* Total number of subtrees read in: {len(infall_subtree_grouped)}", file=out_f)
print(f"* Number of subtrees with infall between:", file=out_f)
print(f"  * 0.1<z<=1: {len(subtree_idx_0_z_1)}", file=out_f)
print(f"  * 1.0<z<=2: {len(subtree_idx_1_z_2)}", file=out_f)
print(f"  * 2.0<z<=3: {len(subtree_idx_2_z_3)}", file=out_f)
#   
t_e = time.time()
print("", file=out_f)
print(f"* Time taken: {t_e - t_s:.3f}s", file=out_f)
print("*********************************************************************************", file=out_f)
print("", file=out_f)
###################################################################################################
#
# Get Subtree ID - my Rockstar ID pairs from the ID pair file.
#
###################################################################################################
print(f"  * Getting Subtree ID - my Rockstar ID pairs from the ID pair file... ", file=out_f)
print("", file=out_f)
# Read in the .csv file as a pandas dataframe.
subID_hID_df = pd.read_csv(subid_hid_pair_path)
disk_subIDs = subID_hID_df.subtree_id_disk.values
disk_hIDs = subID_hID_df.halo_id_disk.values
#
# Find repeated hIDs: ones with trailing 00000.
repeated_hIDs = []
repeated_subIDs = []
repeated_infall_snaps = []
for i in range(len(disk_hIDs)):
    current_hID = disk_hIDs[i]
    if str(current_hID)[-5:] == '00000':
        repeated_hID = int(str(current_hID)[:-5])
        repeated_subID = disk_subIDs[i]
        repeated_infall_snap = int(subID_hID_df.iloc[i].infall_snap)
        repeated_hIDs.append(repeated_hID)
        repeated_subIDs.append(repeated_subID)
        repeated_infall_snaps.append(repeated_infall_snap)
#
print(f"* Number of Disk tID-hID pairs read in: {len(subID_hID_df)}", file=out_f)
print("* Repeated hID information:", file=out_f)
print(f"  * hID: {repeated_hIDs}, subID: {repeated_subIDs}, infall snapshot:{repeated_infall_snaps}", file=out_f)
print("*********************************************************************************", file=out_f)
print("", file=out_f)
###################################################################################################
#
# Get tracked subhalo particle file names.
#
###################################################################################################
print(f"  * Getting tracked subhalo particle file names...", file=out_f)
t_s = time.time()
#
print(f"* Disk subhalo particle file path:", file=out_f)
print(disk_part_dir, file=out_f)
# Get file names from the directories above. fnames = [hID_list, infall_snap_list, full_file_name_list]
disk_fnames = halo_util.get_halo_particle_file_names(disk_part_dir)
dmo_fnames = halo_util.get_halo_particle_file_names(dmo_part_dir)
#
hIDs_to_open = disk_fnames[0]
fnames_to_open_dmo = dmo_fnames[2]
fnames_to_open_disk = disk_fnames[2]
#
# Total number of halo particle files and setting the current index.
print(f"* Total number of halo particle files to use: {len(hIDs_to_open)}", file=out_f)
print("", file=out_f)
#
# Print the file size.
print(f"  * File sizes:", file=out_f)
for i in range(len(fnames_to_open_disk)):
    current_fname = fnames_to_open_disk[i]
    current_fsize = os.path.getsize(current_fname)
    print(f"  * Halo {hIDs_to_open[i]}: {current_fsize/1024./1024.:03f} MB", file=out_f)
t_e = time.time()
print("", file=out_f)
print(f"* Time taken: {t_e - t_s:.3f} s", file=out_f)
print("*********************************************************************************", file=out_f)
print("", file=out_f)
###################################################################################################
#
# Read in tracked halo particles, then do analysis, one halo at a time.
#- Store results in two files:
#  - 1) A catalog with subhalo properties: much like the merger tree catalog.
#  - 2) A data file with Vcirc, pdist, density, mid_r arrays
#
###################################################################################################
t_s = time.time()
'''
* A list of keys to use: halo property names, put in manually.
* Keys (and values) shared by both DMO and Disk results:
  - rock_hID: halo ID from Rockstar result the halo particles came from - Disk runs for this case.
  - scale
  - snapnum
* keys_use holds the key names I want to use as column names for the final catalog.
* attr_use holds the attribute names that correspond to keys_use names.

*** For now, DMO results are going to have missing data:  "dist" etc.
    - I need to add these later, soon!
'''
#keys_use_dmo = ['halo_ID', 'subtree_id', 'scale', 'snapnum', 'vmax', 'rmax', 'x', 'y', 'z', 'disrupt_scale',
#           'dist']
attr_use_dmo = ['halo_ID', 'subtree_id', 'subtree_type', 'scale_factors', 'num_part', 'snapnums',  'vmax',
                'rmax', 'com', 'dist', 'disrupt_scale']
#
#keys_use_disk = ['halo_ID', 'subtree_id', 'scale', 'snapnum', 'vmax', 'rmax', 'x', 'y', 'z', 'disrupt_scale',
#           'dist']
attr_use_disk = ['halo_ID', 'subtree_id', 'scale_factors', 'num_part', 'snapnums', 'vmax', 'rmax', 
                 'com', 'dist', 'disrupt_scale']
#
# An empty list to append result to.
# Each element will be a dataframe for each halo.
full_df_list_dmo = []
full_df_list_disk = []
#
# prev_hID checks for repeated halo_IDs.
prev_hID = 0
#
# Loop through all halo files.
for i in range(len(hIDs_to_open)):
    #i=i+25
    t_s_halo = time.time()
    '''
    * Input data for my analysis function
    '''
    # Current halo's Disk Rockstar ID - from halo particle file title.
    # File names to open
    current_hID = hIDs_to_open[i]
    current_dmo_path = fnames_to_open_dmo[i]
    current_disk_path = fnames_to_open_disk[i]
    print(f"  * Analyzing halo {current_hID}...", file=out_f)
    print("", file=out_f)
    #
    '''
    * Matching halo particle data with my subtree data.
    * This is already done when I first match subtrees to my Rockstar, so use that information.
    * There could be halos with the same Rockstar halo ID since Rockstar IDs are only unique within the snapshot.
    * I found one pair with the same ID in 1<z<2 set, but it looks like I did their 
      initial matching (trees-Rockstar) correctly such that there are 155 unique subtree_ids for
      154 unique halo_IDs.
    * I just need to match them correctly.
    * I assume I will never run into the case where there are three same IDs.
      If this happens, I need to come up with a better fix then.
    *** tID-hID pair file has the snapshot number for the repeated subhalo. Using this, change the tracked subhalo file name manually!
    '''
    ID_idx = np.where(disk_hIDs == current_hID)[0][0]
    current_disk_subid = disk_subIDs[ID_idx]
    current_disk_subtree = infall_subtree_grouped.get_group(current_disk_subid)
    #
    '''
    * Use the function subhalo_analysis to run my subhalo analysis.
    '''
    print(f"  * Running the function 'subhalo_analysis'...", file=out_f)
    t_s_func = time.time()
    current_halo = sub_analysis.subhalo_analysis(current_hID, current_dmo_path, current_disk_path, current_disk_subtree, full_a_arr, full_z_arr, part_mass, host_coords_dmo, host_scale_dmo, host_coords, host_scale)
    t_e_func = time.time()
    print(f"  * Time taken: {t_e_func - t_s_func:.3f} s", file=out_f)
    print("", file=out_f)
    '''
    * Now that halo properties are computed, find the corresponding DMO tree using the halo 
      at the infall snapshot and save the tree ID as a class attribute.
      - .disk_subtree_id already exists, so make .dmo_subtree_id
      - use match_halo_to_tree(halo_com, halo_vmax, scale_factor, surv_tree, dest_tree, coord_query_range, vmax_range)
      - match_halo_to_tree() returns [match_tree_id, ftype].
      - Convert COM from particles to cMpc/h to match the units in the merger tree data.
    '''
    print(f"  * Finding the corresponding DMO tree using the halo at the infall snapshot...", file=out_f)
    dmo_tree_iden_result = sub_analysis.match_halo_to_tree(current_halo.dmo_com[0]* h / 1000.,
                                              current_halo.dmo_vmax[0],
                                              current_halo.scale_factors[0],
                                              surv_dmo_tree_file,
                                              dest_dmo_tree_file,
                                              0.0015, 0.3)
    #
    # Add DMO tree ID and type.
    current_halo.dmo_subtree_id = dmo_tree_iden_result[0]
    current_halo.dmo_subtree_type = dmo_tree_iden_result[1]
    #print(dir(current_halo))
    #
    '''
    * Make and save a halo property catalog for tracked subhalos.
    '''
    print(f"  * Making the halo property catalog for tracked subhalos...", file=out_f)
    df_dmo = sub_analysis.make_halo_property_catalog(current_halo, attr_use_dmo, "DMO")
    df_disk = sub_analysis.make_halo_property_catalog(current_halo, attr_use_disk, "Disk")
    #
    # Append the result to the final dataframe list.
    full_df_list_dmo.append(df_dmo)
    full_df_list_disk.append(df_disk)
    #
    t_e_halo = time.time()
    print("", file=out_f)
    print(f"  * Subhalo analysis done!", file=out_f)
    print(f"* Time taken for halo {current_hID}: {t_e_halo - t_s_halo:.3f} s", file=out_f)
    print("*********************************************************************************", file=out_f)
    print("", file=out_f)
#
t_e = time.time()
print("", file=out_f)
print(f"* Time taken to analyze all {len(hIDs_to_open)} halos: {t_e - t_s:.3f} s", file=out_f)
print("*********************************************************************************", file=out_f)
print("", file=out_f)
###################################################################################################
#
# Make a subtree/tree file for analyzed subhalos.
# The halo property catalog I made above should have the dmo tree information.
#
###################################################################################################
print(f"  * Making and saving the tracked subhalo property files with the result...", file=out_f)
t_s = time.time()
#
# Convert the list of dataframes to one big dataframe.
df_dmo_all = pd.concat(full_df_list_dmo)
df_disk_all = pd.concat(full_df_list_disk)
print(f"* Total number of DMO subtrees found: {len(df_dmo_all.groupby('halo_ID'))}", file=out_f)
print(f"* Total number of Disk subtrees found: {len(df_disk_all.groupby('halo_ID'))}", file=out_f)
print(f"* Length of the final dataframe: {len(df_dmo_all), len(df_disk_all)}", file=out_f)
print("  * The lengths should be the same as the dataframes contain results from infall to z=0.", file=out_f)
#
# Save the final dataframe as a .hdf5 file.
df_dmo_all.to_hdf(out_df_path_dmo, key='df', mode='w')
df_disk_all.to_hdf(out_df_path_disk, key='df', mode='w')
#
print("", file=out_f)
print(f"* Final subtree dataframes saved at:", file=out_f)
print(out_df_path_dmo, file=out_f)
print(out_df_path_disk, file=out_f)
#
t_e = time.time()
print("", file=out_f)
print(f"* Time taken: {t_e - t_s:.3f} s", file=out_f)
print("*********************************************************************************", file=out_f)
print("", file=out_f)
###################################################################################################
#
# Check the saved files.
#
###################################################################################################
print("  * Checking the saved files...", file=out_f)
print("", file=out_f)
t_s = time.time()
mytree_df_dmo = pd.read_hdf(out_df_path_dmo)
mytree_df_disk = pd.read_hdf(out_df_path_disk)
hids_all = np.unique(mytree_df_dmo.halo_ID.values)
#
# Group the dataframe by halo IDs.
mytree_dmo_grouped = mytree_df_dmo.groupby('halo_ID')
mytree_disk_grouped = mytree_df_disk.groupby('halo_ID')
#
print(f"* Total number of DMO subhalos: {len(mytree_dmo_grouped)}", file=out_f)
print(f"* Total number of Disk subhalos: {len(mytree_disk_grouped)}", file=out_f)
print("", file=out_f)
print(f"  * DMO file:", file=out_f)
print(mytree_df_dmo[:5], file=out_f)
print("*********************************************************************************", file=out_f)
print("", file=out_f)
print(f"  * Disk file:", file=out_f)
print(mytree_df_disk[:5], file=out_f)
print("", file=out_f)
#
t_e = time.time()
print("", file=out_f)
print(f"* Time taken: {t_e - t_s:.3f} s", file=out_f)
print("*********************************************************************************", file=out_f)
print("", file=out_f)
###################################################################################################
#
# Make a subtree file for identified DMO trees/subtrees.
#
###################################################################################################
print("  * Making a tree file for identified DMO trees/subtrees...", file=out_f)
print("", file=out_f)
t_s = time.time()
no_tree_counter = 0
dmo_trees_list = []
print(f"  * Number of subhalos analyzed: {len(hids_all)}", file=out_f)
for i in range(len(hids_all)):
    current_hid = hids_all[i]
    current_tree = mytree_dmo_grouped.get_group(current_hid)
    current_tid = current_tree.subtree_id.values[0]
    current_ttype = current_tree.subtree_type.values[0]
    if current_tid == -1:
        no_tree_counter+=1
        #print(current_ttype)
        continue
    else:
        if current_ttype == "dest":
            # Get the tree from the destroyed tree file.
            found_tree = dest_dmo_tree_file.query('subtree_id == @current_tid')
            #
            # Rename columns: {"Mvir":"mvir", "Rvir":"rvir"} df.rename(columns={"A": "a", "B": "c"})
            found_tree.rename(columns={"Mvir":"mvir", "Rvir":"rvir"}, inplace=True)
            #
            # Final tree to use.
            final_tree = found_tree[["subtree_id", "scale", "id", 'Orig_halo_ID', "pid", "upid", "mvir", "vmax", "rvir", "rs", 'x', 'y', 'z', 'vx', 'vy', 'vz', "dist"]]
        elif current_ttype == "surv":
            found_tree = surv_dmo_tree_file.query('tree == @current_tid')
            # Surviving trees are ordered from late to early snapshots. Invert this.
            found_tree = found_tree.iloc[::-1]
            #
            # Rename columns: {"tree": "subtree_id"}
            found_tree.rename(columns={"tree": "subtree_id"}, inplace=True)
            #
            # Compute the distance.
            # Match scale factors between the current subtree and host's tree.
            scales_sub = found_tree.scale.values
            if scales_sub[0] < host_scale_dmo[0]:
                # Subhalo's tree starts before host's main branch tree.
                first_idx = 0
                sub_idx = np.where(np.isclose(scales_sub, host_scale_dmo[0], atol=1e-4))[0][0]
                sub_coords = sub_coords[sub_idx:]
                found_tree = found_tree[sub_idx:]
                scales_sub = scales_sub[sub_idx:]
            else:
                # Subhalo's tree starts at the same or a later snapshot than the host's main branch tree.
                first_idx = np.where(np.isclose(host_scale_dmo, scales_sub[0], atol=1e-4))[0][0]
            #
            # Host's coordinates at these scale factors.
            matched_host_coords = host_coords_dmo[first_idx:]
            #
            # Subtree's coordinates
            sub_coords = found_tree[['x','y','z']].values / h #cMpc, non-h
            #
            # Compute subhalo distance and set it do the tree datafame.
            current_dist = np.linalg.norm(sub_coords - matched_host_coords, None, 1) * 1000. #ckpc, non-h
            found_tree["dist"] = current_dist
            #
            # Add a Orig_halo_ID column with -1: Tyler's main branch tree .csv file doesn't have this information.
            orig_hID_col = np.empty(len(scales_sub))
            orig_hID_col.fill(-1)
            found_tree["Orig_halo_ID"] = orig_hID_col
            #
            # Final tree to use.
            final_tree = found_tree[["subtree_id", "scale", "id", 'Orig_halo_ID', "pid", "upid", "mvir", "vmax", "rvir", "rs", 'x', 'y', 'z', 'vx', 'vy', 'vz', "dist"]]     
    #
    # Append the found tree to the final list.
    dmo_trees_list.append(final_tree)
#
# Combine the list of dataframes to on big dataframe.
dmo_subtree_df_full = pd.concat(dmo_trees_list)
#
# Save the final dataframe as a .hdf5 file.
dmo_subtree_df_full.to_hdf(iden_dmo_tree_fpath, key='df', mode='w')
print(f"  * Number of subhalos without an identified DMO tree/subtree: {no_tree_counter}", file=out_f)
print(f"  * Number of subhalos I was able to find a corresponding tree/subtree for: {len(dmo_trees_list)}", file=out_f)
t_e = time.time()
print("", file=out_f)
print(f"* Time taken: {t_e - t_s:.3f} s", file=out_f)
print("*********************************************************************************", file=out_f)
print("", file=out_f)
#
t_e_overall = time.time()
print("", file=out_f)
print(f"* Total time taken for simulation {simnum}: {t_e_overall - t_s_overall:.3f} s", file=out_f)
out_f.close()
