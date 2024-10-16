#####################################################################################
##### This is the main script file for the subhalo tracking package Bloodhound. #####
##### author: Hyunsu Kong                                                       #####
##### email: hyunsukong@utexas.edu                                              #####
##### website: https://hyunsukong.github.io/                                    #####
##### GitHub (for Bloodhound):                                                  #####
#####################################################################################
'''
Units:

Required input data:
    - 
'''
#-----------------------------------------------------------------------------------
# Import libraries.
#-----------------------------------------------------------------------------------
from astropy.cosmology import FlatLambdaCDM
from astropy.modeling.physical_models import NFW
import h5py
import numpy as np
import pandas as pd
import struct
import time
#-----------------------------------------------------------------------------------
# Import local modules/libraries/scripts.
#-----------------------------------------------------------------------------------
import halo_analysis
import halo_utilities
import infall_subhalo_criteria
import utilities
#
#-----------------------------------------------------------------------------------
# Input parameters
#-----------------------------------------------------------------------------------
parameter_fname = '/scratch/05097/hk9457/FIREII/m12c_r7100/bloodhound_subhalo_tracking/Bloodhound/BH_parameters/bloodhound_parameters_m12c_r7100.txt'
#
#-----------------------------------------------------------------------------------
# Fuctions
#-----------------------------------------------------------------------------------
def BH_initialization(parameter_fname, header_statement):
    '''
    * This function handles some of the initialization processes for Bloodhound.
    - Reads in the parameter file.
    - Open an output statement file.
    - Write the header for the output statement file: description and date/time.
    - Set two global variables COSMO and T0:
        - COSMO: Astropy FlatLambdaCDM object.
                 Although it is NOT a constant, I follow python's convention and capitalize the variable
                 because, well, it's useful.
        - T0: Age of the universe, constant.
    - Read in snapshot number, redshift, scale factor data.
    - Read in simulation numbers to use, from the parameter dictionary.
    - Returns:
        - BH_parameters: a dictionary containing the parameters read in from the parameter file.
        - sim_nums: a list containing the simulation numbers to use.
        - base_dir: a string containing the path for the parent directory of Bloodhound.
        - out_f: a text file to write output statements.
        - snapnum_info_dict: a dictionary containing the snapshot number, redshift, and scale factor data.
    '''
    #
    # Read in Bloodhound parameters.
    BH_parameters = utilities.read_parameters(parameter_fname)
    #
    # Open an output statement file.
    text_fdir = BH_parameters["out_statement_dir"]
    text_fname_base = BH_parameters["bloodhound_out_statement_fname_base"]
    out_f = utilities.open_output_statement_file(text_fdir, text_fname_base)
    #
    # Write the header for the output statement file: description and date/time.
    utilities.write_header_for_result_text_file(header_statement, out_f)
    # Print the parameters.
    utilities.print_params(BH_parameters, out_f)
    #
    # Read in snapshot number, redshift, scale factor data.
    snapnum_info_fname = BH_parameters["snapnum_info_fname"]
    sim_name = BH_parameters["simulation_name"]
    snapnum_info_dict = utilities.open_snap_header_file(snapnum_info_fname, sim_name)
    # Add the snapshot time information to the BH parameter dictionary.
    BH_parameters['time_info_dict'] = snapnum_info_dict
    #
    # Get simulation numbers to use, from the parameter dictionary.
    sim_nums = BH_parameters['sim_nums']
    base_dir = BH_parameters['base_dir']
    #
    # Add a couple of useful items to the parameter data.
    h = BH_parameters['h']
    omega_m = BH_parameters['omega_m']
    temp_cmb = BH_parameters['temp_cmb']
    H0_kpc = 0.1*h # km/s/kpc
    H0_Mpc = 100. * h # km/s/Mpc
    cosmo = FlatLambdaCDM(H0=H0_Mpc, Om0=omega_m, Tcmb0=temp_cmb)
    t0 = cosmo.age(0).value
    BH_parameters['cosmo'] = cosmo
    BH_parameters['t0'] = t0
    #
    # Return the result.
    return(BH_parameters, sim_nums, base_dir, out_f, snapnum_info_dict)
#def halo_particle_tracking_wrapper_function():
'''
* This function
'''
def read_in_infalling_subtree_data_FIRE(BH_parameters, sim_num, tree_type, out_f):
    #
    # File name for the infall criteria subtree file.
    fdir = BH_parameters['infall_subtree_out_dir']
    fname_base = BH_parameters['infall_subtree_out_fname_base']
    fname = f"{fdir}/{sim_num}_{tree_type}_{fname_base}.hdf5"
    #
    # Read in the data.
    infall_subtree_df = pd.read_hdf(fname)
    return(infall_subtree_df)
#
def read_in_infalling_subtree_data(BH_parameters, sim_num, out_f):
    #
    # File name for the infall criteria subtree file.
    sim_type = BH_parameters['run_type']
    fdir = BH_parameters['infall_subtree_out_dir']
    fname_base = BH_parameters['infall_subtree_out_fname_base']
    fname = f"{fdir}/{sim_num}_{sim_type}_{fname_base}.hdf5"
    #
    # Read in the data.
    infall_subtree_df = pd.read_hdf(fname)
    return(infall_subtree_df)
'''

'''
def remove_incomplete_subtrees_FIRE(infall_subtree_df, BH_parameters, out_f):
    print(f"* 1) Number of subhalos in the infalling subtree file: {len(infall_subtree_df.groupby('tree.tid'))}", file=out_f)
    # Take only infalling subhalos: infalling? = 1.
    infalling_query = infall_subtree_df.query("`infalling?` == 1")
    print(f"* 2) Number of actually infalling subhalos in 1) (infalling? = 1): {len(infalling_query.groupby('tree.tid'))}", flush=True, file=out_f)
    #
    print("", flush=True, file=out_f)
    return(infalling_query)
#
def remove_incomplete_subtrees(infall_subtree_df, BH_parameters, out_f):
    print(f"* 1) Number of subhalos in the infalling subtree file: {len(infall_subtree_df.groupby('subtree_id'))}", file=out_f)
    if BH_parameters["two_rockstars"] == 1:
        # Halo matching between two Rockstar sets was done, so
        # take only subtrees that has a real value for ID.halo.infall: remove those with ID.halo.infall=-1.
        hID_query = infall_subtree_df.query("`ID.halo.infall` != -1")
        #
        # Take only infalling subhalos: infalling? = 1.
        infalling_query = hID_query.query("`infalling?` == 1")
        print(f"* 2) Number of subtrees with a real ID.halo.infall value (not -1): {len(hID_query.groupby('subtree_id'))}", flush=True, file=out_f)
        print(f"* 3) Number of actually infalling subhalos in 2) (infalling? = 1): {len(infalling_query.groupby('subtree_id'))}", flush=True, file=out_f)
    else:
        # Take only infalling subhalos: infalling? = 1.
        infalling_query = infall_subtree_df.query("`infalling?` == 1")
        print(f"* 2) Number of actually infalling subhalos in 1) (infalling? = 1): {len(infalling_query.groupby('subtree_id'))}", flush=True, file=out_f)
    #
    print("", flush=True, file=out_f)
    return(infalling_query)
#
def get_infall_information(infall_subtree_df):
    '''
    * 
    - Every returned array should be ordered by scale.infall from early to late times.
    - For ID.subtree and ID.halo.infall arrays, each element contains an array of IDs corresponding to the given scale.infall.
    '''
    # Initialize the result dictionary.
    infall_information_dict = {}
    sid_list = []
    hid_list = []
    #
    # Get unique arrays for scale.infall and snapshot.infall, preserving the order.
    infall_information_dict["scale.infall"] = infall_subtree_df['scale.infall'].unique()
    infall_information_dict["snapshot.infall"] = infall_subtree_df['snapshot.infall'].unique()
    for i in range(len(infall_information_dict["snapshot.infall"])):
        current_snapnum = infall_information_dict["snapshot.infall"][i]
        current_subtrees = infall_subtree_df.query("`snapshot.infall`== @current_snapnum")
        #
        # Get subtree_id and ID.halo.infall values corresponding to the current scale.infall value.
        sid_list.append(current_subtrees['tree.tid'].unique())
        hid_list.append(current_subtrees['ID.halo.infall'].unique())
    #
    infall_information_dict["ID.subtree"] = sid_list
    infall_information_dict["ID.halo.infall"] = hid_list
    return(infall_information_dict)
#
def get_hIDs_and_num_ps(catalog_ascii_fnames):
    hID_numP_pairs_df_list = []
    for i in range(len(catalog_ascii_fnames)):
        fname = catalog_ascii_fnames[i]
        #
        # Open the .ascii halo catalog file.
        text_df=pd.read_csv(fname, sep=" ", low_memory=False)
        #
        # Take the data part: the first 19 rows contain simulation/snapshot information and I don't need them.
        reduced_data=text_df[19:]
        #
        # Take two columns: id and num_p
        two_columns=reduced_data[['#id', 'num_p']]
        two_columns=two_columns.rename(columns={'#id':'id'})
        two_columns=two_columns.astype(int)
        #
        # Append the dataframe to the result list.
        hID_numP_pairs_df_list.append(two_columns)
    return(hID_numP_pairs_df_list)
#
def get_rockstar_particle_ID_data(catalog_bin_fnames, hID_numP_pairs_df_list, out_f):
    particle_ID_list = []
    for i in range(len(catalog_bin_fnames)):
        # File name for the .bin Rockstar result file.
        bin_fname = catalog_bin_fnames[i]
        #
        # Number of halos in the current catalog block.
        hID_numP_pairs_df = hID_numP_pairs_df_list[i]
        num_halos=len(hID_numP_pairs_df)
        #
        # Get the number of particles array for all halos in the current file.
        num_p_arr = hID_numP_pairs_df.num_p.values
        #
        # Open the binary file.
        bin_file = open(bin_fname, 'rb')
        #
        # Skip header and halo tables.
        bin_file.seek(256+296*num_halos)
        #
        # Get integer particle IDs for each halo and attend the particle ID arrays (tuples) to a list.
        pID_list_current_block = []
        for num_p in num_p_arr:
            particle_IDs = struct.unpack("Q" * num_p, bin_file.read(8 * num_p))
            pID_list_current_block.append(particle_IDs)
        particle_ID_list.append(pID_list_current_block)
        #
        # Check if EOF has been reached: "not remaining_dat" is True if remaining_dat is empty.
        remaining_dat = bin_file.read()
        #if not remaining_dat:
        #    print(f"  * Block {i}: successfully reached EOF!", flush=True, file=out_f)
        if remaining_dat:
            print(f"  * Block {i}: EOF not reached!", flush=True, file=out_f)
            print(f"    * Remaining data is: {remaining_dat}", flush=True, file=out_f)
        #
        # Close the current binary file.
        bin_file.close()
    #
    # Return the particle ID list.
    return(particle_ID_list)
#
def check_num_halos(particle_ID_list, hID_numP_pairs_df_list, out_f):
    if len(particle_ID_list) != len(hID_numP_pairs_df_list):
        print("  * Number of blocks for particle_ID_list and hID_numP_pairs_df_list are different!", flush=True, file=out_f)
    for i in range(len(particle_ID_list)):
        if len(particle_ID_list[i]) != len(hID_numP_pairs_df_list[i]):
            print(f"  * Block {i}: number of halos in particle ID data and hID-numP pair data are different!", flush=True, file=out_f)
#
def get_particle_IDs_of_halo(hID, hID_numP_pairs_df_list, particle_ID_list, out_f):
    '''
    
    - hID_numP_pairs_df_list and particle_ID_list have exactly the same structure.
    '''
    for i in range(len(hID_numP_pairs_df_list)):
        current_block_hID_arr = hID_numP_pairs_df_list[i].id.values
        #
        # Check if the current block contains the current hID.
        if hID in current_block_hID_arr:
            block_idx = i
            halo_idx = np.where(current_block_hID_arr==hID)[0][0]
            break
        else:
            continue
    #
    # Get the particle IDs for the current halo: the result is a tuple, so convert it to an array.
    pID_arr = np.array(particle_ID_list[block_idx][halo_idx])
    #
    # Test if the number of particles in pID_arr is the same as that given by the num_p column in hID_numP_pairs_df_list.
    # This should always be true, but it might be useful to have a error message when it isn't!
    if hID_numP_pairs_df_list[block_idx].num_p.values[halo_idx] != len(pID_arr):
        print(f"  * Number of particles inconsistent! halo ID: {hID}, block index: {block_idx}, halo index: {halo_idx}", flush=True, file=out_f)
    #
    # Return the particle IDs.
    return(pID_arr)
#
def get_infall_particle_IDs(infall_information_dict, BH_parameters, sim_num, out_f):
    '''
    '''
    # Particle IDs for each halo will be appended to a list in infall_information_dict.
    infall_information_dict['ID.particle'] = []
    #
    # Number of rockstar output files per snapshot.
    num_rockstar_files = BH_parameters['num_rockstar_files']
    #
    #
    if BH_parameters['simulation_name']=='pELVIS':
        halo_finding_output_dir = f"{BH_parameters['base_dir']}/{BH_parameters['run_type']}/halo_{sim_num}/rockstar_catalogs/rockstar_output"
    elif BH_parameters['simulation_name']=='FIRE':
        halo_finding_output_dir = f"{BH_parameters['base_dir']}/halo/rockstar_dm/catalog"
    #
    #
    infall_snapshot_arr = infall_information_dict["snapshot.infall"]
    infall_hid_list = infall_information_dict["ID.halo.infall"]
    #
    for i in range(len(infall_snapshot_arr)):
        current_snap = infall_snapshot_arr[i]
        current_snap_infall_hid_arr = infall_hid_list[i]
        #
        # Read in Rockstar halo particle data for the current snapshot.
        catalog_ascii_fnames = utilities.make_rockstar_fnames(halo_finding_output_dir, num_rockstar_files, current_snap, 'ascii')
        catalog_bin_fnames = utilities.make_rockstar_fnames(halo_finding_output_dir, num_rockstar_files, current_snap, 'bin')
        hID_numP_pairs_df_list = get_hIDs_and_num_ps(catalog_ascii_fnames)
        #
        print(f"* Reading in halo particle ID data for snapshot {current_snap}... ", end="", flush=True, file=out_f)
        t_s_step = time.time()
        particle_ID_list = get_rockstar_particle_ID_data(catalog_bin_fnames, hID_numP_pairs_df_list, out_f)
        t_e_step = time.time()
        utilities.print_time_taken(t_s_step, t_e_step, "*" ,True, out_f)
        #
        # Just in case, check that the number of Rockstar blocks and number of halos are consistent between
        # the data read in from the .bin file and the .ascii file.
        check_num_halos(particle_ID_list, hID_numP_pairs_df_list, out_f)
        #
        # Get particle IDs of halos infalling at the current snapshot.
        current_snap_pID_list = []
        print(len(current_snap_infall_hid_arr), len(particle_ID_list))
        for j in range(len(current_snap_infall_hid_arr)):
            current_hID = current_snap_infall_hid_arr[j]
            current_halo_pID_tuple = get_particle_IDs_of_halo(current_hID, hID_numP_pairs_df_list, particle_ID_list, out_f)
            current_snap_pID_list.append(current_halo_pID_tuple)
        #
        # Append the pID list for the current snapshot to the result dictionary.
        infall_information_dict['ID.particle'].append(current_snap_pID_list)
    #
    return(infall_information_dict)
#
def initialize_halo_tracking_FIRE(BH_parameters, sim_num, out_f):
    '''
    * This function initializes subhalo tracking by retrieving the particle ID data for identified infalling subhalos.
    '''
    # Read in the infalling subtree data for the given simulation number.
    print("# Reading in infalling subhalo data identified by infalling_subhalo_criteria.py #", flush=True, file=out_f)
    infall_subtree_df = read_in_infalling_subtree_data_FIRE(BH_parameters, sim_num, 'subtree', out_f)
    infall_tree_df = read_in_infalling_subtree_data_FIRE(BH_parameters, sim_num, 'tree', out_f)
    # Merge the two dataframes.
    infall_subhalo_df = pd.concat([infall_subtree_df, infall_tree_df])
    #
    # Remove broken-link subhalos.
    cleaned_infall_subtree_df = remove_incomplete_subtrees_FIRE(infall_subhalo_df, BH_parameters, out_f)
    #
    # Get ID.subtree and ID.halo.infall arrays organized according to the infall time.
    infall_information_dict = get_infall_information(cleaned_infall_subtree_df)
    #
    t_s_step = time.time()
    print("# Getting the halo particle data for infalling subhalos #", flush=True, file=out_f)
    infall_information_dict = get_infall_particle_IDs(infall_information_dict, BH_parameters, sim_num, out_f)
    t_e_step = time.time()
    utilities.print_time_taken(t_s_step, t_e_step, "#", True, out_f)
    print("", flush=True, file=out_f)
    #
    # Return the infalling subhalo data dictionary.
    return(infall_information_dict)
#
def initialize_halo_tracking(BH_parameters, sim_num, out_f):
    '''
    * This function initializes subhalo tracking by retrieving the particle ID data for identified infalling subhalos.
    '''
    # Read in the infalling subtree data for the given simulation number.
    print("# Reading in infalling subhalo data identified by infalling_subhalo_criteria.py #", flush=True, file=out_f)
    infall_subtree_df = read_in_infalling_subtree_data(BH_parameters, sim_num, out_f)
    #
    # Remove broken-link subhalos (and unmatched halos if halo matching between two Rockstar sets was done).
    cleaned_infall_subtree_df = remove_incomplete_subtrees(infall_subtree_df, BH_parameters, out_f)
    #
    # Get ID.subtree and ID.halo.infall arrays organized according to the infall time.
    infall_information_dict = get_infall_information(cleaned_infall_subtree_df)
    #
    t_s_step = time.time()
    print("# Getting the halo particle data for infalling subhalos #", flush=True, file=out_f)
    infall_information_dict = get_infall_particle_IDs(infall_information_dict, BH_parameters, sim_num, out_f)
    t_e_step = time.time()
    utilities.print_time_taken(t_s_step, t_e_step, "#", True, out_f)
    print("", flush=True, file=out_f)
    #
    # Return the infalling subhalo data dictionary.
    return(infall_information_dict)
#
def initialize_snapshot_data(sim_num, snap_num, base_dir, sim_types, use_argsort, out_f):
    '''
    * This function uses the SnapshotData class from utilities.py to read in the snapshot data.
    - Need to add a script that checks if the snapshot number is below 38.
      - For <38, snap_disk = snap_dmo!
    '''
    snapshot_data_dict = {}
    #
    # For Phat ELVIS, Disk simulations are identical to DMO simulations for snapshots below 38.
    # So if snap_num < 38, set 'disk' to 'dmo'.
    # I am not sure if this is the best way as this would read in the same DMO data twice, but let's go with it for now.
    if snap_num < 38:
        for i in range(len(sim_types)):
            sim_types[i] = 'dmo'
    #
    if len(sim_types) == 1:
        if sim_types[0] == 'disk':
            blocks=1
        elif sim_types[0] == 'dmo':
            if sim_num == 493:
                blocks = 1
            else:
                blocks = 8
        #
        # Initialize the snapshot dictionary-class.
        snapshot_data_dict[sim_types[0]] = utilities.SnapshotData(sim_num, snap_num, base_dir, sim_types[0], blocks)
    else:
        for i in range(len(sim_types)):
            if sim_types[i] == 'disk':
                blocks=1
            elif sim_types[i] == 'dmo':
                if sim_num == 493:
                    blocks = 1
                else:
                    blocks = 8
            #
            # Initialize the snapshot dictionary-class.
            snapshot_data_dict[sim_types[i]] = utilities.SnapshotData(sim_num, snap_num, base_dir, sim_types[i], blocks)
    #
    for i in range(len(snapshot_data_dict)):
        #
        # Read in the simulation snapshot output data.
        print(f"* Reading in the simulation output data: snapshot {snap_num}, {sim_types[i]}... ", end="", flush=True, file=out_f)
        t_s_step = time.time()
        snapshot_data_dict[sim_types[i]].read_in_snapshot_data(use_argsort)
        t_e_step = time.time()
        utilities.print_time_taken(t_s_step, t_e_step, "*" ,True, out_f)
        #
        # Read in the particle ID sort index data, if it exists.
        if use_argsort:
            print(f"* Reading in the particle ID sort index data: snapshot {snap_num}, {sim_types[i]}... ", end="", flush=True, file=out_f)
            t_s_step = time.time()
            snapshot_data_dict[sim_types[i]].read_in_pID_argsort_data()
            t_e_step = time.time()
            utilities.print_time_taken(t_s_step, t_e_step, "*" ,True, out_f)
    #
    return(snapshot_data_dict)
#
def track_particles(snapshot_obj, halo_pID, use_argsort):
    '''
    * This function finds particles in the given halo_pID data from the given snapshot_obj data.
    '''
    if use_argsort==True:
        # If the particle ID argsort data exists, use it to retrieve snapshot's particles IDs that correspond to the given halo.
        # It's much faster to use it.
        pID_idx_in_snap = snapshot_obj["pID.sort_idx"][halo_pID]
    else:
        # If the particle ID argsort data doesn't exist, look for the particle IDs using numpy.isin()
        pID_idx_in_snap = np.nonzero(np.isin(snapshot_obj["ID.particle"], halo_pID))[0]
    #
    # Get coordinates and velocities for the particles.
    tracked_coords = snapshot_obj["Coordinates"][pID_idx_in_snap]
    tracked_vels = snapshot_obj["Velocities"][pID_idx_in_snap]
    #
    # Return results as a list.
    return([tracked_coords, tracked_vels])
#
def remove_odd_pIDs(hID, snapshot_obj, halo_pID, use_argsort, out_f):
    '''
    * For some odd reason, there are very rare cases where pID_arr contains IDs that are 
      greater than the total number of particles in the snapshot.
      Check it and remove those pIDs from pID_arr.
      Also print their halo ID!
    '''
    if use_argsort==True:
        snapshot_num_particles = len(snapshot_obj["pID.sort_idx"])
    else:
        snapshot_num_particles = len(snapshot_obj["ID.particle"])
    #
    pID_check_mask = np.where(halo_pID > snapshot_num_particles)[0]
    if len(pID_check_mask) > 0:
        print(f"  * Attention! Halo {hID}: particle IDs from Rockstar contains IDs that are greater then the total number of particles in the snapshot {snapshot_num_particles}", flush=True, file=out_f)
        print(f"  * These particles will be removed from the analysis!", flush=True, file=out_f)
        print(f"  * Number of halo particles from Rockstar: {len(halo_pID)}", flush=True, file=out_f)
        print(f'  * Number of "BAD" particles: {len(pID_check_mask)}', flush=True, file=out_f)
        halo_pID = np.delete(halo_pID, pID_check_mask)
    return(halo_pID)
#
def subhalo_tracking_wrapper_function(BH_parameters, sim_num, infall_information_dict, out_f):
    '''
    * This is a wrapper function for performing subhalo particle tracking for all subhalos.
    '''
    use_argsort = BH_parameters['pID_argsort_made'] # 0: false, 1: true, making it a variable for readability.
    #
    # Create an array of snapshot numbers to use.
    # Think about forward tracking and backward tracking: skip this for now.
    last_snapnum = BH_parameters['last_snapnum']
    infall_snapnums = infall_information_dict["snapshot.infall"]
    first_infalling_snap = np.min(infall_snapnums)
    last_infalling_snap = np.max(infall_snapnums)
    track_snapnums = np.arange(np.min(infall_snapnums), last_snapnum+1, 1)
    print("* Infall snapshot numbers:", flush=True, file=out_f)
    print(f"  {infall_snapnums}", flush=True, file=out_f)
    print("* Snapshot numbers to use for tracking:", flush=True, file=out_f)
    print(f"  {track_snapnums}", flush=True, file=out_f)
    print("", flush=True, file=out_f)
    #
    infall_hid_list = infall_information_dict["ID.halo.infall"]
    infall_halo_pID_list = infall_information_dict['ID.particle']
    #
    # Perform halo tracking at each snapshot: retrieves the coordinates and velocities of halo particles at each snapshot.
    num_subs_tracking_started = 0
    num_subs_last_infall_snap = 0
    for i in range(len(track_snapnums)):
        t_s_snap = time.time()
        current_snap = track_snapnums[i]
        print(f"# Current snapshot: {current_snap} #", flush=True, file=out_f)
        #
        # Initialize the snapshot dictionary which contains a snapshot dictionary-class for each element in BH_parameters['tracking_order'].
        snapshot_data_dict = initialize_snapshot_data(sim_num, current_snap, BH_parameters['base_dir'], BH_parameters['tracking_order'], use_argsort, out_f)
        #
        # For the first snapshot used, print some useful information.
        if i == 0:
            print("* Snapshot file information:", flush=True, file=out_f)
            print("  * Only for the first snapshot used", flush=True, file=out_f)
            print("  * Only for the first block, if multiple blocks", flush=True, file=out_f)
            for key in snapshot_data_dict:
                snapshot_data = snapshot_data_dict[key]
                print(f"  * File path/name: {snapshot_data['file.path']}", flush=True, file=out_f)
                snapshot_data.print_header(out_f)
                print("", flush=True, file=out_f)
        #
        if current_snap == last_infalling_snap:
            print(f"*** Snapshot {current_snap} is the last snapshot with infalling subhalos! ***", flush=True, file=out_f)
        #
        if current_snap in infall_snapnums:
            # There are infalling subhalos in the current snapshot.
            infall_idx = np.where(infall_snapnums == current_snap)[0][0]
            infall_snap = infall_snapnums[infall_idx]
            current_snap_infalling_hids = infall_hid_list[infall_idx]
            current_snap_infalling_halos_pID_list = infall_halo_pID_list[infall_idx]
            #
            # Update the number of subhalos that have already started tracking.
            num_subs_tracking_started += num_subs_last_infall_snap
            num_subs_last_infall_snap = len(current_snap_infalling_hids)
            #
            # Infall snapshot numbers that have been used already, excluding current_snap.
            infall_snapnums_used = infall_snapnums[:infall_idx]
            #
            # Tracking halos infalling at the current snapshot.
            print(f"  * Tracking {len(current_snap_infalling_hids)} subhalo(s) infalling at the current snapshot ({infall_snap})...", flush=True, file=out_f)
            t_s_step = time.time()
            for j in range(len(current_snap_infalling_hids)):
                # Get halo ID and particle IDs for the current halo.
                current_hID = current_snap_infalling_hids[j]
                current_halo_pIDs = current_snap_infalling_halos_pID_list[j]
                #
                # Tracking halo particles.
                for key in snapshot_data_dict:
                    # Remove particle IDs that are larger than the total number of particles in the snapshot.
                    # This is weird, but they exist for a very rare number of cases.
                    current_halo_pIDs = remove_odd_pIDs(current_hID, snapshot_data_dict[key], current_halo_pIDs, use_argsort, out_f)
                    infall_information_dict['ID.particle'][infall_idx][j] = current_halo_pIDs
                    # Get halo particles' coordinates and velocities at the current snapshot.
                    tracked_particle_data = track_particles(snapshot_data_dict[key], current_halo_pIDs, use_argsort)
                    #
                    # Save tracked particle data in a .hdf5 file.
                    utilities.output_halo_particles_hdf5(BH_parameters["tracked_halo_particle_dir"], sim_num, current_hID, infall_snap, [current_snap], current_halo_pIDs, [tracked_particle_data], key, out_f)
            #
            t_e_step = time.time()
            utilities.print_time_taken(t_s_step, t_e_step, "  *", True, out_f)
        #
        else:
            # There are no infalling subhalos in the current snapshot.
            print("  * There are no infalling subhalos at the current snapshot!", flush=True, file=out_f)
            #
            # Infall snapshot numbers that have been used already, excluding current_snap.
            infall_snapnums_used = infall_snapnums[:infall_idx+1]
            #
            # Update the number of subhalos that have already started tracking.
            num_subs_tracking_started += num_subs_last_infall_snap
            # If there are consecutive snapshots with no infalling subhalos, we do not want the count
            # of num_subs_tracking_started to increase after the first one.
            num_subs_last_infall_snap = 0
        #
        # Track subhalos that infalled already and started their tracking at an earlier snapshot (i.e. infall_snapnums_used).
        t_s_step = time.time()
        print(f"  * Tracking {num_subs_tracking_started} subhalos that started tracking at earlier snapshots: t_infall < t_current... ", flush=True, file=out_f)
        print(f"    * {infall_snapnums_used}", flush=True, file=out_f)
        for j in range(len(infall_snapnums_used)):
            earlier_infall_snap = infall_snapnums_used[j]
            earlier_snap_infalling_hids = infall_hid_list[j]
            earlier_snap_infalling_halos_pID_list = infall_halo_pID_list[j]
            for k in range(len(earlier_snap_infalling_hids)):
                # Get halo ID and particle IDs for the current halo.
                current_hID = earlier_snap_infalling_hids[k]
                current_halo_pIDs = earlier_snap_infalling_halos_pID_list[k]
                # Tracking halo particles.
                for key in snapshot_data_dict:
                    # Get halo particles' coordinates and velocities at the current snapshot.
                    tracked_particle_data = track_particles(snapshot_data_dict[key], current_halo_pIDs, use_argsort)
                    #
                    # Save tracked particle data in a .hdf5 file.
                    utilities.output_halo_particles_hdf5(BH_parameters["tracked_halo_particle_dir"], sim_num, current_hID, earlier_infall_snap, [current_snap], current_halo_pIDs, [tracked_particle_data], key, out_f)
        #
        t_e_step = time.time()
        utilities.print_time_taken(t_s_step, t_e_step, "  *", True, out_f)
        #
        t_e_snap = time.time()
        utilities.print_time_taken(t_s_snap, t_e_snap, "#", True, out_f)
        print("", flush=True, file=out_f)
#
def subhalo_analysis_wrapper_function(BH_parameters, sim_num, snapnum_info_dict, out_f):
    '''
    * This is a wrapper function for performing subhalo analysis for all subhalos.
    - Need file paths for
      - host tree main branch file: DMO and Disk
      - DMO surviving halo main branch file
      - DMO subtree main branch file
      - Disk subtree file with subhalos found by using the infall criteria
    '''
    # Make the file names for host halo main tree data: host_halo_dat_fnames will also contain file names for other data, but these will not be used.
    host_halo_dat_fnames = utilities.various_halo_file_names(BH_parameters['base_dir'], sim_num)
    # Directory path for tracked halo particles.
    tracked_halo_particle_dir = BH_parameters["tracked_halo_particle_dir"]
    #
    sim_types = BH_parameters["tracking_order"]
    #
    #
    halo_particle_file_info_dict = {}
    #
    ### Test ###
    halo_list = []
    ### ###
    for i in range(len(sim_types)):
        sim_type = sim_types[i]
        # Read in the host halo main branch data.
        # Everything is read in with non-h units.
        host_halo_dict = utilities.read_in_host_main_branch_file(host_halo_dat_fnames, sim_num, sim_type, BH_parameters)
        print()
        print(f"sim_type: {sim_type}")
        # Directory path to all tracked halo particle data for the current simulation number and type
        dpath = f"{tracked_halo_particle_dir}/{sim_type}/{sim_num}"
        print(dpath, flush=True, file=out_f)
        hID_arr, infall_snap_arr, full_name_arr = utilities.get_halo_particle_file_names_in_dir(dpath)
        print(f"{sim_type}, {len(hID_arr)}, {len(infall_snap_arr)}, {len(full_name_arr)}", flush=True, file=out_f)
        #
        print(len(hID_arr))
        for j in range(len(hID_arr[:5])):
            current_hID = hID_arr[j]
            current_infall_snap = infall_snap_arr[j]
            current_fname = full_name_arr[j]
            ### Test ###
            current_halo = halo_analysis.analyze_halo(current_hID, current_infall_snap, current_fname, host_halo_dict, snapnum_info_dict, BH_parameters, out_f)
            halo_list.append(current_halo)
            ### ###
            #halo_analysis.analyze_halo(current_hID, current_infall_snap, current_fname, snapnum_info_dict, BH_parameters, out_f)
    ### Test ###
    return(halo_list)
    ### ###
    
#
#-----------------------------------------------------------------------------------
# Main function
#-----------------------------------------------------------------------------------
def main():
    t_s = time.time()
    #
    # Initialization process.
    # BH_initialization defines two global variables (COSMO and T0), in addition to returned variables.
    header_statement = "Bloodhound subhalo tracking progress report"
    BH_parameters, sim_nums, base_dir, out_f, snapnum_info_dict = BH_initialization(parameter_fname, header_statement)
    '''
    * Step 1: Identifying infalling subhalos
    '''
    #
    # Check whether infalling subhalos have been identified or not.
    subhalo_selection_done = BH_parameters['subhalo_selection_done']
    if subhalo_selection_done == 0:
        # If it hasn't been done yet, do it now.
        t_s_step = time.time()
        print("####### Infalling subhalos have not been identified yet! #######", flush=True, file=out_f)
        print("* Doing this step first!", flush=True, file=out_f)
        print("* Its progress will be recorded in a separate text file.", flush=True, file=out_f)
        infall_subhalo_criteria.main()
        t_e_step = time.time()
        utilities.print_time_taken(t_s_step, t_e_step, "#######", True, out_f)
        print("", flush=True, file=out_f)
    elif subhalo_selection_done == 1:
        print("####### Infalling subhalos have already been identified, so skip this step! #######", flush=True, file=out_f)
        print("", flush=True, file=out_f)
    '''
    * Step 2: subhalo particle tracking
    '''
    #
    # Check whether subhalo particle tracking has already been done or not.
    halo_particle_tracking_done = BH_parameters['halo_particle_tracking_done']
    if halo_particle_tracking_done == 0:
        t_s_tracking = time.time()
        print("####### Starting subhalo tracking, one simulation at a time #######", flush=True, file=out_f)
        print("", flush=True, file=out_f)
        for sim_num in sim_nums:
            t_s_sim = time.time()
            print(f"##### Simulation {sim_num} #####", flush=True, file=out_f)
            print("", flush=True, file=out_f)
            #
            # Initialize halo tracking.
            t_s_step = time.time()
            print("### Initializing halo tracking ###", flush=True, file=out_f)
            if BH_parameters['simulation_name'] == 'pELVIS':
                infall_information_dict = initialize_halo_tracking(BH_parameters, sim_num, out_f)
            elif BH_parameters['simulation_name'] == 'FIRE':
                infall_information_dict = initialize_halo_tracking_FIRE(BH_parameters, sim_num, out_f)
            t_e_step = time.time()
            print("### Initializing halo tracking finished! ###", flush=True, file=out_f)
            utilities.print_time_taken(t_s_step, t_e_step, "###", True, out_f)
            print("", flush=True, file=out_f)
            #
            # Perform subhalo tracking.
            t_s_step = time.time()
            print("### Tracking subhalos: one snapshot at a time ###", flush=True, file=out_f)
            subhalo_tracking_wrapper_function(BH_parameters, sim_num, infall_information_dict, out_f)
            t_e_step = time.time()
            print("### Subhalo tracking for current simulation finished! ###", flush=True, file=out_f)
            utilities.print_time_taken(t_s_step, t_e_step, "###", True, out_f)
            print("", flush=True, file=out_f)
            #
            #
            t_e_sim = time.time()
            print(f"##### Simulation {sim_num} finished! #####", flush=True, file=out_f)
            utilities.print_time_taken(t_s_sim, t_e_sim, "#####", True, out_f)
            print("", flush=True, file=out_f)
        t_e_tracking = time.time()
        print("####### Subhalo tracking finished! #######", flush=True, file=out_f)
        utilities.print_time_taken(t_s_tracking, t_e_tracking, "#######", True, out_f)
        print("", flush=True, file=out_f)
        #
    elif halo_particle_tracking_done == 1:
        print("####### Infalling subhalo particle tracking is already done, so skip this step! #######", flush=True, file=out_f)
        print("", flush=True, file=out_f)
    '''
    * Step 3: subhalo analysis
    '''
    print("####### Computing subhalo properties using tracked particles #######", flush=True, file=out_f)
    print("", flush=True, file=out_f)
    #
    '''
    Steps:
    - Do subhalo analysis
    - I want to be able to do backward (in time) tracking as well!
    Need file paths for
      - host tree main branch file: DMO and Disk
      - DMO surviving halo main branch file
      - DMO subtree main branch file
      - Disk subtree file with subhalos found by using the infall criteria
    Flow: think about it a little more!
      - Use a subhalo analysis initialize function to initialize subhalo analysis.
      - Then use a halo analysis wrapper function in the halo_analysis module to analyze one halo at a time.
    '''
    t_s_analysis = time.time()
    test_result_list = []
    for sim_num in sim_nums:
        ### Test ###
        test_result = subhalo_analysis_wrapper_function(BH_parameters, sim_num, snapnum_info_dict, out_f)
        test_result_list.append(test_result)
        ### ###
    # Use the halo analysis wrapper function in the halo_analysis module to analyze each halo.
    #halo_analysis.analyze_halo(BH_parameters)
    t_e_analysis = time.time()
    print("####### Subhalo analysis finished! #######", flush=True, file=out_f)
    utilities.print_time_taken(t_s_analysis, t_e_analysis, "#######", True, out_f)
    #
    #
    # Print the total execution time and close the output statement file.
    print("", flush=True, file=out_f)
    t_e = time.time()
    print(f"########## Total execution time: {t_e - t_s:.03f} s ##########", flush=True, file=out_f)
    #
    out_f.close()
    return(test_result_list)
#
if __name__ == "__main__":
    main()