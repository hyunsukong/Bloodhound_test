'''
***** Load my environment first: conda activate my_env *****

* This script runs Bloodhound.
'''
####################################################################################
# Import libraries.
####################################################################################
from astropy.cosmology import FlatLambdaCDM
from astropy.modeling.physical_models import NFW
import h5py
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import subprocess
import time
####################################################################################
# Import my modules/libraries/scripts.
####################################################################################
from from_stampede2 import rockstar_handling as rh
import infall_subhalo_criteria
import utilities
####################################################################################
# Functions
# MAKE SURE THE MOVE THESE TO ANOTHER FILE! I JUST WANT THE MAIN FUNCTION HERE.
####################################################################################
def are_subhalos_identified(subhalos_found_param):
    if subhalos_found_param == 0 or subhalos_found_param == False:
        print("* Infalling subhalos have not been identified yet.")
        print("* Running infall_subhalo_criteria.py now...")
    #
    elif subhalos_found_param == 1 or subhalos_found_param == True:
        print("* Infalling subhalos have already been identified.")
    #
    else:
        print("***** Parameter 'subhalos_found' can only take True or False (or 1 or 0). Something else was given.")
        print("***** Exiting the execution...")
        sys.exit()
####################################################################################
# The main function
####################################################################################
def main():
    '''
    * Flow:
    - Read-in parameters.
    - For each simulation number specified in sim_num,
    - 
    '''
    #
    # Read-in Bloodhound parameters.
    parameter_fname = '/scratch/05097/hk9457/pELVIS/z13/BH_input_data/bloodhound_parameters.txt'
    parameters = utilities.read_parameters(parameter_fname)
    h=parameters['h']
    print(parameters)
    print()
    #
    # Read-in snapshot number, redshift, scale factor data.
    snapnum_info_fname = parameters["snapnum_info_fname"]
    snapnum_info_dict = utilities.open_snap_header_file(snapnum_info_fname)
    #
    # Check whether infalling subhalos have already been identified or not. If not, do this step first.
    are_subhalos_identified(parameters["subhalos_found"])
    '''
    if parameters["subhalos_found"] == 0 or parameters["subhalos_found"] == False:
        print("* Infalling subhalos have not been identified yet.")
        print("* Running infall_subhalo_criteria.py now...")
    #
    elif parameters["subhalos_found"] == 1 or parameters["subhalos_found"] == True:
        print("* Infalling subhalos have already been identified.")
    #
    else:
        print("***** Parameter 'subhalos_found' can only take True or False (or 1 or 0). Something else was given.")
        print("***** Exiting the execution...")
        sys.exit()
    '''
    print("Hi!")
#
#
if __name__ == "__main__":
    main()