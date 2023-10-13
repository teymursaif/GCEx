import os, sys
import warnings

copy_initialize_file_command = 'cp '+str(sys.argv[1])+' ./modules/initialize.py'
os.system(copy_initialize_file_command)
#print (sys.argv)

from modules.initialize import *
from modules.pipeline_functions import *
from modules.plots_functions import *
from modules.source_det import *
from modules.fit_galaxy import *
from modules.psf import *
from modules.gc_det import *

for gal_id in gal_params.keys():

    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    methods = gal_methods[gal_id]

    # step 0. inistialize pipleine and prepare data
    intro(gal_id)
    if 'PREPARE_DATA' in sys.argv:
        copy_data(gal_id)

    get_data_info(gal_id)

    if 'RESAMPLE'in methods:
        resample_data(gal_id)
    
    if 'PREPARE_FRAME' in sys.argv:
        make_galaxy_frames(gal_id)

    if 'MODEL_PSF' in methods:
        make_psf_all_filters(gal_id)

    initial_psf(gal_id)

    # step 1. fit sersic and subtract the light
    if 'FIT_GAL' in methods:
        fit_galaxy_sersic_all_filters(gal_id)

    # step 2. detection, photometry and make the main source catalogue
    if 'MAKE_CAT' in methods:
        make_source_cat_full(gal_id)
        #assess_photom(gal_id)

    # step 3. GC analysis: completeness, selection, measurments
    if 'SIM_GC' in methods:
        simulate_GCs_all(gal_id)
        make_source_cat_for_sim(gal_id)
        assess_GC_simulations(gal_id)

    # step 4. GC selection, assessment, GC catalogs and properties
    if 'MAKE_GC_CAT' in methods:
        select_GC_candidadates(gal_id)
        #measure_GC_properties(gal_id)

    finalize(gal_id)
    # step FINALE

if (MERGE_CATS == True):
    merge_cats()

if (MERGE_SIM_GC_CATS == True):
    merge_sims()

if (MERGE_GC_CATS == True):
    merge_gc_cats()
    #select_GC_candidadates_merged()
    #measure_GC_properties_merged()
