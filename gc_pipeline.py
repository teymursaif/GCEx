
import os, sys
import warnings

#warnings.filterwarnings('ignore')

from modules.initialize import *
from modules.pipeline_functions import *
from modules.plots_functions import *
from modules.source_det import *
from modules.fit_galaxy import *
from modules.psf import *
#from modules.sbf import *

# running the pipeline
for gal_id in gal_params.keys():

    # step 0. prepare data
    intro(gal_id)
    copy_data(gal_id)
    resample_data(gal_id) #compulsary
    make_galaxy_frames(gal_id, resampled=True)

    # step 1. fit sersic and subtract the light
    fit_galaxy_sersic_all_filters(gal_id)

    # step 2. detection, photometry and make the main source catalogue
    make_source_cat(gal_id)
    make_multiwavelength_cat(gal_id, mode='forced-photometry')

    # step 3. GC analysis: completeness, selection, measurments
    #measure_completeness(gal_id)
    #select_GC_candidates(gal_id)
    #measure_GC_properties(gald_id)
