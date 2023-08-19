
import os, sys
import warnings

#warnings.filterwarnings('ignore')

from modules.pipeline_functions import *
from modules.plots_functions import *
from modules.source_det import *
from modules.fit_galaxy import *
#from modules.sbf import *

# GC pipeline v2.0 (2023)
# Developed by Teymoor Saifollahi
# Kapteyn Astronomical Institute

# 1. Make sure data is prepared for the analysis by keeping them in "data_dir"
# parameter as is define in "modules/pipeline_functions". Frames for a given galaxy should be
# in the following format (as an example for HST data of galaxy DF44 in F606W):
# DF44_F606W_*.science.fits -> science frame (and stacked)
# DF44_F606W_*.weight.fits -> weight map associated to the science frame

# 2. Make sure you have PSF models stored in "psf_dir" parameter as is define in
# "modules/pipeline_functions". PSF model for a given filter should be given
# in the following format (as an example for HST F606W): psf_F606W.fits

# 3. Make sure that you have adjusted all the inputs in "modules/pipeline_functions".

# running the pipeline
for gal_id in gal_params.keys():

    # step 1. prepare data
    intro(gal_id)
    copy_data(gal_id)
    resample_data(gal_id) #compulsary
    make_galaxy_frames(gal_id, resampled=True)

    # step 2. fit sersic and subtract the light
    fit_galaxy_sersic_all_filters(gal_id)

    # step 3. detection, photometry and make the main source catalogue
    #make_source_cat(gal_id)
    #make_multiwavelength_cat(gal_id, mode='forced-photometry')

    # step 4. GC analysis: completeness, selection, measurments
    #measure_completeness(gal_id)
    #select_GC_candidates(gal_id)
    #measure_GC_properties(gald_id)
