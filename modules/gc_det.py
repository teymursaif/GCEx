import os, sys
import math
import numpy as np
import matplotlib.pyplot as plt
from astroquery.mast import Observations
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from astropy.stats import sigma_clip
from astropy.visualization import *
from astropy.visualization import make_lupton_rgb
from astropy.table import Table, join_skycoord
from astropy import table
import photutils
from photutils.aperture import CircularAperture
from photutils.aperture import CircularAnnulus
from photutils.centroids import centroid_1dg, centroid_2dg, centroid_com, centroid_quadratic
#from photutils.utils import CutoutImage
import time as TIME
from modules.initialize import *
from modules.pipeline_functions import *

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def select_gcs_for_param(param_sources,mag_sources,param_det_art_gcs,mag_det_art_gcs):

    N = len(param_sources)

    for i in range(N):

        mag = mag_sources[i]
        param = param_sources[i]

        param_lower_limit = 0   
        param_upper_limit = 1

        if (param > param_lower_limit) and (param < param_upper_limit) :
            mask.append(1)
        else :
            mask.append(1)

    return mask

def select_GC_candidadates(gal_id):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]

    fn_det = filters[0]
    ART_GCs_cat = art_dir+gal_name+'_'+fn_det+'_ALL_ART_GCs.fits'
    DET_ART_GCs_cat = art_dir+gal_name+'_'+fn_det+'_ALL_DET_ART_GCs.fits'
    source_cat = cats_dir+gal_name+'_master_cat_forced.fits'

    art_gcs = (fits.open(ART_GCs_cat))[1].data
    det_art_gcs = (fits.open(DET_ART_GCs_cat))[1].data
    sources = (fits.open(source_cat))[1].data

    selected_gcs_mask = np.ones(len(sources))

    for param in GC_SEL_PARAMS:
        
        mag_param = 'APER_MAG_CORR_'+fn_det

        param_art_gcs = art_gcs[param]
        param_det_art_gcs = det_art_gcs[param]
        param_sources = sources[param]

        mag_art_gcs = art_gcs[mag_param]
        mag_det_art_gcs = det_art_gcs[mag_param]
        mag_sources = sources[mag_param]

        mask = select_gcs_for_param(param_sources,mag_sources,param_det_art_gcs,mag_det_art_gcs)

        selected_gcs_mask = selected_gcs_mask * mask
