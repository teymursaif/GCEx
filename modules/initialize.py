import os, sys
import numpy as np

def initialize_params() :
    print ('+ Initializing the pipeline')
    # Defining the working directories
    global working_directory,input_dir,output_dir,data_dir,main_data_dir,clean_data_dir,img_dir,sex_dir,fit_dir,plots_dir,\
    detection_dir,cats_dir,psfs_dir,art_dir,final_cats_dir,temp_dir,sbf_dir,psf_dir, check_plots_dir, external_dir
    # Getting the objcts and data info
    global gal_id, gal_name, ra, dec, distance, filters, comments, gal_params
    # Configuring the pipeline parameters
    global PRIMARY_FRAME_SIZE_ARCSEC, FRAME_SIZE_ARCSEC, GAL_FRAME_SIZE_ARCSEC, N_ART_GCS, N_SIM_GCs, PSF_IMAGE_SIZE, INSTR_FOV, COSMIC_CLEAN, \
    PHOTOM_APERS, FWHMS_ARCSEC, APERTURE_SIZE, PSF_REF_RAD_FRAC, BACKGROUND_ANNULUS_START, BACKGROUND_ANNULUS_TICKNESS, TARGETS, APERTURE_SIZE, \
    MAG_LIMIT_CAT, CROSS_MATCH_RADIUS_ARCSEC, GC_SIZE_RANGE, GC_MAG_RANGE, RATIO_OVERSAMPLE_PSF, \
    PIXEL_SCALES, ZPS, PRIMARY_FRAME_SIZE, FRAME_SIZE, GAL_FRAME_SIZE, EXPTIME, GAIN, GC_REF_MAG, \
    SE_executable,galfit_executable,swarp_executable

    ##################################################
    ##### PARAMETERS THAT USER NEEDS TO CONFIGURE


    WORKING_DIR = '/data/users/saifollahi/Euclid/ERO/'
    FRAME_SIZE_ARCSEC = 240 #cut-out size from the original frame for the general anlaysis (arcsec)

    # List of targets as a string with:
    # Object-ID Object name RA Dec Distance-in-Mpc List-of-filters comment
    # About list of filters: first filter is the detection filter (separated by ",")
    # comments: LSB,N,etc
    # (lines with # in the beginning will be skipped)
    # example: '1 DF44 195.2416667 +26.9763889 100 F814W,F475W,F606W'
    TARGETS = ['1 MATLAS2019 226.33460 +01.81282 25 F814W,F606W LSB,nN']
    #TARGETS = ['2 DWARF-MER-SIM 269.06658 +65.00640 20 VIS LSB,N']
    GC_REF_MAG = {'F814W':-8,'F606W':-7.5}

    # defining the executables (what you type in the command-line that executes the program)
    SE_executable = 'sex'
    swarp_executable = 'swarp'

    ##################################################
    ### MORE ADVANCED PARAMETERS
    ### (!!! DO NOT CHANGE UNLESS YOU KNOW WHATYOU ARE DOING)
    working_directory = WORKING_DIR
    input_dir = working_directory+'inputs/'
    output_dir = working_directory+'outputs/'
    main_data_dir = input_dir+'main_data/'
    data_dir = input_dir+'data/'
    psf_dir = input_dir+'psf/'
    clean_data_dir = output_dir+'clean_data/'
    img_dir = output_dir+'img/'
    sex_dir = output_dir+'sex/'
    fit_dir = output_dir+'fit/'
    plots_dir = output_dir+'plots/'
    detection_dir = output_dir+'detection/'
    cats_dir = output_dir+'cats/'
    psfs_dir = output_dir+'psfs/'
    art_dir = output_dir+'artificial/'
    final_cats_dir = output_dir+'final_cats/'
    temp_dir = output_dir+'temp_files/'
    sbf_dir = output_dir+'sbf/'
    check_plots_dir = output_dir+'check_plots/'
    external_dir = input_dir+'external/'

    galfit_executable = external_dir+'galfit'

    PRIMARY_FRAME_SIZE_ARCSEC = 1*FRAME_SIZE_ARCSEC #arcsec
    GAL_FRAME_SIZE_ARCSEC  = 1*FRAME_SIZE_ARCSEC #cut-out size from the original frame for sersic fitting anlaysis (arcsec)
    PHOTOM_APERS = '2,4,8,16,24,32' #aperture-sizes (diameters) in pixels for aperture photometry with Sextractor
    BACKGROUND_ANNULUS_START = 3 #The size of background annulus for forced photoemtry as a factor of FWHM
    BACKGROUND_ANNULUS_TICKNESS = 20 # the thickness of the background annulus in pixels
    CROSS_MATCH_RADIUS_ARCSEC = 0.25
    MAG_LIMIT_CAT = 26
    PSF_IMAGE_SIZE = 2 #radius in arcsec
    N_ART_GCS = 50
    N_SIM_GCs = 1
    COSMIC_CLEAN = False
    GC_SIZE_RANGE = [1,5]
    GC_MAG_RANGE = [-9,-5]
    #RATIO_OVERSAMPLE_PSF = 5

    for dir in [working_directory,input_dir,output_dir,data_dir,main_data_dir,clean_data_dir,img_dir,sex_dir,fit_dir,plots_dir,\
    detection_dir,cats_dir,psfs_dir,art_dir,final_cats_dir,temp_dir,sbf_dir,psf_dir,check_plots_dir] :
        if not os.path.exists(dir): os.makedirs(dir)

    gal_params = {}
    for line in TARGETS:
        line = line.split(' ')
        gal_id, gal_name, ra, dec, distance, filters, comments = int(line[0]), str(line[1]), \
        float(line[2]), float(line[3]), float(line[4]), np.array(line[5].split(',')), np.array((line[6].split('\n')[0]).split(','))
        gal_params[gal_id] = [gal_name, ra, dec, distance, filters, comments]
        #print (gal_name, ra, dec, distance, filters, comments)

    # if no PSF is given to the pipeline, FWHM and flux-correction value in each filter must be given to the pipeline
    FWHMS_ARCSEC = {}
    APERTURE_SIZE = {}
    PSF_REF_RAD_FRAC = {}
    PIXEL_SCALES = {}
    ZPS = {}
    PRIMARY_FRAME_SIZE = {}
    FRAME_SIZE = {}
    GAL_FRAME_SIZE = {}
    EXPTIME = {}
    GAIN = {}

############################################################

def welcome():

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

    #print (f"\n{bcolors.OKCYAN}   *****************************************"+ bcolors.ENDC)
    print (f"{bcolors.OKCYAN} \n+ GCTOOLS (version: August 2023) "+ bcolors.ENDC)
    print (f"+ Developed by Teymoor Saifollahi "+ bcolors.ENDC)
    print (f"+ Kapteyn Astronomical Institute"+ bcolors.ENDC)
    print (f"+ contact: saifollahi@astro.rug.nl\n"+ bcolors.ENDC)
    #print (f"{bcolors.OKCYAN}   *****************************************\n"+ bcolors.ENDC)

welcome()
initialize_params()
