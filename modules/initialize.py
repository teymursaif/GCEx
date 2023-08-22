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
    global PRIMARY_FRAME_SIZE_ARCSEC, FRAME_SIZE_ARCSEC, GAL_FRAME_SIZE_ARCSEC, N_ART_GCS, PSF_IMAGE_SIZE, INSTR_FOV, COSMIC_CLEAN, GAIN, \
    PHOTOM_APERS, FWHMS_ARCSEC, PSF_REF_RAD_ARCSEC, PSF_REF_RAD_FRAC, BACKGROUND_ANNULUS_START, BACKGROUND_ANNULUS_TICKNESS, TARGETS, \
    MAG_LIMIT_CAT, CROSS_MATCH_RADIUS_ARCSEC, \
    SE_executable,galfit_executable,swarp_executable

    ##################################################
    ##### PARAMETERS THAT USER NEEDS TO CONFIGURE

    working_directory = '/data/users/saifollahi/Euclid/ERO/'
    gal_input_cat = '/data/users/saifollahi/Euclid/ERO/udg_input.csv'
    FRAME_SIZE_ARCSEC = 240 #cut-out size from the original frame for the general anlaysis (arcsec)
    GAL_FRAME_SIZE_ARCSEC  = 120 #cut-out size from the original frame for sersic fitting anlaysis (arcsec)
    PHOTOM_APERS = '2,4,8,16,24,32' #aperture-sizes (diameters) in pixels for aperture photometry with Sextractor

    # List of targets as a string with:
    # Object-ID Object name RA Dec Distance-in-Mpc List-of-filters comment
    # About list of filters: first filter is the detection filter (separated by ",")
    # comments: LSB,N,etc
    # (lines with # in the beginning will be skipped)
    # example: '1 DF44 195.2416667 +26.9763889 100 F814W,F475W,F606W'
    TARGETS = ['1 MATLAS2019 226.33460 +01.81282 25 F814W,F606W,g,i LSB,nN']

    # defining the executables (what you type in the command-line that executes the program)
    SE_executable = 'sex'
    swarp_executable = 'swarp'

    ##################################################
    ### MORE ADVANCED PARAMETERS
    ### (!!! DO NOT CHANGE UNLESS YOU KNOW WHATYOU ARE DOING)

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
    BACKGROUND_ANNULUS_START = 3 #The size of background annulus for forced photoemtry as a factor of FWHM
    BACKGROUND_ANNULUS_TICKNESS = 20 # the thickness of the background annulus in pixels
    CROSS_MATCH_RADIUS_ARCSEC = 0.25
    MAG_LIMIT_CAT = 30
    PSF_IMAGE_SIZE_PIXELS = 100
    N_ART_GCS = 1000
    COSMIC_CLEAN = False

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
    FWHMS_ARCSEC = {'F606W':0.08, 'F814W':0.09,'g':1, 'i':1}
    PSF_REF_RAD_ARCSEC = {'F606W':0.16, 'F814W':0.2, 'g':0.8, 'i':0.8}
    PSF_REF_RAD_FRAC = {'F606W':0.8, 'F814W':0.8, 'g':0.8, 'i':0.8}
    GAIN = {'F606W':1.5, 'F814W':2, 'g':2.5, 'i':2.5}

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
