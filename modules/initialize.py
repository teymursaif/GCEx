import os, sys
import numpy as np

def initialize_params() :
    print (f"{bcolors.OKCYAN}- Initializing the pipeline ... "+ bcolors.ENDC)
    # Defining the working directories
    global working_directory,input_dir,output_dir,data_dir,main_data_dir,clean_data_dir,img_dir,sex_dir,fit_dir,plots_dir,\
    detection_dir,cats_dir,psfs_dir,art_dir,final_cats_dir,temp_dir,sbf_dir,psf_dir, check_plots_dir, external_dir, data_dir_orig
    # Getting the objcts and data info
    global gal_id, gal_name, ra, dec, distance, filters, comments, gal_params
    # Configuring the pipeline parameters
    global PRIMARY_FRAME_SIZE_ARCSEC, FRAME_SIZE_ARCSEC, GAL_FRAME_SIZE_ARCSEC, N_ART_GCS, N_SIM_GCS, PSF_IMAGE_SIZE, INSTR_FOV, COSMIC_CLEAN, \
    PHOTOM_APERS, FWHMS_ARCSEC, APERTURE_SIZE, PSF_REF_RAD_FRAC, BACKGROUND_ANNULUS_START, BACKGROUND_ANNULUS_TICKNESS, TARGETS, APERTURE_SIZE, \
    MAG_LIMIT_CAT, CROSS_MATCH_RADIUS_ARCSEC, GC_SIZE_RANGE, GC_MAG_RANGE, RATIO_OVERSAMPLE_PSF, PSF_PIXEL_SCALE, PSF_SIZE, MODEL_PSF, \
    PIXEL_SCALES, ZPS, PRIMARY_FRAME_SIZE, FRAME_SIZE, GAL_FRAME_SIZE, EXPTIME, GAIN, GC_REF_MAG, PSF_PIXELSCL_KEY, FWHM_LIMIT, \
    SE_executable,galfit_executable,swarp_executable

    ##################################################
    ##### PARAMETERS THAT USER NEEDS TO CONFIGURE


    WORKING_DIR = './'
    FRAME_SIZE_ARCSEC = 1660 #cut-out size from the original frame for the general anlaysis (arcsec)
    GAL_FRAME_SIZE_ARCSEC  = 830 #cut-out size from the original frame for sersic fitting anlaysis (arcsec)
    #FRAME_SIZE_ARCSEC = 720

    # List of targets as a string with:
    # Object-ID Object name RA Dec Distance-in-Mpc List-of-filters comment
    # About list of filters: first filter is the detection filter (separated by ",")
    # comments: LSB,N,etc
    # (lines with # in the beginning will be skipped)
    # example: '1 DF44 195.2416667 +26.9763889 100 F814W,F475W,F606W'
    
    #TARGETS = ['1 MATLAS2019 226.33460 +01.81282 25 HST-ACS-F606W,HST-ACS-F814W LSB,nN']
    #GC_REF_MAG = {'HST-ACS-F606W':-7.5,'HST-ACS-F814W':-8.0}

    #JWST
    #TARGETS = ['1 CEERS-LSB1 214.8588333333 +52.7629166667 80 F115W,F150W,F200W,F277W,F356W,F444W LSB,SF']
    #GC_REF_MAG = {'F115W':-8.0,'F150W':-8.0,'F200W':-8.0,'F277W':-8.0,'F356W':-8.0,'F444':-8.0}

    #Euclid SIMS
    #TARGETS = ['1 EUC-SIM1 231.50075 +30.45227 20 VIS E,N']
    #TARGETS = ['2 DWARF-MER-SIM 269.06658 +65.00640 20 VIS LSB,N']
    #GC_REF_MAG = {'VIS':-8}

    #Euclid ERO
    #TARGETS = ['0 ERO-FORNAX 054.01542 -35.27031 20 VIS NA'] # too big to handle!
    TARGETS = ['1 ERO-FORNAX-1 54.29734653840737 -35.49463984609042 20 VIS1 NA',\
               '2 ERO-FORNAX-2 53.73781934462309 -35.4946397673397 20 VIS2 NA',\
               '3 ERO-FORNAX-3 54.29577802781735 -35.03909208984263 20 VIS3 NA',\
               '4 ERO-FORNAX-4 53.739388046497496 -35.03909201240899 20 VIS4 NA']
            
    GC_REF_MAG = {'VIS1':-8,'VIS2':-8,'VIS3':-8,'VIS4':-8}

    # defining the executables (what you type in the command-line that executes the program)
    #SE_executable = 'sex'
    #swarp_executable = 'swarp'
    SE_executable = 'sextractor'
    swarp_executable = 'SWarp'

    ##################################################
    ### MORE ADVANCED PARAMETERS
    ### (!!! DO NOT CHANGE UNLESS YOU KNOW WHATYOU ARE DOING)
    
    working_directory = WORKING_DIR
    
    #input_dir = working_directory+'inputs/'
    #output_dir = working_directory+'outputs/'
    #main_data_dir = input_dir+'main_data/'

    input_dir = '/home/teymoor/Astronomy/Euclid/ERO-data/inputs/'
    output_dir = '/home/teymoor/Astronomy/Euclid/ERO-data/outputs/'
    main_data_dir = '/home/teymoor/Astronomy/Euclid/ERO-data/inputs/main_data/'

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

    data_dir_orig = data_dir

    galfit_executable = external_dir+'galfit'

    PRIMARY_FRAME_SIZE_ARCSEC = 1*FRAME_SIZE_ARCSEC #arcsec
    PHOTOM_APERS = '1,2,4,6,8,10,12,16,20,24,32,40' #aperture-sizes (diameters) in pixels for aperture photometry with Sextractor
    BACKGROUND_ANNULUS_START = 3 #The size of background annulus for forced photoemtry as a factor of FWHM
    BACKGROUND_ANNULUS_TICKNESS = 20 # the thickness of the background annulus in pixels
    CROSS_MATCH_RADIUS_ARCSEC = 0.25
    MAG_LIMIT_CAT = 27
    N_ART_GCS = 250
    N_SIM_GCS = 2
    COSMIC_CLEAN = False
    GC_SIZE_RANGE = [1,8] #lower value should be small enough to make some point-sources for performance check, in pc
    GC_MAG_RANGE = [-10,-4]
    PSF_PIXELSCL_KEY = 'PIXELSCL'
    PSF_PIXEL_SCALE = 0.0 #if 'PIXELSCL' is not in the header, specify it here.

    ### for making PSF
    MODEL_PSF = True
    RATIO_OVERSAMPLE_PSF = 5 #do not go beyond 10, this will have consequences for undersampling later
    PSF_IMAGE_SIZE = 40 #PSF size in the instruments pixel-scale

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

def welcome():

    #print (f"\n{bcolors.OKCYAN}   *****************************************"+ bcolors.ENDC)
    print (f"{bcolors.OKCYAN} \n+ GCTOOLS (version: August 2023) "+ bcolors.ENDC)
    print (f"+ Developed by Teymoor Saifollahi "+ bcolors.ENDC)
    print (f"+ Kapteyn Astronomical Institute"+ bcolors.ENDC)
    print (f"+ contact: saifollahi@astro.rug.nl\n"+ bcolors.ENDC)
    #print (f"{bcolors.OKCYAN}   *****************************************\n"+ bcolors.ENDC)

############################################################

def finalize(gal_id):
    rm_keys = ['*.fits','*.log','galfit*','*.xml']
    for rm_key in rm_keys:
        os.system('rm '+rm_key)
    print (f"{bcolors.OKGREEN}- Everything is done for this objects."+ bcolors.ENDC)


############################################################

welcome()
initialize_params()
