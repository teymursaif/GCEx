import os, sys
import numpy as np

def initialize_params() :
    print (f"{bcolors.OKCYAN}- Initializing the pipeline ... "+ bcolors.ENDC)
    # Defining the working directories
    global working_directory,input_dir,output_dir,data_dir,main_data_dir,clean_data_dir,img_dir,sex_dir,fit_dir,plots_dir,\
    detection_dir,cats_dir,psfs_dir,art_dir,final_cats_dir,temp_dir,sbf_dir,psf_dir, check_plots_dir, external_dir, data_dir_orig, sub_data_dir
    # Getting the objcts and data info
    global gal_id, gal_name, ra, dec, distance, filters, comments, gal_params, gal_methods, gal_data_name
    # Configuring the pipeline parameters
    global PRIMARY_FRAME_SIZE_ARCSEC, FRAME_SIZE_ARCSEC, GAL_FRAME_SIZE_ARCSEC, N_ART_GCS, N_SIM_GCS, PSF_IMAGE_SIZE, INSTR_FOV, COSMIC_CLEAN, \
    PHOTOM_APERS, FWHMS_ARCSEC, APERTURE_SIZE, PSF_REF_RAD_FRAC, BACKGROUND_ANNULUS_START, BACKGROUND_ANNULUS_TICKNESS, TARGETS, APERTURE_SIZE, \
    MAG_LIMIT_CAT, CROSS_MATCH_RADIUS_ARCSEC, GC_SIZE_RANGE, GC_MAG_RANGE, RATIO_OVERSAMPLE_PSF, PSF_PIXEL_SCALE, PSF_SIZE, MODEL_PSF, \
    PIXEL_SCALES, ZPS, PRIMARY_FRAME_SIZE, FRAME_SIZE, GAL_FRAME_SIZE, EXPTIME, GAIN, GC_REF_MAG, PSF_PIXELSCL_KEY, FWHM_LIMIT, INPUT_ZP, INPUT_EXPTIME, \
    MAG_LIMIT_SAT, MAG_LIMIT_PSF, GC_SEL_PARAMS, ELL_LIMIT_PSF, GC_SIM_MODE, MERGE_CATS, MERGE_SIM_GC_CATS, MERGE_GC_CATS, EXTRACT_DWARFS,\
    PARAM_SEL_METHOD, PARAM_SEL_RANGE, EXTERNAL_CROSSMATCH, EXTERNAL_CROSSMATCH_CAT
    global SE_executable,galfit_executable,swarp_executable

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
    INPUT_ZP = {}
    INPUT_EXPTIME = {}

    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    ##### PARAMETERS THAT USER NEEDS TO CONFIGURE

    #-------------------------- SETUP AND DATA PREPRATION --------------------------

    ### (if ZP, EXPTIME and GAIN are missing from the header, define them for a given filter)

    WORKING_DIR = './'
    PRIMARY_FRAME_SIZE_ARCSEC = 60 #arcsec
    FRAME_SIZE_ARCSEC = 60 #cut-out size from the original frame for the general anlaysis (arcsec)

    # defining the executables (what you type in the command-line that executes the program)
    SE_executable = 'sex'
    swarp_executable = 'swarp'
    #SE_executable = 'sextractor'
    #swarp_executable = 'SWarp'

    ### (if ZP, EXPTIME and GAIN are missing from the header, define them for a given filter)
    INPUT_ZP = {'VIS':30,'NISP-Y':30,'NISP-J':30,'NISP-H':30}
    INPUT_EXPTIME = {'VIS':565,'NISP-Y':121,'NISP-J':116,'NISP-H':81}
    INPUT_GAIN = {'VIS':2,'NISP-Y':1,'NISP-J':1,'NISP-H':1}

    # ------------------------------ GALAXIES/TARGETS ------------------------------

    # List of targets as a string with:
    # Object-ID Object name RA Dec Distance-in-Mpc List-of-filters comment
    # About list of filters: first filter is the detection filter (separated by ",")
    # comments: LSB,N,etc
    # (lines with # in the beginning will be skipped)
    # example: '1 DF44 195.2416667 +26.9763889 100 F814W,F475W,F606W'

    # HST
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
    TARGETS = []

    
    coords = [  [49.85203041666667, 42.10677777777777],\
                [49.71799875, 42.02483250000001],\
                [49.54738749999999, 42.031305833333334],\
                [49.62074041666666, 42.0425675],\
                [49.52060708333333, 42.009254444444444],\
                [49.81803624999999, 41.92738388888888],\
                [49.50566208333334, 41.826798333333336],\
                [49.468284583333336, 41.92698194444444],\
                [49.44762458333334, 41.79328111111111],\
                [49.53191166666667, 41.79473555555556],\
                [49.931839583333335, 41.713005555555554],\
                [49.88220708333333, 41.74505666666666],\
                [49.957474999999995, 41.66903194444444],\
                [49.96710958333333, 41.648158888888894],\
                [50.18777666666667, 41.71359027777777],\
                [50.029556666666664, 41.80436194444445],\
                [49.99816916666667, 41.81075222222222],\
                [49.353784999999995, 41.73928361111111],\
                [49.477501249999996, 41.71769166666666],\
                [49.18420208333333, 41.77249083333333],\
                [50.11558791666666, 41.62373361111111],\
                [50.004912499999996, 41.618896944444444],\
                [50.32273375, 41.59031944444445],\
                [50.32323291666666, 41.579314999999994],\
                [50.33499916666668, 41.54946555555556],\
                [50.235469583333334, 41.48937138888889],\
                [50.107512083333326, 41.51985694444445],\
                [50.10502291666667, 41.48415527777778],\
                [50.05839999999999, 41.50322694444444],\
                [49.60850583333333, 41.58960388888889],\
                [49.564410833333326, 41.476661666666665],\
                [49.56311791666667, 41.47668222222223],\
                [49.61458833333334, 41.49648277777778],\
                [49.74901291666667, 41.51044611111111],\
                [49.8025775, 41.51052111111111],\
                [49.94685125, 41.4814713888889],\
                [49.72702291666666, 41.22264166666667],\
                [49.740249999999996, 41.35898666666667],\
                [49.623790416666665, 41.30263027777778],\
                [49.452315, 41.4084075],\
                [49.331235, 41.677952222222224],\
                [49.176267083333336, 41.694655],\
                [49.09025458333333, 41.621990555555556],\
                [49.52851458333334, 41.640724444444444],\
                [50.101828749999996, 41.54338527777778],\
                [49.978704166666674, 41.61796138888889],\
                [50.010654166666676, 41.61427055555555],\
                [50.07407791666667, 41.60098444444444],\
                [50.083259583333344, 41.59210222222223],\
                [50.137605833333325, 41.55683611111111],\
                [50.09706916666666, 41.465561666666666],\
                [49.65419083333333, 41.29229722222222],\
                [49.70098291666667, 41.187785833333336],\
                [50.29034416666666, 41.61803638888889],\
                [49.88211458333333, 41.91444333333334],\
                [49.62916208333333, 41.632711111111114],\
                [50.032653333333336, 41.89375638888889],\
                [50.059477083333334, 41.73036333333334],\
                [50.042787916666676, 41.52692388888889],\
                [49.872615833333334, 41.620336944444446],\
                [49.69296583333334, 41.268159999999995],\
                [49.66474583333333, 41.51727305555556],\
                [49.723105, 41.530585],\
                [49.60768333333334, 41.54533472222222],\
                [49.77149708333333, 41.58009805555555],\
                [49.77935916666667, 41.710310555555566],\
                [49.618425833333326, 41.84444166666667],\
                [49.56439333333333, 41.406349999999996],\
                [49.47159666666666, 41.99441444444445],\
                [49.45511666666666, 41.81292638888889],\
                [49.49092041666667, 41.684511388888886],\
                [49.46139250000001, 41.40080527777778],\
                [49.32788291666667, 41.4788225],\
                [49.373932499999995, 41.71793638888889],\
                [49.37223416666667, 41.79393194444445],\
                [49.07775000000001, 41.597478611111114]]


    #coords = [[049.95896,+41.41584],[049.14865,+41.60580],[049.73977,+41.35899],[049.44806,+41.79297],[049.35376,+41.73917],[050.27769,+41.53715],[049.50575,+41.82677]]
    
    
    i = -1
    for coord in coords:
        i = i+1
        ra, dec = coord
        if (i>-1):
            #target_str = str(i) +' ERO-FORNAX FORNAX-DWARF-'+str(i)+' '+str(ra)+' '+str(dec)+' 20 VIS,NISP-Y,NISP-J,NISP-H MAKE_GC_CAT DWARF,LSB'
            target_str = str(i) +' ERO-PERSEUS PERSEUS-DWARF-'+str(i)+' '+str(ra)+' '+str(dec)+' 70 VIS,NISP-Y,NISP-J,NISP-H MAKE_CAT,MAKE_GC_CAT DWARF,LSB' #,NISP-Y,NISP-J,NISP-H
            TARGETS.append([target_str])
            #print (target_str)

    # NOTE: possible methods -> RESAMPLE_DATA, MODEL_PSF, FIT_GAL, USE_SUB_GAL, MAKE_CAT, MAKE_GC_CAT
    # NOTE: possible comments -> MASSIVE,DWARF,LSB

 
    MERGE_CATS = True
    MERGE_SIM_GC_CATS = False
    MERGE_GC_CATS = True

    global TABLES
    TABLES = {}

    # ------------------------------  GALAXY FITTING ------------------------------

    GAL_FRAME_SIZE_ARCSEC  = 1*FRAME_SIZE_ARCSEC  #cut-out size from the original frame for sersic fitting anlaysis (arcsec)

    # ---------------------- SOURCE DETECTION AND PHOTOMETRY ----------------------

    PHOTOM_APERS = '1,2,4,8,12,16,20,30,40' #aperture-sizes (diameters) in pixels for aperture photometry with Sextractor
    BACKGROUND_ANNULUS_START = 3 #The size of background annulus for forced photoemtry as a factor of FWHM
    BACKGROUND_ANNULUS_TICKNESS = 20 # the thickness of the background annulus in pixels
    CROSS_MATCH_RADIUS_ARCSEC = 0.25
    MAG_LIMIT_CAT = 28.0
    EXTRACT_DWARFS = False

    # -------------------------------- PSF MODELING -------------------------------

    ### if PSF is given by the user (in the "psf_dir" directory)
    PSF_PIXELSCL_KEY = 'PIXELSCL'
    PSF_PIXEL_SCALE = 0.0 #if 'PIXELSCL' is not in the header, specify it here.
    ### for making PSF (method=MODEL_PSF)
    MODEL_PSF = True
    RATIO_OVERSAMPLE_PSF = 10 #do not go beyond 10, this will have consequences for undersampling later
    PSF_IMAGE_SIZE = 40 #PSF size in the instruments pixel-scale
    MAG_LIMIT_PSF = 21
    MAG_LIMIT_SAT = 19
    ELL_LIMIT_PSF = 0.1
    #FWHM_UPPER_LIMIT_PSF =
    #FWHM_LOWER_LIMIT_PSF =


    #------------------------------ GC SIMULATION ------------------------------
    
    N_ART_GCS = 250
    N_SIM_GCS = 1
    COSMIC_CLEAN = False #does not work at the moment anyways...
    GC_SIZE_RANGE = [2,6] #lower value should be small enough to make some point-sources for performance check, in pc
    GC_MAG_RANGE = [-11,-5]
    GC_REF_MAG = {'VIS':-8, 'NISP-Y':-8.3,'NISP-J':-8.3,'NISP-H':-8.3} #magnitude of a typical GC in the given filters should be defined here.
    GC_SIM_MODE = 'UNIFORM' # 'UNIFORM' or 'CONCENTRATED'

    #------------------------------ GC SELECTION -------------------------------

    GC_SEL_PARAMS = ['CI_2_4','CI_4_8','CI_8_12']#,'CI_2_4','CI_4_6','CI_6_8','CI_8_10','CI_10_12','ELLIPTICITY']
    EXTERNAL_CROSSMATCH = False
    EXTERNAL_CROSSMATCH_CAT = './archival_tables/ERO-FDS-ugriJKs.fits'

    PARAM_SEL_METHOD = 'MANUAL'
    PARAM_SEL_RANGE = {'ELLIPTICITY':[-0.01,0.5],'F_MAG_APER_CORR_VIS':[15,30],'F_MAG_APER_CORR_NISP-Y':[15,30],\
    'color0':['VIS','VIS',-0.5,0.5],'color1':['VIS','NISP-Y',-1.5,1.2],'color2':['NISP-Y','NISP-J',-1,1],'color3':['NISP-J','NISP-H',-1,1]}#,\
    #'color5':['u','i',1.5,3.5], 'color6':['g','i',0.5,1.5], 'color7':['r','i',0,0.6], 'color8':['i','k',1,3.5]}   # clean selection


    ####################################################################################################
    ####################################################################################################
    ####################################################################################################

    # Don't mind this part! (!!! DO NOT CHANGE UNLESS YOU KNOW WHATYOU ARE DOING)

    working_directory = WORKING_DIR

    input_dir = working_directory+'inputs_dwarfs/'
    output_dir = working_directory+'outputs_dwarfs/'
    main_data_dir = working_directory+'ERO-data/ERO-PERSEUS/'#input_dir+'main_data/'

    data_dir = input_dir+'data/'
    psf_dir = input_dir+'psf/'
    clean_data_dir = output_dir+'clean_data/'
    sub_data_dir = output_dir+'sub_data/'
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

    for dir in [working_directory,input_dir,output_dir,data_dir,main_data_dir,clean_data_dir,img_dir,sex_dir,fit_dir,plots_dir,\
    detection_dir,cats_dir,psfs_dir,art_dir,final_cats_dir,temp_dir,sbf_dir,psf_dir,check_plots_dir,sub_data_dir] :
        if not os.path.exists(dir): os.makedirs(dir)

    gal_params = {}
    gal_methods = {}
    gal_data_name = {}
    for line in TARGETS:
        #print (line)
        line = line[0].split(' ')
        gal_id, data_name, gal_name, ra, dec, distance, filters, methods, comments = int(line[0]), str(line[1]), str(line[2]),\
        float(line[3]), float(line[4]), float(line[5]), np.array(line[6].split(',')), np.array(line[7].split(',')), np.array((line[8].split('\n')[0]).split(','))
        gal_params[gal_id] = [gal_name, ra, dec, distance, filters, comments]
        gal_methods[gal_id] = methods
        gal_data_name[gal_id] = data_name
        #print (gal_name, ra, dec, distance, filters, comments)


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
