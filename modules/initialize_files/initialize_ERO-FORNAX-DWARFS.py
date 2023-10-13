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
    PRIMARY_FRAME_SIZE_ARCSEC = 200 #arcsec
    FRAME_SIZE_ARCSEC = 200 #cut-out size from the original frame for the general anlaysis (arcsec)

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

    
    coords = [  [54.347259583333326, -34.90023638888889],\
                [54.26138791666667, -34.875655555555554],\
                [54.22189458333334, -34.938596944444434],\
                [53.74250041666667, -35.04266444444445],\
                [53.73548583333333, -35.19070166666667],\
                [53.74044874999999, -35.22326861111111],\
                [53.77312333333333, -35.21843111111111],\
                [53.87696083333334, -35.252049722222225],\
                [54.11669916666667, -35.210549444444446],\
                [54.421252499999994, -35.29618111111111],\
                [54.00755541666667, -35.31060611111111],\
                [53.99128, -35.34806666666667],\
                [53.947287083333336, -35.36180388888889],\
                [53.9284975, -35.338152777777786],\
                [53.79813708333334, -35.322988055555555],\
                [53.7513775, -35.322134166666665],\
                [53.584193333333324, -35.362225555555554],\
                [54.01704125, -35.388711944444445],\
                [54.1792425, -35.43539805555556],\
                [54.26922208333333, -35.590001111111114],\
                [53.71988666666667, -35.56936861111111],\
                [53.92889833333333, -35.513646944444446],\
                [54.410928750000004, -35.38496416666667],\
                [54.24930416666667, -35.34307916666667],\
                [54.00118749999999, -34.87784333333334],\
                [53.96641833333334, -34.92155666666667],\
                [54.249565, -35.13749361111111],\
                [54.24444249999999, -35.19634111111111],\
                [54.1556075, -35.38543277777778],\
                [53.68137458333334, -35.520675555555556],\
                [53.77251291666667, -35.450301944444455],\
                [53.732805416666665, -35.461907777777775],\
                [53.613577916666664, -35.10570805555555],\
                [53.55079416666667, -35.22870972222222],\
                [54.31581541666666, -35.357863611111114],\
                [54.16131375, -35.61084972222222],\
                [54.11125, -35.639204722222225],\
                [54.09907375000001, -35.522627777777785],\
                [54.160634583333334, -35.2921575],\
                [54.16047916666667, -34.95903055555556],\
                [53.94106958333333, -35.20874361111111],\
                [53.982805, -35.29439583333333],\
                [53.713306666666675, -35.502943611111114],\
                [53.691750416666665, -34.89237111111111]     ]
    i = -1
    for coord in coords:
        i = i+1
        ra, dec = coord
        if i <= 20:
            target_str = str(i) +' ERO-FORNAX FORNAX-DWARF-'+str(i)+' '+str(ra)+' '+str(dec)+' 20 VIS,NISP-Y,NISP-J,NISP-H MAKE_CAT,SIM_GC,MAKE_GC_CAT DWARF,LSB'
            TARGETS.append([target_str])
            print (target_str)

    #TARGETS.append(['3 ERO-FORNAX FCC156 053.92795 -35.33825 20 VIS,NISP-Y,NISP-J,NISP-H --- DWARF,LSB'])
    #TARGETS.append(['4 ERO-FORNAX FCC195 054.34752 -34.89999 20 VIS,NISP-Y,NISP-J,NISP-H --- DWARF,LSB'])
    #TARGETS.append(['5 ERO-FORNAX FCC140 053.73535 -35.19085 20 VIS,NISP-Y,NISP-J,NISP-H --- DWARF,LSB'])
    #TARGETS.append(['6 ERO-FORNAX FCC181 054.22195 -34.93843 20 VIS,NISP-Y,NISP-J,NISP-H --- DWARF,LSB'])
    #TARGETS.append(['7 ERO-FORNAX FCC171 054.15530 -35.38588 20 VIS,FDS-r --- DWARF,LSB'])
    #TARGETS.append(['8 ERO-FORNAX FCC157 053.92910 -35.51372 20 VIS,FDS-r --- DWARF,LSB'])
    #TARGETS.append(['9 ERO-FORNAX FCC146 053.79801 -35.32300 20 VIS,FDS-r --- DWARF,LSB'])
    #TARGETS.append(['10 ERO-FORNAX FCC144 053.75087 -35.32226 20 VIS,FDS-r --- DWARF,LSB'])
    #TARGETS.append(['11 ERO-FORNAX FCC145 053.77289 -35.21845 20 VIS,FDS-r --- DWARF,LSB'])
    #TARGETS.append(['12 ERO-FORNAX FCC142 053.74258 -35.04281 20 VIS,FDS-r --- DWARF,LSB'])
    #TARGETS.append(['13 ERO-FORNAX FCC160 054.01694 -35.38885 20 VIS,NISP-Y,NISP-J,NISP-H --- DWARF,LSB'])
    #TARGETS.append(['14 ERO-FORNAX FCC197 054.42155 -35.29612 20 VIS,FDS-r --- DWARF,LSB'])

    # NOTE: possible methods -> RESAMPLE_DATA, MODEL_PSF, FIT_GAL, USE_SUB_GAL, MAKE_CAT, MAKE_GC_CAT
    # NOTE: possible comments -> MASSIVE,DWARF,LSB

 
    MERGE_CATS = False
    MERGE_SIM_GC_CATS = False
    MERGE_GC_CATS = False

    global TABLES
    TABLES = {}
    TABLES['acsfcs']='./archival_tables/ACS-FCS-GCs.fits'
    TABLES['fornax-spec-gcs']='./archival_tables/Fornax_spec_UCDs_and_GCs.fits' #only Saifollahi+2021b
    TABLES['fornax-spec-gcs-all']='./archival_tables/Fornax_spec_UCDs_and_GCs_all.fits'
    TABLES['fornax-spec-stars']='./archival_tables/Fornax_spec_foreground_stars.fits'
    TABLES['fornax-spec-galaxies']='./archival_tables/Fornax_spec_background_galaxies.fits'
    TABLES['gaia-stars']='./archival_tables/gaia_dr3_sources.fits'

    # ------------------------------  GALAXY FITTING ------------------------------

    GAL_FRAME_SIZE_ARCSEC  = 1*FRAME_SIZE_ARCSEC  #cut-out size from the original frame for sersic fitting anlaysis (arcsec)

    # ---------------------- SOURCE DETECTION AND PHOTOMETRY ----------------------

    PHOTOM_APERS = '1,2,4,8,12,16,20,30,40' #aperture-sizes (diameters) in pixels for aperture photometry with Sextractor
    BACKGROUND_ANNULUS_START = 3 #The size of background annulus for forced photoemtry as a factor of FWHM
    BACKGROUND_ANNULUS_TICKNESS = 20 # the thickness of the background annulus in pixels
    CROSS_MATCH_RADIUS_ARCSEC = 0.25
    MAG_LIMIT_CAT = 26
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
    
    N_ART_GCS = 100
    N_SIM_GCS = 1
    COSMIC_CLEAN = False #does not work at the moment anyways...
    GC_SIZE_RANGE = [1,8] #lower value should be small enough to make some point-sources for performance check, in pc
    GC_MAG_RANGE = [-10,-5]
    GC_REF_MAG = {'VIS':-8, 'NISP-Y':-8.3,'NISP-J':-8.3,'NISP-H':-8.3} #magnitude of a typical GC in the given filters should be defined here.
    GC_SIM_MODE = 'UNIFORM' # 'UNIFORM' or 'CONCENTRATED'

    #------------------------------ GC SELECTION -------------------------------

    GC_SEL_PARAMS = ['CI_2_4','CI_4_8','CI_8_12']#,'CI_2_4','CI_4_6','CI_6_8','CI_8_10','CI_10_12','ELLIPTICITY']
    EXTERNAL_CROSSMATCH = True
    EXTERNAL_CROSSMATCH_CAT = './archival_tables/ERO-FDS-ugriJKs.fits'

    PARAM_SEL_METHOD = 'MANUAL'
    PARAM_SEL_RANGE = {'color1':['VIS','NISP-Y',-1,0.8],'color2':['VIS','NISP-J',-2,2],'color3':['VIS','NISP-H',-2,2], \
        'color5':['u','i',0.5,4], 'color6':['g','i',0.2,1.8], 'color7':['r','i',-0.4,1], 'color8':['i','k',0.5,4], \
        'ELLIPTICITY':[0,0.5],'F_MAG_APER_CORR':[15,30]}


    ####################################################################################################
    ####################################################################################################
    ####################################################################################################

    # Don't mind this part! (!!! DO NOT CHANGE UNLESS YOU KNOW WHATYOU ARE DOING)

    working_directory = WORKING_DIR

    input_dir = working_directory+'inputs/'
    output_dir = working_directory+'outputs/'
    main_data_dir = working_directory+'ERO-data/ERO-FORNAX/'#input_dir+'main_data/'

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
