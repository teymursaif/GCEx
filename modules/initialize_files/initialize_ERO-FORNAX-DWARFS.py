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
    PRIMARY_FRAME_SIZE_ARCSEC = 480 #arcsec
    FRAME_SIZE_ARCSEC = 480 #cut-out size from the original frame for the general anlaysis (arcsec)

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

    
    coords = [  [53.7467, -35.1711],\
                [54.2263, -35.3747],\
                [53.6228, -35.5465],\
                [54.2689, -35.5901],\
                [54.3472, -34.9001],\
                [054.15532, -35.38590],\
                [53.9282, -35.3383],\
                [54.2219, -34.9384],\
                [54.0169, -35.3889],\
                [53.584, -35.3624],\
                [54.1553, -35.3859],\
                [54.1792, -35.4356],\
                [53.9283, -35.5141],\
                [54.2492, -35.3432],\
                [54.1165, -35.2107],\
                [53.7354, -35.1909],\
                [53.7426, -35.0427],\
                [54.4213, -35.2962],\
                [53.7728, -35.2184],\
                [53.876800537109375, -35.251800537109375],\
                [53.751, -35.3224],\
                #[54.2917, -35.3865],\
                [54.2611, -34.8755],\
                [53.798, -35.3229],\
                #[54.0011, -34.8781],\
                [53.7195, -35.5694],\
                #[53.7197, -35.5695],\
                [53.9912, -35.348],\
                [54.2494010925293, -35.137298583984375],\
                [53.74, -35.2235],\
                #[53.5507, -35.2285],\
                [54.4107, -35.3854],\
                [53.7727, -35.4506],\
                #[53.75279998779297, -35.42210006713867],\
                [53.7167, -35.6233],\
                #[54.16120147705078, -35.61140060424805],\
                #[53.635398864746094, -35.04359817504883],\
                #[54.31629943847656, -35.357398986816406],\
                #[53.74290084838867, -35.258399963378906],\
                [53.8001, -35.4342],\
                [53.9048, -34.9795],\
                [53.680999755859375, -35.52109909057617],\
                [53.94729995727539, -35.361900329589844],\
                [54.00740051269531, -35.310699462890625],\
                #[54.36289978027344, -35.30580139160156],\
                #[53.672000885009766, -35.506900787353516],\
                #[53.69160079956055, -34.89250183105469],\
                #[53.618900299072266, -34.94409942626953]   
                ] #35 objects
    i = -1
    for coord in coords:
        i = i+1
        ra, dec = coord
        if ((i>2) and (i<=40)):
        #if i==3 :
            #target_str = str(i) +' ERO-FORNAX FORNAX-DWARF-'+str(i)+' '+str(ra)+' '+str(dec)+' 20 VIS,NISP-Y,NISP-J,NISP-H MAKE_GC_CAT DWARF,LSB'
            target_str = str(i) +' ERO-FORNAX FORNAX-DWARF-'+str(i)+' '+str(ra)+' '+str(dec)+' 20 VIS,NISP-Y,NISP-J,NISP-H MAKE_GC_CAT DWARF,LSB' #,NISP-Y,NISP-J,NISP-H
            TARGETS.append([target_str])
            #print (target_str)

    #TARGETS.append(['100 ERO-FORNAX FORNAX-DWARF-100 054.23864 -35.50702 20 VIS,NISP-Y,NISP-J,NISP-H MAKE_GC_CAT MASSIVE']) #FCC184
    #TARGETS.append(['101 ERO-FORNAX FORNAX-DWARF-101 054.13133 -35.29561 20 VIS,NISP-Y,NISP-J,NISP-H MAKE_GC_CAT MASSIVE']) #FCC170
    #TARGETS.append(['102 ERO-FORNAX FORNAX-DWARF-102 054.28715 -35.19500 20 VIS,NISP-Y,NISP-J,NISP-H MAKE_GC_CAT MASSIVE']) #FCC190

    #TARGETS.append(['103 ERO-FORNAX FORNAX-DWARF-103 054.11516 -34.97445 20 VIS,NISP-Y,NISP-J,NISP-H MAKE_GC_CAT MASSIVE']) #FCC167
    #TARGETS.append(['104 ERO-FORNAX FORNAX-DWARF-104 053.81938 -35.22624 20 VIS,NISP-Y,NISP-J,NISP-H MAKE_GC_CAT MASSIVE']) #FCC147-148
    #TARGETS.append(['105 ERO-FORNAX FORNAX-DWARF-105 054.01621 -35.44146 20 VIS,NISP-Y,NISP-J,NISP-H MAKE_GC_CAT MASSIVE']) #FCC161

    # NOTE: possible methods -> RESAMPLE_DATA, MODEL_PSF, FIT_GAL, USE_SUB_GAL, MAKE_CAT, MAKE_GC_CAT
    # NOTE: possible comments -> MASSIVE,DWARF,LSB

 
    MERGE_CATS = False
    MERGE_SIM_GC_CATS = False
    MERGE_GC_CATS = True

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
    MAG_LIMIT_CAT = 27
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
    EXTERNAL_CROSSMATCH = True
    EXTERNAL_CROSSMATCH_CAT = './archival_tables/ERO-FDS-ugriJKs.fits'

    PARAM_SEL_METHOD = 'MANUAL'
    PARAM_SEL_RANGE = {'ELLIPTICITY':[-0.01,0.5],'F_MAG_APER_CORR':[19,26],'F_MAG_APER_CORR_VIS':[15,30],'F_MAG_APER_CORR_NISP-Y':[15,30],\
    'color0':['VIS','VIS',-0.5,0.5],'color1':['VIS','NISP-Y',-1.5,1.2],'color2':['NISP-Y','NISP-J',-1,1],'color3':['NISP-J','NISP-H',-1,1]}#,\
    #'color5':['u','i',1.5,3.5], 'color6':['g','i',0.5,1.5], 'color7':['r','i',0,0.6], 'color8':['i','k',1,3.5]}   # clean selection


    ####################################################################################################
    ####################################################################################################
    ####################################################################################################

    # Don't mind this part! (!!! DO NOT CHANGE UNLESS YOU KNOW WHATYOU ARE DOING)

    working_directory = WORKING_DIR

    input_dir = working_directory+'inputs_dwarfs/'
    output_dir = working_directory+'outputs_dwarfs/'
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
