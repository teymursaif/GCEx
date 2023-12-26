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
    PIXEL_SCALES, ZPS, PRIMARY_FRAME_SIZE, FRAME_SIZE, GAL_FRAME_SIZE, EXPTIME, GAIN, GC_REF_MAG, PSF_PIXELSCL_KEY, FWHM_LIMIT, INPUT_ZP, INPUT_EXPTIME, INPUT_GAIN, \
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
    #SE_executable = 'sex'
    #swarp_executable = 'swarp'
    SE_executable = 'sextractor'
    swarp_executable = 'SWarp'

    ### (if ZP, EXPTIME and GAIN are missing from the header, define them for a given filter)
    INPUT_ZP = {'VIS':30,'NISP-Y':30,'NISP-J':30,'NISP-H':30}
    INPUT_EXPTIME = {'VIS':565,'NISP-Y':112,'NISP-J':112,'NISP-H':112}
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

    
    coords = [  \
        ['CGCG540-074', 49.115, 41.627], \
        ['WISEAJ031637--12+414721--3', 49.155, 41.789], \
        ['UGC02626', 49.249, 41.357], \
        ['WISEAJ031702--05+413758--6', 49.258, 41.633], \
        ['CGCG540-079', 49.265, 41.634], \
        ['WISEAJ031713--24+413903--5', 49.305, 41.651], \
        ['WISEAJ031713--61+412607--8', 49.307, 41.436], \
        ['NGC1259', 49.322, 41.385], \
        ['WISEAJ031717--58+412517--7', 49.324, 41.421], \
        ['WISEAJ031724--65+414421--4', 49.353, 41.739], \
        ['NGC1260', 49.363, 41.405], \
        ['WISEAJ031727--24+415459--7', 49.363, 41.917], \
        ['WISEAJ031727--88+412917--1', 49.366, 41.488], \
        ['WISEAJ031727--91+413646--9', 49.366, 41.613], \
        ['PGC012221', 49.372, 41.375], \
        ['WISEAJ031732--09+412442--3', 49.383, 41.412], \
        ['WISEAJ031732--88+411749--0', 49.387, 41.297], \
        ['UGC02639', 49.46, 41.968], \
        ['CGCG540-085', 49.463, 41.451], \
        ['WISEAJ031753--71+410519--8', 49.474, 41.089], \
        ['WISEAJ031757--57+412950--4', 49.49, 41.497], \
        ['NGC1264', 49.498, 41.52], \
        ['2MASXJ03180070+4117567', 49.503, 41.299], \
        ['WISEAJ031801--75+413533--4', 49.507, 41.593], \
        ['WISEAJ031805--86+413452--9', 49.524, 41.581], \
        ['WISEAJ031806--47+414757--0', 49.527, 41.799], \
        ['03180840+4145157', 49.535, 41.754], \
        ['SDSSJ031813--09+414809--0', 49.555, 41.803], \
        ['NGC1265', 49.565, 41.858], \
        ['WISEAJ031816--46+414405--8', 49.568, 41.735], \
        ['WISEAJ031818--89+412809--3', 49.579, 41.469], \
        ['WISEAJ031820--98+412748--8', 49.587, 41.464], \
        ['CGCG540-087', 49.594, 41.41], \
        ['WISEAJ031822--90+411607--3', 49.596, 41.269], \
        ['WISEAJ031827--65+411916--7', 49.615, 41.321], \
        ['WISEAJ031831--67+411626--6', 49.632, 41.274], \
        ['2MFGC02715', 49.647, 41.667], \
        ['PUDGR24', 49.648, 41.809], \
        ['CGCG540-089', 49.65, 41.477], \
        ['WISEAJ031836--53+411131--7', 49.652, 41.192], \
        ['WISEAJ031838--93+414008--2', 49.662, 41.669], \
        ['NGC1267', 49.687, 41.468], \
        ['WISEAJ031844--81+413041--2', 49.687, 41.511], \
        ['NGC1268', 49.688, 41.489], \
        ['WISEAJ031847--39+412542--1', 49.697, 41.428], \
        ['WISEAJ031847--27+415850--3', 49.697, 41.981], \
        ['WISEAJ031851--10+412332--5', 49.713, 41.392], \
        ['WISEAJ031854--61+414810--8', 49.727, 41.803], \
        ['NGC1270', 49.742, 41.47], \
        ['LCSBS0532P', 49.742, 41.635], \
        ['WISEAJ031858--26+414208--7', 49.743, 41.702], \
        ['PCC3444', 49.752, 41.484], \
        ['WISEAJ031902--12+413711--6', 49.759, 41.62], \
        ['WISEAJ031902--42+413301--0', 49.76, 41.55], \
        ['WISEAJ031903--51+415826--9', 49.765, 41.974], \
        ['WISEAJ031904--31+411358--0', 49.768, 41.233], \
        ['PGC012358', 49.769, 41.468], \
        ['WISEAJ031904--70+412807--6', 49.769, 41.469], \
        ['NSA133302', 49.77, 41.469], \
        ['PCC3627', 49.775, 41.439], \
        ['WISEAJ031910--46+412936--4', 49.793, 41.494], \
        ['NGC1271', 49.797, 41.353], \
        ['PCC3832', 49.802, 41.511], \
        ['WISEAJ031917--76+413839--6', 49.824, 41.644], \
        ['PCC4036', 49.828, 41.442], \
        ['NGC1272', 49.839, 41.491], \
        ['WISEAJ031922--39+412545--6', 49.843, 41.429], \
        ['2MASSJ03192305+4129282', 49.846, 41.491], \
        ['NGC1273', 49.861, 41.541], \
        ['WISEAJ031926--97+412717--0', 49.863, 41.454], \
        ['UGC02665', 49.864, 41.635], \
        ['WISEAJ031931--38+412629--0', 49.881, 41.441], \
        ['WISEAJ031931--61+413121--7', 49.882, 41.523], \
        ['WISEAJ031933--64+413312--7', 49.89, 41.554], \
        ['PGC012405', 49.893, 41.58], \
        ['WISEAJ031937--27+412909--6', 49.905, 41.486], \
        ['WISEAJ031937--61+412248--3', 49.907, 41.38], \
        ['GALEXASCJ031939--68+413105--6', 49.917, 41.517], \
        ['WISEAJ031940--20+411945--0', 49.918, 41.329], \
        ['NGC1274', 49.919, 41.549], \
        ['WISEAJ031941--64+412916--8', 49.924, 41.488], \
        ['WISEAJ031941--95+412958--4', 49.925, 41.5], \
        ['WISEAJ031942--89+413602--3', 49.929, 41.601], \
        ['PCC4862', 49.931, 41.481], \
        ['WISEAJ031943--81+412725--1', 49.933, 41.457], \
        ['WISEAJ031944--54+411641--1', 49.936, 41.278], \
        ['WISEAJ031944--63+412647--3', 49.936, 41.446], \
        ['WISEAJ031945--85+415834--8', 49.941, 41.976], \
        ['PCC4960', 49.942, 41.414], \
        ['WISEAJ031946--78+411613--4', 49.945, 41.27], \
        ['WISEAJ031947--80+413546--8', 49.949, 41.596], \
        ['NGC1275', 49.951, 41.512], \
        ['WISEAJ031948--61+413329--1', 49.952, 41.558], \
        ['PGC012433', 49.963, 41.535], \
        ['NGC1277', 49.965, 41.573], \
        ['WISEAJ031952--47+413259--4', 49.969, 41.55], \
        ['WISEAJ031952--93+413631--6', 49.97, 41.609], \
        ['WISEAJ031952--98+411808--5', 49.971, 41.302], \
        ['NGC1278', 49.976, 41.563], \
        ['WISEAJ031954--38+420052--6', 49.977, 42.015], \
        ['WISEAJ031955--55+413123--4', 49.981, 41.523], \
        ['NGC1279', 49.996, 41.48], \
        ['VZw339', 50.004, 41.554], \
        ['WISEAJ032002--73+413033--6', 50.011, 41.509], \
        ['VZw338', 50.011, 41.51], \
        ['NGC1281', 50.025, 41.63], \
        ['WISEAJ032007--23+414958--5', 50.03, 41.833], \
        ['WISEAJ032010--12+412104--2', 50.042, 41.351], \
        ['NGC1282', 50.05, 41.367], \
        ['NGC1283', 50.065, 41.399], \
        ['WISEAJ032017--37+412056--0', 50.072, 41.349], \
        ['WISEAJ032020--96+412225--4', 50.087, 41.374], \
        ['MCG+07-07-070', 50.092, 41.641], \
        ['BGP073', 50.112, 41.483], \
        ['WISEAJ032028--61+412919--9', 50.119, 41.489], \
        ['WISEAJ032029--37+413052--4', 50.122, 41.515], \
        ['WISEAJ032030--90+413032--3', 50.129, 41.509], \
        ['WISEAJ032032--08+414530--5', 50.133, 41.759], \
        ['WISEAJ032032--97+413426--4', 50.137, 41.574], \
        ['PGC012520', 50.137, 41.732], \
        ['WISEAJ032042--17+412414--2', 50.176, 41.404], \
        ['WISEAJ032045--99+414405--8', 50.192, 41.735], \
        ['WISEAJ032049--31+412214--4', 50.205, 41.371], \
        ['WISEAJ032050--70+413601--5', 50.211, 41.6], \
        ['WISEAJ032056--57+412619--8', 50.236, 41.439], \
        ['WISEAJ032057--80+413022--8', 50.241, 41.506], \
        ['WISEAJ032100--40+413344--9', 50.252, 41.562], \
        ['WISEAJ032101--39+412604--0', 50.256, 41.434], \
        ['WISEAJ032102--23+412434--1', 50.259, 41.409], \
        ['WISEAJ032104--23+413320--2', 50.268, 41.556], \
        ['CGCG540-113', 50.337, 41.46], \
        ['LEDA215033', 49.511, 41.839], \
        ['SDSSJ031858--90+411915--1', 49.745, 41.321]

        ]


    #coords = [[049.95896,+41.41584],[049.14865,+41.60580],[049.73977,+41.35899],[049.44806,+41.79297],[049.35376,+41.73917],[050.27769,+41.53715],[049.50575,+41.82677]]
    
    
    i = -1
    for coord in coords:
        i = i+1
        name, ra, dec = coord
        if (i>-1):
            #target_str = str(i) +' ERO-FORNAX FORNAX-DWARF-'+str(i)+' '+str(ra)+' '+str(dec)+' 20 VIS,NISP-Y,NISP-J,NISP-H MAKE_GC_CAT DWARF,LSB'
            target_str = str(i) +' ERO-PERSEUS PERSEUS-MASSIVE-'+str(i)+' '+str(ra)+' '+str(dec)+' 70 VIS,NISP-Y,NISP-J,NISP-H MAKE_CAT,MAKE_GC_CAT DWARF,LSB' #,NISP-Y,NISP-J,NISP-H
            TARGETS.append([target_str])
            #print (target_str)

    # NOTE: possible methods -> RESAMPLE_DATA, MODEL_PSF, FIT_GAL, USE_SUB_GAL, MAKE_CAT, MAKE_GC_CAT
    # NOTE: possible comments -> MASSIVE,DWARF,LSB

 
    MERGE_CATS = False
    MERGE_SIM_GC_CATS = False
    MERGE_GC_CATS = False

    global TABLES
    TABLES = {}

    # ------------------------------  GALAXY FITTING ------------------------------

    GAL_FRAME_SIZE_ARCSEC  = 1*FRAME_SIZE_ARCSEC  #cut-out size from the original frame for sersic fitting anlaysis (arcsec)

    # ---------------------- SOURCE DETECTION AND PHOTOMETRY ----------------------

    PHOTOM_APERS = '1,2,4,8,12,16,20,30,40' #aperture-sizes (diameters) in pixels for aperture photometry with Sextractor
    BACKGROUND_ANNULUS_START = 5 #The size of background annulus for forced photoemtry as a factor of FWHM
    BACKGROUND_ANNULUS_TICKNESS = 40 # the thickness of the background annulus in pixels
    CROSS_MATCH_RADIUS_ARCSEC = 0.25
    MAG_LIMIT_CAT = 29.0
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

    GC_SEL_PARAMS = ['CI_2_4']#,'CI_4_8','CI_8_12']#,'CI_2_4','CI_4_6','CI_6_8','CI_8_10','CI_10_12','ELLIPTICITY']
    EXTERNAL_CROSSMATCH = False
    EXTERNAL_CROSSMATCH_CAT = './archival_tables/ERO-FDS-ugriJKs.fits'

    PARAM_SEL_METHOD = 'MANUAL'
    PARAM_SEL_RANGE = {'ELLIPTICITY':[-0.01,0.5],'F_MAG_APER_CORR_VIS':[23,28],'color0':['VIS','VIS',-0.5,0.5]}
    #,'color1':['VIS','NISP-Y',-1.5,1.2],'color2':['NISP-Y','NISP-J',-1,1],'color3':['NISP-J','NISP-H',-1,1]}#,\
    #'color5':['u','i',1.5,3.5], 'color6':['g','i',0.5,1.5], 'color7':['r','i',0,0.6], 'color8':['i','k',1,3.5]}   # clean selection


    ####################################################################################################
    ####################################################################################################
    ####################################################################################################

    # Don't mind this part! (!!! DO NOT CHANGE UNLESS YOU KNOW WHATYOU ARE DOING)

    working_directory = WORKING_DIR

    input_dir = working_directory+'inputs_dwarfs/'
    output_dir = working_directory+'outputs_dwarfs/'
    main_data_dir = input_dir+'main_data/'

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
