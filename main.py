import os, sys
import math
import numpy as np
import matplotlib.pyplot as plt
from fit_galaxy import fit_galaxy_sersic
import plots
from functions import *
from extract_sources import *

def initialize_params() :
    global data_dir, sex_dir, fit_dir, img_dir, plots_dir, detection_dir
    global udgs, coords, instruments, aper_corr_values

    data_dir = 'data/'
    img_dir = 'img/'
    sex_dir = 'sex/'
    fit_dir = 'fit/'
    plots_dir = 'plots/'
    detection_dir = 'detection/'

    # first filter is the main filter, following filters are for colors
    ### 606 for DF44 is UVIS -> ###'606'
    udgs = {'DF44':['814','475'],\
            'DF07':['814','475'],\
            'DF08':['814','475'],\
            'PU1251013':['814','475'],\
            'DFX1':['606','814'],\
            'DF17':['814','475','606']} 

    #udgs = {'DF44':['606','814']}
    #coords = {'DF44':[195.2416667,+26.9763889]} #ra and dec
    #instruments = {'DF44':['UVIS',0.03962,26.10,25.14]}

    #udgs = {'DF44':['814']}
    coords = {'DF44':[195.2416667,+26.9763889],\
             'DF07':[194.2570833,+28.3902778],\
             'DF08':[195.3766667,+28.3744444],\
             'PU1251013':[192.7554167, +27.7980556],\
             'DFX1':[195.3158333, +27.2102778],\
             'DF17':[195.4929167, +27.8363889]} #ra and dec

    instruments = {'DF44':['ACS',0.05,26.054,25.943,26.10],\
                   'DF07':['ACS',0.05,26.054,25.943],\
                   'DF08':['ACS',0.05,26.054,25.943],\
                   'PU1251013':['ACS',0.05,26.054,25.943],\
                   'DFX1':['ACS',0.05,26.498,25.944],\
                   'DF17':['ACS',0.05,25.944,26.06,26.498]}

    aper_corr_values = {'DF44':[0.19,0.19,0.18],\
                        'DF07':[0.19,0.19],\
                        'DF08':[0.19,0.19],\
                        'PU1251013':[0.19,0.19],\
                        'DFX1':[0.20,0.19],\
                        'DF17':[0.19,0.19,0.20]} 
    

    # fill in coordinates and zero-points

###
initialize_params()

for udg in udgs.keys() :
    ################
    if udg in ['DF44'] :
        check=0
    else :
        continue
    ###############
    print ('\n+ Analyzing the UDG '+str(udg))

    filters = udgs[udg]
    main_filter = filters[0]
    other_filters = filters[1:]
    n_other_filters = len(other_filters)
    print ('\n+++ Available filters are : '+str(filters))
    print ('+++ The main filter is : '+str(main_filter))
    #print ('+++ The number of other filters is : '+str(n_other_filters))
    #print ('+++ The other filters are : '+str(other_filters))
    coord = coords[udg]
    ra = coord[0]
    dec = coord[1]
    print ('+++ Coordinates : '+'RA : ' + str(ra)+', DEC : ' + str(dec))
    ###

    
    main_data = data_dir+str(udg)+'_'+str(main_filter)+'.fits'
    pix_size = (instruments[udg])[1]
    zp = (instruments[udg])[2]

    """
    ### resampling data
    main_data_resampled = resample(main_data,udg,main_filter)
    ### make a fancy frame for galaxies
    plots.make_fancy_frame(main_data_resampled,ra,dec,udg,main_filter, img_dir)
    ### sersic fitting -> mag, Re, n, colors
    ### main filter
    fit_galaxy_sersic(main_data_resampled,ra,dec,udg,main_filter,pix_size,fit_dir,zp,plotting=True,\
    r_cut=int(16./pix_size),r_cut_back=int(40./pix_size),\
    r_cut_back_mask=int(16./pix_size),r_cut_fit=int(16./pix_size)) 
    """
    ### secondary filter
    """
    for other_filter in other_filters :
        other_data = data_dir+str(udg)+'_'+str(other_filter)+'.fits'
        other_data_resampled = resample(other_data,udg,other_filter)
        fit_galaxy_sersic(other_data_resampled,ra,dec,udg,other_filter,pix_size,fit_dir,zp,plotting=True,\
        r_cut=int(16./pix_size),r_cut_back=int(40./pix_size),\
        r_cut_back_mask=int(16./pix_size),r_cut_fit=int(16./pix_size)) 
    """

    ############
    
    ### read sersic params from files
    sersic_params_file_main = open(fit_dir+udg+'_'+main_filter+'_sersic_params.csv')
    for line in sersic_params_file_main :
        sersic_params_main = line.split(',')
    print ('+ sersic parameters for galaxy '+udg+' are:\n')
    print (sersic_params_main)

    re_best_kpc = sersic_params_main[0]
    n_best = sersic_params_main[1]
    pa_best_corr = sersic_params_main[2]
    axis_ratio_best = sersic_params_main[3]
    mag_best = sersic_params_main[4]
    d_re_best_kpc = sersic_params_main[5]
    d_n_best = sersic_params_main[6]
    d_pa_best = sersic_params_main[7]
    d_axis_ratio_best = sersic_params_main[8]
    d_mag_best = sersic_params_main[9]
    x_best = sersic_params_main[10]
    y_best = sersic_params_main[11]

    # remove galaxy light and making detection frame
    print ('+++ resampling the main data')
    main_data_resampled = resample(main_data,udg,main_filter)
    print ('+++ making the detection frame')
    detection_frame = make_detection_frame(main_data_resampled,udg, main_filter, data_dir, sex_dir, detection_dir, \
        coords[udg][0], coords[udg][1], pix_size, backsize=32, backfiltersize=1, iteration=3)

    for i in range(n_other_filters) :
        other_data = data_dir+str(udg)+'_'+str(other_filters[i])+'.fits'
        other_data_resampled = resample(other_data,udg,other_filters[i])
        detection_frame = make_detection_frame(other_data_resampled,udg, other_filters[i], data_dir, sex_dir, detection_dir, \
        coords[udg][0], coords[udg][1], pix_size, backsize=32, backfiltersize=1, iteration=3)
    
    os.system('mv *'+udg+'*'+'resample* '+detection_dir)
    os.system('mv *'+udg+'*'+'detection_frame* '+detection_dir)
    os.system('rm coadd.weight.fits')
   

    ### resampling weights
    
    for i in range(len(filters)): 
        data_weight_resampled = resample_weight(detection_dir+udg+'_'+filters[i]+'_resampled.fits',\
        data_dir+str(udg)+'_'+str(filters[i])+'.weight.fits',udg,filters[i])
    os.system('mv *'+udg+'*'+'resample* '+detection_dir)
    
    # extract sources in different filters and make Multi-wavelength catalogues

    resampled_frames = list()
    detection_frames = list()
    resampled_weight_frames = list()

    for i in range(len(filters)): 

        resampled_frames.append(detection_dir+udg+'_'+filters[i]+'_resampled.fits')
        resampled_weight_frames.append(detection_dir+udg+'_'+filters[i]+'.weight_resampled.fits')
        detection_frames.append(detection_dir+udg+'_'+filters[i]+'.detection_frame.fits')

    make_sex_cats(resampled_frames, udg, filters, resampled_weight_frames, detection_frames, sex_dir, detection_dir, \
        coords[udg][0], coords[udg][1], pix_size, (instruments[udg])[2+i], (aper_corr_values[udg])[i], 28)

    ###########################

    # simulations of stars

    # select points sources based on simulations -> make catalogues of candidates

    ###########################

    # measure half number radius using MLE code + radial distribution (R1. R_GC, R_GC/Re)

    # luminosity function -> peak ? sigma ? stacked peak and sigma ? (R2. N_GC, S_N, M_halo, M/L)

    # colors of globular clusters (R3. ?)


    #gc_cat = select_point_sources(main_data)
    #gc_dist = gc_distribution(gc_cat)
    #count_gc_number(gc_cat,gc_dist)
    #gc_colors(main_data,other_data)

    os.system('mv *_resampled.fits '+fit_dir)
    os.system('mv *weight.fits '+fit_dir)
    os.system('mv *.png '+plots_dir)




