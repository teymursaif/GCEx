import os, sys
import math
import numpy as np
import matplotlib.pyplot as plt
from fit_galaxy import fit_galaxy_sersic
from plots import *
from functions import *
from extract_sources import *
from lacosmic import lacosmic
from plots import *

def initialize_params() :
    global data_dir, sex_dir, fit_dir, img_dir, plots_dir, detection_dir, clean_data_dir, psf_image_size 
    global udgs, coords, instruments, aper_corr_values, pix_sizes, zero_points, gains, cats_dir, psfs_dir, art_dir, final_cats_dir
    global mags, N_art_stars, GC_SEL_METHOD, GC_SEL_PARAMS, average_color_shift, color_range, Z_range, coords_corr

    data_dir = 'data/'
    clean_data_dir = 'clean_data/'
    img_dir = 'img/'
    sex_dir = 'sex/'
    fit_dir = 'fit/'
    plots_dir = 'plots/'
    detection_dir = 'detection/'
    cats_dir = 'cats/'
    psfs_dir = 'psfs/'
    art_dir = 'artificial/'
    final_cats_dir = 'final_cats/'

    # first filter is the main filter, following filters are for colors
    ### 606 for DF44 is UVIS -> ###'606'
    ### 606 for DF17 is ACS
    udgs = {'DF44':['814','475'],\
            'DF07':['814','475'],\
            'DF08':['814','475'],\
            'PU1251013':['814','475'],\
            'DFX1':['606','814'],\
            'DF17':['814','475'],\
            'DF4':['814','606'],\
            'DF2':['814','606']} 

    #udgs = {'DF44':['606','814']}
    #coords = {'DF44':[195.2416667,+26.9763889]} #ra and dec
    #instruments = {'DF44':['UVIS',0.03962,26.10,25.14]}

    #udgs = {'DF44':['814']}
    coords = {'DF44':[195.2416667,+26.9763889],\
             'DF07':[194.2570833,+28.3902778],\
             'DF08':[195.3766667,+28.3744444],\
             'PU1251013':[192.7554167, +27.7980556],\
             'DFX1':[195.3158333, +27.2102778],\
             'DF17':[195.4929167, +27.8363889],\
             'DF4':[39.81279167, -08.11597222],\
             'DF2':[40.44523, -08.40309]}

    coords_corr = \
        {'DF44':[[0,0],[0,0]],\
        'DF07':[[0,0],[0,0]],\
        'DF08':[[0,0],[-0.0001,-0.00012]],\
        'PU1251013':[[0,0], [+0.00003,-0.00002]],\
        'DFX1':[[0,0], [0,0]],\
        'DF17':[[0,0], [0,0]],\
        'DF4':[[0,0], [0,0]],\
        'DF2':[[0,0], [0,0]]}

    #instruments = {'DF44':['ACS',0.05,26.054,25.943,26.10],\
    #               'DF07':['ACS',0.05,26.054,25.943],\
    #               'DF08':['ACS',0.05,26.054,25.943],\
    #               'PU1251013':['ACS',0.05,26.054,25.943],\
    #               'DFX1':['ACS',0.05,26.498,25.944],\
    #               'DF17':['ACS',0.05,25.944,26.06,26.498]}


    instruments = {'DF44':['ACS','ACS','UVIS'],\
                   'DF07':['ACS','ACS',],\
                   'DF08':['ACS','ACS',],\
                   'PU1251013':['ACS','ACS',],\
                   'DFX1':['ACS','ACS'],\
                   'DF17':['ACS','ACS','ACS'],\
                   'DF4':['ACS','ACS'],\
                   'DF2':['ACS','ACS']}

    pix_sizes = {'DF44':[0.05,0.05,0.0396],\
                   'DF07':[0.05,0.05],\
                   'DF08':[0.05,0.05],\
                   'PU1251013':[0.05,0.05],\
                   'DFX1':[0.05,0.05],\
                   'DF17':[0.05,0.05,0.05],\
                   'DF4':[0.05,0.05],\
                   'DF2':[0.05,0.05]}

    zero_points = {'DF44':[25.936,26.043,26.08],\
                   'DF07':[25.936,26.043],\
                   'DF08':[25.936,26.043],\
                   'PU1251013':[25.936,26.043],\
                   'DFX1':[26.489,25.937],\
                   'DF17':[25.939,26.046,26.490],\
                   'DF4':[25.936,26.487],\
                   'DF2':[25.936,26.487]}

    aper_corr_values = {'DF44':[0.54,0.44,0.54],\
                        'DF07':[0.54,0.44],\
                        'DF08':[0.54,0.44],\
                        'PU1251013':[0.54,0.44],\
                        'DFX1':[0.45,0.54],\
                        'DF17':[0.54,0.44,0.45],\
                        'DF4':[0.54,0.45],\
                        'DF2':[0.54,0.45]} 

    gains = {'DF44':[2*5000,2*5000,3.2*2400],\
            'DF07':[2*5000,2*5000],\
            'DF08':[2*5000,2*5000],\
            'PU1251013':[2*5000,2*5000],\
            'DFX1':[4*2400,2*2400],\
            'DF17':[6.66*1100,6.66*1100,6.66*1100],\
            'DF4':[8*2000,8*2000],\
            'DF2':[26*2000,26*2000]} 

    GC_SEL_METHOD = \
            {'DF44':['UNRESOLVED'],\
            'DF07':['UNRESOLVED'],\
            'DF08':['UNRESOLVED'],\
            'PU1251013':['UNRESOLVED'],\
            'DFX1':['UNRESOLVED'],\
            'DF17':['UNRESOLVED'],\
            'DF4':['RESOLVED'],\
            'DF2':['RESOLVED']} 

    average_color_shift = \
        {'DF44':[0,-0.8],\
        'DF07':[0,-0.8],\
        'DF08':[0,-0.8],\
        'PU1251013':[0,-0.8],\
        'DFX1':[0,0.35],\
        'DF17':[0,-0.8],\
        'DF4':[0,-0.35],\
        'DF2':[0,-0.35]} 

    color_range = \
        {'DF44':['475','814',0.5,1.4],\
        'DF07':['475','814',0.5,1.4],\
        'DF08':['475','814',0.5,1.4],\
        'PU1251013':['475','814',0.5,1.4],\
        'DFX1':['606','814',0.2,0.65],\
        'DF17':['475','814',0.5,1.4],\
        'DF4':['606','814',0.2,0.65],\
        'DF2':['606','814',0.2,0.65]} 

    Z_range = \
        {'DF44':[-99],\
        'DF07':[-99],\
        'DF08':[-99],\
        'PU1251013':[-99],\
        'DFX1':[-99],\
        'DF17':[-99],\
        'DF4':[0.2,0.7,0.4],\
        'DF2':[0.2,0.7,0.4]} 

    ###
    
    psf_image_size = 40
    #mags = np.arange(220,261,5)/10.
    #N_art_stars = 100
    mags = np.arange(220,291,1)/10.
    N_art_stars = 1000
    # fill in coordinates and zero-points

###
initialize_params()
#os.system('echo 1 > /proc/sys/vm/overcommit_memory')

for udg in udgs.keys() :
    initialize_params()
    ################ 
    if udg in ['DF44','DF07','DF08','DFX1','PU1251013','DF17']:#,'DF44','DF07']:#,'DF08','DFX1','PU1251013','DF17'] : 'DF44','DF07'
    #if udg in ['DF08']:#,'DF44','DF07']:#,'DF08','DFX1','PU1251013','DF17'] : 'DF44','DF07'
        #continue
        check=0
    else :
        #check=0
        continue
    ###############
    cosmic_clean = True
    #if udg in ['DF2','DF4'] :
    #    cosmic_clean = False

    #if udg == 'DF44' :
    #    cosmic_clean = False

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
    pix_size = (pix_sizes[udg])[0]
    print ('+++ Coordinates : '+'RA : ' + str(ra)+', DEC : ' + str(dec))
    ###
    sc = 0.5
    crop_sc = 0.5
    if udg in ['DF2','DF4'] :
        sc = 0.065
        crop_sc = 0.5

    ### clean cosmic rays
    
    if cosmic_clean == True :
        for i in range(len(filters)): 
            donothing=1
            if udg in ['DF2','DF4'] :
                clean_cosmic_rays(data_dir+str(udg)+'_'+str(filters[i])+'.fits',clean_data_dir+str(udg)+'_'+str(filters[i])+'.fits',(gains[udg])[i])
    
    #continue
    #old_data_old = data_dir

    old_main_data = data_dir+str(udg)+'_'+str(main_filter)+'.fits'
    old_main_weight = data_dir+str(udg)+'_'+str(main_filter)+'.weight.fits'

    if cosmic_clean == False :
        for filter in filters :
            os.system('cp '+data_dir+udg+'_'+filter+'.fits '+clean_data_dir)
            os.system('cp '+data_dir+udg+'_'+filter+'.weight.fits '+clean_data_dir)

    if cosmic_clean == True :
        for filter in filters :
            os.system('cp '+data_dir+udg+'_'+filter+'.weight.fits '+clean_data_dir)

    data_dir = clean_data_dir
    main_data = data_dir+str(udg)+'_'+str(main_filter)+'.fits'
    main_weight = data_dir+str(udg)+'_'+str(main_filter)+'.weight.fits'


    ### resampling data and make a fancy frame for galaxies
    """
    pix_size = (pix_sizes[udg])[0]
    main_data_resampled = resample(main_data,udg,main_filter,pix_size)
    make_fancy_frame(main_data_resampled,ra,dec,udg,main_filter,img_dir)

    if udg in ['DF2','DF4'] :
        make_fancy_frame(main_data_resampled,ra,dec,udg,main_filter, img_dir,frame_size=500)
    

    ### sersic fitting -> mag, Re, n, colors

    zp = (zero_points[udg])[0]
    fit_galaxy_sersic(main_data_resampled,ra,dec,udg,main_filter,pix_size,fit_dir,zp,plotting=True,\
    r_cut=int(8./crop_sc/pix_size),r_cut_back=int(20./crop_sc/pix_size),\
    r_cut_back_mask=int(8./crop_sc/pix_size),r_cut_fit=int(8./crop_sc/pix_size),scale=sc) 
    
    i = 0
    for other_filter in other_filters :
        i = i+1
        zp = (zero_points[udg])[i]
        pix_size = (pix_sizes[udg])[i]
        other_data = data_dir+str(udg)+'_'+str(other_filter)+'.fits'
        other_data_resampled = resample(other_data,udg,other_filter,pix_size)
        fit_galaxy_sersic(other_data_resampled,ra,dec,udg,other_filter,pix_size,fit_dir,zp,plotting=True,\
        r_cut=int(8./crop_sc/pix_size),r_cut_back=int(20./crop_sc/pix_size),\
        r_cut_back_mask=int(8./crop_sc/pix_size),r_cut_fit=int(8./crop_sc/pix_size),scale=sc) 

    os.system('rm '+fit_dir+'*.fits')

    ############
    
    # remove galaxy light and making detection frame
    
    print ('+++ resampling the main data')
    pix_size = (pix_sizes[udg])[0]
    zp = (zero_points[udg])[0]
    main_data_resampled = resample(main_data,udg,main_filter,pix_size)
    print ('+++ making the detection frame')
    detection_frame = make_detection_frame(main_data_resampled,udg, main_filter, data_dir, sex_dir, detection_dir, \
        coords[udg][0], coords[udg][1], pix_size, backsize=32, backfiltersize=1, iteration=3)

    for i in range(n_other_filters) :
        pix_size = (pix_sizes[udg])[i+1]
        zp = (zero_points[udg])[i+1]
        other_data = data_dir+str(udg)+'_'+str(other_filters[i])+'.fits'
        other_data_resampled = resample(other_data,udg,other_filters[i],pix_size)
        detection_frame = make_detection_frame(other_data_resampled,udg, other_filters[i], data_dir, sex_dir, detection_dir, \
        coords[udg][0], coords[udg][1], pix_size, backsize=32, backfiltersize=1, iteration=3)
    
    os.system('mv *'+udg+'*'+'resample* '+detection_dir)
    os.system('mv *'+udg+'*'+'detection_frame* '+detection_dir)
    os.system('rm coadd.weight.fits')
    
    
    ### resampling weights
    
    for i in range(len(filters)): 
        pix_size = (pix_sizes[udg])[i]
        zp = (zero_points[udg])[i]
        data_weight_resampled = resample_weight(detection_dir+udg+'_'+filters[i]+'_resampled.fits',\
        data_dir+str(udg)+'_'+str(filters[i])+'.weight.fits',udg,filters[i],pix_size)
    os.system('mv *'+udg+'*'+'resample* '+detection_dir)
    
    # extract sources in different filters 
    """
    resampled_frames = list()
    detection_frames = list()
    resampled_weight_frames = list()
    
    for i in range(len(filters)): 

        resampled_frames.append(detection_dir+udg+'_'+filters[i]+'_resampled.fits')
        resampled_weight_frames.append(detection_dir+udg+'_'+filters[i]+'.weight_resampled.fits')
        detection_frames.append(detection_dir+udg+'_'+filters[i]+'.detection_frame.fits')
    
    out_cat = make_sex_cats(resampled_frames, udg, filters, resampled_weight_frames, detection_frames, sex_dir, detection_dir, \
        coords[udg][0], coords[udg][1], pix_sizes[udg], (zero_points[udg]), (aper_corr_values[udg]), gains[udg], 30.0, coords_corr[udg])
    
    os.system('mv *'+udg+'*'+'catalogue* '+cats_dir)
    """
    ###########################
    
    for i in range(len(filters)) :

        # select stars in frames
        #if filters[i] == '814' :
        #    continue
        
        if filters[i] == '814' or filters[i] == '606' :
            c1 = 0.25
            c2 = 0.45

        elif filters[i] == '475' :
            c1 = 0.2
            c2 = 0.4
    
        clean_fits_table(cats_dir+udg+'_'+filters[i]+'_'+'catalogue.fits', {'mag_'+filters[i]:[20,25], 'c48_'+filters[i]:[c1,c2]}, \
        output_table=cats_dir+udg+'_'+filters[i]+'_'+'star_catalogue.fits') 

        psf_frames = make_psf(udg,filters[i],data_dir,cats_dir,psfs_dir,sex_dir,'RA_'+filters[i],'DEC_'+filters[i], \
            'X_IMAGE_'+filters[i],'Y_IMAGE_'+filters[i], psf_image_size/2, udg+'_'+filters[i]+'_'+'star_catalogue.fits')

        plot_comp_mag_diagram([cats_dir+udg+'_'+filters[i]+'_'+'catalogue.fits',cats_dir+udg+'_'+filters[i]+'_'+'star_catalogue.fits'],\
        ['extracted sources','bright stars'],'c48_'+filters[i],'mag_'+filters[i],\
        'c$_{4-8}$','m$_{'+filters[i]+'}$',[0,1.21],[28.5,18],\
        'UDG '+udg+' (F'+filters[i]+'W)','compactness_plot_'+udg+'_'+filters[i]+'.png',['grey','red'])

        plot_radial_profile(psfs_dir+udg+'_'+filters[i]+'_PSF.fits',psf_frames,np.arange(1,31,1),sex_dir,\
            'UDG '+udg+' (F'+filters[i]+'W)',udg+'_'+filters[i]+'_PSF.png')

        # add artificial stars
        add_psfs_to_frame(old_main_data,old_main_weight,udg,filters[i],data_dir,psfs_dir,sex_dir,cats_dir,art_dir,N_art_stars,\
            mags,(zero_points[udg])[i],psf_image_size,(gains[udg])[i]) 
        
        os.system('cp '+clean_data_dir+str(udg)+'_'+str(filters[i])+'.weight.fits'+' '+art_dir)
        
        # clean the artificial data from cosmic darys
        
        #for mag in mags :
            #clean_cosmic_rays(art_dir+udg+'_'+filters[i]+'_+artificial_stars_mag'+str(mag)+'.fits',\
            #art_dir+udg+'_'+filters[i]+'_+artificial_stars_mag'+str(mag)+'_cr.fits',(gains[udg])[0])

        for mag in mags :
            #making detection frame
            #if filters[i] == '814' :
            #    continue 
            pix_size = (pix_sizes[udg])[i]
            zp = (zero_points[udg])[i]
            art_data_resampled = resample(art_dir+udg+'_'+filters[i]+'_+artificial_stars_mag'+str(mag)+'.fits',\
                udg,filters[i],pix_size,label='_+artificial_stars_mag'+str(mag))
            os.system('mv *artificial_stars_* '+art_dir)

            #resampling weight for artificial frame
            art_data_weight_resampled = resample_weight(art_dir+udg+'_'+filters[i]+'_+artificial_stars_mag'+str(mag)+'_resampled.fits',\
            art_dir+str(udg)+'_'+str(filters[i])+'.weight.fits',udg,filters[i],pix_size,label='_+artificial_stars_mag'+str(mag))
            os.system('mv *artificial_stars_* '+art_dir)

            art_detection_frame = make_detection_frame(art_dir+art_data_resampled,udg, filters[i], art_dir, sex_dir, art_dir, \
            coords[udg][0], coords[udg][1], pix_size, backsize=32, backfiltersize=1, iteration=3,label='_+artificial_stars_mag'+str(mag))

            art_out_cat = make_sex_cats([art_dir+art_data_resampled], udg, [filters[i]], [art_dir+art_data_weight_resampled], [art_detection_frame], sex_dir, art_dir, \
            coords[udg][0], coords[udg][1], [(pix_sizes[udg])[i]], [(zero_points[udg])[i]], (aper_corr_values[udg]), [(gains[udg])[0]], 30, label='_+artificial_stars_mag'+str(mag))
        
        os.system('mv *artificial_stars_* '+art_dir)
        
        #print ('+++ The measured source extraction completeness is:\n')
        comp_cat = open(cats_dir+udg+'_'+filters[i]+'.sex_completeness.cat', 'w')
        comp_cat.write('mag,completeness\n')
        tables = list()
        for mag in mags :
            ### crossmathc and completeness
            clean_fits_table(art_dir+udg+'_'+filters[i]+'_+artificial_stars_mag'+str(mag)+'_'+'catalogue.fits',{'mag_'+filters[i]:[mag-1,mag+1]})
            convert_csv_to_fits(art_dir+udg+'_'+filters[i]+'_+artificial_stars_mag'+str(mag)+'.cat',\
                art_dir+udg+'_'+filters[i]+'_+artificial_stars_mag'+str(mag)+'.cat.fits')

            n1,n2 = cross_match(art_dir+udg+'_'+filters[i]+'_+artificial_stars_mag'+str(mag)+'.cat.fits',\
                art_dir+udg+'_'+filters[i]+'_+artificial_stars_mag'+str(mag)+'_'+'catalogue.fits',\
                'RA','DEC','RA_'+filters[i],\
                'DEC_'+filters[i],art_dir+udg+'_'+filters[i]+'_+artificial_stars_mag'+str(mag)+'_cross_matched.fits')

            comp = n1/N_art_stars
            #COMP.append(comp)
            print (mag, comp)
            comp_cat.write(str(mag)+','+str(comp)+'\n')
            expand_fits_table(art_dir+udg+'_'+filters[i]+'_+artificial_stars_mag'+str(mag)+'_cross_matched.fits',\
                'mag_bin_'+filters[i],np.full(shape=n2,fill_value=mag,dtype=np.float))
            tables.append(art_dir+udg+'_'+filters[i]+'_+artificial_stars_mag'+str(mag)+'_cross_matched.fits')
        comp_cat.close()
        attach_fits_tables(tables,art_dir+udg+'_'+filters[i]+'_+artificial_stars_MASTER_cross_matched.fits')

        os.system('rm '+art_dir+'*artificial_stars_mag*')
    """
    ########################################
    # source selection and selection completeness -> catalogue of point sources -> later : catalogue of GC candidates (3Re)
    # points sources are selected in the main_filter
    print ('+++ Indefitying point sources (GCs):\n')

    points, art_points, medians1, medians2, stds1, stds2 = select_point_sources(main_data,udg,main_filter, psfs_dir+udg+'_'+main_filter+'_PSF_resampled.fits',10,
        cats_dir+udg+'_'+main_filter+'_catalogue.fits',\
        art_dir+udg+'_'+main_filter+'_+artificial_stars_MASTER_cross_matched.fits',\
        cats_dir+udg+'_'+main_filter+'_catalogue_of_points.fits','C_'+main_filter, 'mag_'+main_filter, True,
        ['c34'+'_'+main_filter,'c45'+'_'+main_filter,'c56'+'_'+main_filter,'c68'+'_'+main_filter,'c810'+'_'+main_filter],\
        'ELLIPTICITY'+'_'+main_filter, mags) 
        #'c34'+'_'+main_filter,'c45'+'_'+main_filter,'c56'+'_'+main_filter,'c68'+'_'+main_filter,'c810'+'_'+main_filter

    if (GC_SEL_METHOD[udg])[0] == 'UNRESOLVED' :
        os.system('cp '+points+' '+final_cats_dir+udg+'_'+main_filter+'.selected_sources.cat')

    elif (GC_SEL_METHOD[udg])[0] == 'RESOLVED' :
        TBW=1

    ####

    for i in range(len(filters)):
        filter = filters[i]
        if filter != main_filter :
            continue
        comp_cat = open(cats_dir+udg+'_'+filter+'.point_selection_completeness.cat', 'w')
        comp_cat.write('mag,completeness\n')
        for mag in mags :
            art = get_fits(art_points)
            art_data = art[1].data
            temp = art_data[(art_data['mag_bin_'+filter]<mag+0.05) & (art_data['mag_bin_'+filter]>mag-0.05)]
            comp = len(temp) / N_art_stars
            comp_cat.write(str(mag)+','+str(comp)+'\n')

        comp_cat.close()
        os.system('mv *artificial_stars* '+art_dir)
        os.system('mv *catalogue* '+final_cats_dir)

        if len(filters) == 1 :
            sex_comp_cat = cats_dir+udg+'_'+filter+'.sex_completeness.cat'
            point_comp_cat = cats_dir+udg+'_'+filter+'.point_selection_completeness.cat'
            total_comp_cat = cats_dir+udg+'.total_point_selection_completeness.cat'
            os.system('cp '+cats_dir+udg+'_'+filter+'.point_selection_completeness.cat'+' '+total_comp_cat)

            plot_completeness([sex_comp_cat,point_comp_cat],filter,'completeness_'+udg+'_'+filter+'.png',['black','red'],\
                ['extracted sources\n(F'+filter+'W)','identified point-sources\n(F'+filter+'W)'])
        
        elif len(filters) >= 2 :
            sex_comp_cat = cats_dir+udg+'_'+filter+'.sex_completeness.cat'
            secondary_sex_comp_cat = cats_dir+udg+'_'+filters[1]+'.sex_completeness.cat'
            point_comp_cat = cats_dir+udg+'_'+filter+'.point_selection_completeness.cat'
            total_comp_cat = cats_dir+udg+'.total_point_selection_completeness.cat'
            make_total_comp_cat([point_comp_cat,secondary_sex_comp_cat],total_comp_cat,average_color_shift[udg])

            plot_completeness([sex_comp_cat,point_comp_cat,total_comp_cat],filter,'completeness_'+udg+'_'+filter+'.png',['black','auto','violet'],\
                ['extracted sources\n(F'+filter+'W)','identified point-sources\n(F'+filter+'W)','identified point-sources\n(F'+filter+'W + '+'F'+filters[1]+'W)'])

    main_cat = final_cats_dir+udg+'_'+main_filter+'_'+'catalogue+Z.fits'
    art_cat = art_dir+udg+'_'+main_filter+'_artificial_stars_catalogue+Z.fits'
    selected_cat = final_cats_dir+udg+'_'+main_filter+'_catalogue_points_sources.fits'
    #make_plot_comp_Z_mag_diagrams(main_cat,art_cat,selected_cat,udg,main_filter,'selected_sources_compactness_plot_'+udg+'_'+main_filter+'.png',\
    #mags, medians1, medians2, stds, selection_mode=(GC_SEL_METHOD[udg])[0])

    make_plot_comp_mag_diagrams(main_cat,art_cat,selected_cat,udg,main_filter,'selected_sources_compactness_plot_'+udg+'_'+main_filter+'.png',\
    mags, medians1, medians2, stds1, stds2, selection_mode=(GC_SEL_METHOD[udg])[0])

    # Multi-wavelength catalogues and colour-selection

    selected_cat = final_cats_dir+udg+'_'+main_filter+'_catalogue_points_sources.fits'
    secondary_cat = cats_dir+udg+'_'+filters[1]+'_'+'catalogue.fits'
    final_selected_cat = final_cats_dir+udg+'_final_catalogue_of_selected_sources.fits'
    color_select(selected_cat, secondary_cat, udg, main_filter, filters[1], final_selected_cat,  (color_range[udg])[0],  (color_range[udg])[1], \
        (color_range[udg])[2], (color_range[udg])[3])
    
    ### read sersic params of the galaxy from files
    
    sersic_params_file_main = open(fit_dir+udg+'_'+main_filter+'_sersic_params.csv')
    for line in sersic_params_file_main :
        sersic_params_main = line.split(',')
    print ('+ sersic parameters for galaxy '+udg+' are:')
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

    # making some plots

    for mag in [27.5]:

        selected_cat = final_cats_dir+udg+'_'+main_filter+'.selected_sources.cat'

        plot_gal_gc(detection_dir+udg+'_'+main_filter+'.detection_frame.fits',ra,dec,udg,main_filter,200/crop_sc,pix_size,sc,selected_cat,'RA_'+main_filter,'DEC_'+main_filter,\
            'mag_'+main_filter,mag,float(re_best_kpc),float(pa_best_corr),float(axis_ratio_best),'selected_point_sources_around_galaxy_'+udg+'_'+main_filter+'_'+str(mag)+'.png')

        plot_gal_gc(detection_dir+udg+'_'+main_filter+'.detection_frame.fits',ra,dec,udg,main_filter,400/crop_sc,pix_size,sc,selected_cat,'RA_'+main_filter,'DEC_'+main_filter,\
            'mag_'+main_filter,mag,float(re_best_kpc),float(pa_best_corr),float(axis_ratio_best),'selected_point_sources_around_galaxy_'+udg+'_'+main_filter+'_zoomout'+'_'+str(mag)+'.png')


    for mag in [27.5]:

        final_selected_cat = final_cats_dir+udg+'_final_catalogue_of_selected_sources.fits'

        plot_gal_gc(detection_dir+udg+'_'+main_filter+'.detection_frame.fits',ra,dec,udg,main_filter,200/crop_sc,pix_size,sc,final_selected_cat,'RA_'+main_filter,'DEC_'+main_filter,\
            'mag_'+main_filter,mag,float(re_best_kpc),float(pa_best_corr),float(axis_ratio_best),'selected_sources_around_galaxy_'+udg+'_'+main_filter+'_'+str(mag)+'.png')

        plot_gal_gc(detection_dir+udg+'_'+main_filter+'.detection_frame.fits',ra,dec,udg,main_filter,400/crop_sc,pix_size,sc,final_selected_cat,'RA_'+main_filter,'DEC_'+main_filter,\
            'mag_'+main_filter,mag,float(re_best_kpc),float(pa_best_corr),float(axis_ratio_best),'selected_sources_around_galaxy_'+udg+'_'+main_filter+'_zoomout'+'_'+str(mag)+'.png')


    ###########################

    # measure half number radius using MLE code + radial distribution (R1. R_GC, R_GC/Re)

    # luminosity function -> peak ? sigma ? stacked peak and sigma ? (R2. N_GC, S_N, M_halo, M/L)

    # colors of globular clusters (R3. ?)

    #gc_cat = select_point_sources(main_data)
    #gc_dist = gc_distribution(gc_cat)
    #count_gc_number(gc_cat,gc_dist)
    #gc_colors(main_data,other_data)

    ############################

    os.system('mv *.png '+plots_dir)
    os.system('rm ' + udg+'_'+main_filter+ '*cropped*')
    #os.system('mv *_resampled.fits '+fit_dir)
    #os.system('mv *weight.fits '+fit_dir)





