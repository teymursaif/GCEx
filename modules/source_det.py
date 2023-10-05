import os, sys
import math
import numpy as np
import matplotlib.pyplot as plt
#from astroquery.mast import Observations
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
#from scipy.ndimage import gaussian_filter
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

############################################################

def make_source_cat_full(gal_id):
    print (f"{bcolors.OKCYAN}- making source catalogs and photometry"+ bcolors.ENDC)
    make_source_cat(gal_id)
    make_multiwavelength_cat(gal_id, mode='forced-photometry')

############################################################

def prepare_sex_cat(source_cat_name_input,source_cat_name_output,gal_name,filter_name,distance):
    main = fits.open(source_cat_name_input)
    sex_cat_data = main[1].data
    fn = filter_name
    try:
        FWHM_limit = 0.75*FWHMS_ARCSEC[filter_name]/PIXEL_SCALES[filter_name]
    except:
        FWHM_limit = 0.5
    #print (FWHM_limit)
    #print (len(sex_cat_data))
    mask = ((sex_cat_data['FLAGS'] < 4) & \
    (sex_cat_data ['ELLIPTICITY'] < 1) & \
    (sex_cat_data ['MAG_AUTO'] > 0) & \
    (sex_cat_data ['MAG_AUTO'] < MAG_LIMIT_CAT) & \
    (sex_cat_data ['FWHM_IMAGE'] < 99999) & \
    (sex_cat_data ['FWHM_IMAGE'] > FWHM_limit) )
    sex_cat_data = sex_cat_data[mask]
    #print (len(sex_cat_data))
    hdul = fits.BinTableHDU(data=sex_cat_data)
    hdul.writeto(source_cat_name_output,overwrite=True)

    mag_apers = np.array(sex_cat_data['MAG_APER'])
    magerr_apers = np.array(sex_cat_data['MAGERR_APER'])
    flux_apers = np.array(sex_cat_data['FLUX_APER'])
    fluxerr_apers = np.array(sex_cat_data['FLUXERR_APER'])
    psf_dia_ref_pixel = 2*(APERTURE_SIZE[filter_name])/PIXEL_SCALES[filter_name]
    psf_dia_ref_pixel = int(psf_dia_ref_pixel*1000+0.4999)/1000
    apertures = (str(psf_dia_ref_pixel)+','+PHOTOM_APERS).split(',')
    expand_fits_table(source_cat_name_output,'MAG_APER_REF',mag_apers[:,0])
    expand_fits_table(source_cat_name_output,'MAGERR_APER_REF',magerr_apers[:,0])

    expand_fits_table(source_cat_name_output,'FLUX_APER_REF',flux_apers[:,0])
    expand_fits_table(source_cat_name_output,'FLUXERR_APER_REF',fluxerr_apers[:,0])

    expand_fits_table(source_cat_name_output,'APER_DIA_REF_PIXEL',mag_apers[:,0]*0+float(apertures[0]))

    fraction_corr = 2.512*np.log10(1./(PSF_REF_RAD_FRAC[fn]))
    flux_fraction_corr = 1./(PSF_REF_RAD_FRAC[fn])
    expand_fits_table(source_cat_name_output,'MAG_APER_CORR',mag_apers[:,0]-fraction_corr)
    expand_fits_table(source_cat_name_output,'MAGERR_APER_CORR',magerr_apers[:,0])

    expand_fits_table(source_cat_name_output,'FLUX_APER_CORR',flux_apers[:,0]*flux_fraction_corr)
    expand_fits_table(source_cat_name_output,'FLUXERR_APER_CORR',fluxerr_apers[:,0])

    # add size
    fwhm_obs = np.array(sex_cat_data['FWHM_WORLD'])*3600.0
    fwhm = (FWHMS_ARCSEC[filter_name])
    fwhm_int = fwhm_obs**2-fwhm**2
    #print (fwhm_obs, fwhm, fwhm_int)
    fwhm_int[fwhm_int<0] = 0
    fwhm_int = np.sqrt(fwhm_int)
    Re = fwhm_int/2
    expand_fits_table(source_cat_name_output,'FWHM_INT',fwhm_int)
    expand_fits_table(source_cat_name_output,'Re_arcsec',Re)
    expand_fits_table(source_cat_name_output,'Re_pc',Re*distance/0.206265)

    for a in range(len(apertures)-1):
        expand_fits_table(source_cat_name_output,'MAG_APER_'+str(apertures[a+1]),mag_apers[:,a+1])
        expand_fits_table(source_cat_name_output,'MAGERR_APER_'+str(apertures[a+1]),magerr_apers[:,a+1])
        expand_fits_table(source_cat_name_output,'FLUX_APER_'+str(apertures[a+1]),flux_apers[:,a+1])
        expand_fits_table(source_cat_name_output,'FLUXERR_APER_'+str(apertures[a+1]),fluxerr_apers[:,a+1])

    for a in range(len(apertures)-2):
        j = int(apertures[a+2])
        i = int(apertures[a+1])
        ci = mag_apers[:,a+1]-mag_apers[:,a+2]
        ci_err = np.sqrt(magerr_apers[:,a+2]**2+magerr_apers[:,a]**1)
        expand_fits_table(source_cat_name_output,'CI_'+str(i)+'_'+str(j),ci)
        expand_fits_table(source_cat_name_output,'CI_ERR_'+str(i)+'_'+str(j),ci_err)

    ttypes = list()
    cat = fits.open(source_cat_name_output)
    n_cols = (len((cat[1].data)[0]))
    for i in range (0,n_cols) :
        #i = i+1
        #ttypes.append('TTYPE'+str(i))
        #for ttype in ttypes :
        #print (cat[1].header[ttype])
        if cat[1].columns[i].name == 'ALPHA_SKY' :
            if filter_name == filters[0]:
                cat[1].columns[i].name = 'RA'
            else:
                cat[1].columns[i].name = 'RA'+'_'+(filter_name)

        elif cat[1].columns[i].name == 'DELTA_SKY' :
            if filter_name == filters[0]:
                cat[1].columns[i].name = 'DEC'
            else:
                cat[1].columns[i].name = 'DEC'+'_'+(filter_name)

        else:
            cat[1].columns[i].name = cat[1].columns[i].name+'_'+(filter_name)

    cat.writeto(source_cat_name_output,overwrite=True)

############################################################

def make_mask(frame, sex_source_cat, weight_frame, seg_map, mask_out, mask_out2, weight_out, flag_out, ra, dec, mode='cat'):

    X = get_header(frame,keyword='NAXIS1')
    Y = get_header(frame,keyword='NAXIS2')
    img = fits.open(frame)
    data = img[0].data
    data1 = data

    weight = fits.open(weight_frame)
    weight_data = weight[0].data

    mask = fits.open(seg_map)
    mask_data = mask[0].data
    mask_data[mask_data>0.5]=1
    mask_data[mask_data<0.5]=0
    data2 = data*0+mask_data
    data2[abs(weight_data)<1e-15]=1
    """
    if mode == 'cat' :
        cat = fits.open(sex_source_cat)
        table = cat[1].data
        N = len(table)
        ra_star = table['ALPHA_SKY']
        dec_star = table['DELTA_SKY']
        x = table['X_IMAGE']
        y = table['Y_IMAGE']
        mag = table['MAG_AUTO']
        fwhm = table['FWHM_IMAGE']
        flag = table['FLAGS']
        A = table['A_IMAGE']
        B = table['B_IMAGE']
        for ii in range (0,N) :
            r = math.sqrt((ra-ra_star[ii])**2+(dec-dec_star[ii])**2)
            if flag[ii] <= 9999 and r >= 0.0001/3600. and mag[ii] < 20:
                if x[ii] >= X or y[ii] >= Y :
                    continue
                x0 = int(x[ii])
                y0 = int(y[ii])

                mask_size = 1 #int(2*(26-mag[ii])*B[ii])

                #if mask_size < 5:
                #    mask_size = 5
                #if mask_size > 20:
                #    mask_size = 20

                for i in range(0,mask_size+1) :
                    for j in range(0,mask_size+1) :
                        rr = math.sqrt((i)**2+(j)**2)

                        if rr <= mask_size :
                            if x0+i < X and y0+j < Y :
                                data[y0+j][x0+i] = 1
                                data2[y0+j][x0+i] = 1
                            if x0+i < X and y0-j > 0 :
                                data[y0-j][x0+i] = 1
                                data2[y0-j][x0+i] = 1
                            if x0-i > 0 and y0+j < Y :
                                data[y0+j][x0-i] = 1
                                data2[y0+j][x0-i] = 1
                            if x0-i > 0 and y0-j > 0 :
                                data[y0-j][x0-i] = 1
                                data2[y0-j][x0-i] = 1
    """

    where_are_NaNs = np.isnan(data2)
    data2[where_are_NaNs] = 1
    img[0].data = (data2)
    img.writeto(mask_out,overwrite=True)

    data_flag = data2*64
    data_flag = data_flag.astype(np.int16)
    img[0].data = data_flag
    img.writeto(flag_out,overwrite=True)

    w = 1 - data2
    data1 = data1*w
    img[0].data = (data1)
    img.writeto(mask_out2,overwrite=True)

    data = w*weight_data
    img[0].data = (data)
    img.writeto(weight_out,overwrite=True)

############################################################

def make_detection_frame(gal_id, input_frame, weight_frame, fn, output_frame, backsize=64, backfiltersize=1, iteration=3):
    print ('- Making the detection frame for filter ', fn)
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    data_name = gal_data_name[gal_id]

    os.system('cp '+input_frame+' '+temp_dir+'temp_det.fits')
    for i in range(0,iteration):
        print ('+ iteration '+str(i))
        #os.system('mv temp.check.fits temp0.fits')
        if i == 0 :
            weight_command = '-WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE '+weight_frame+' -WEIGHT_THRESH 0.001 '
        elif i > 0 :
            weight_command = '-WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE '+temp_dir+'temp_weight'+str(i)+'.fits -WEIGHT_THRESH 0.001 '

        #print (weight_command)

        # make segmentation map
        #if i == 0 :
        command = SE_executable+' '+temp_dir+'temp_det.fits'+' -c '+str(external_dir)+'default.sex -CATALOG_NAME '+temp_dir+'temp_sex_cat.fits'+str(i)+'.fits '+ \
        '-PARAMETERS_NAME '+str(external_dir)+'sex_default.param -DETECT_MINAREA 5 -DETECT_MAXAREA 200 -DETECT_THRESH 1.5 -ANALYSIS_THRESH 1.5 ' + \
        '-DEBLEND_NTHRESH 16 -DEBLEND_MINCONT 0.005 ' + weight_command + \
        '-FILTER_NAME  '+str(external_dir)+'default.conv -STARNNW_NAME '+str(external_dir)+'default.nnw -PIXEL_SCALE ' + str(PIXEL_SCALES[filters[0]]) + ' ' \
        '-BACK_SIZE 256 -BACK_FILTERSIZE 3 -CHECKIMAGE_TYPE SEGMENTATION ' +  \
        '-CHECKIMAGE_NAME '+temp_dir+'temp_seg'+str(i)+'.fits'+' -VERBOSE_TYPE NORMAL'
        os.system(command)

        if i>0:
            img1 = fits.open(temp_dir+'temp_seg'+str(i)+'.fits')
            img2 = fits.open(temp_dir+'temp_seg'+str(i-1)+'.fits')
            data1 = img1[0].data
            data2 = img2[0].data
            data1 = data1+data2
            img1[0].data = data1
            img1.writeto(temp_dir+'temp_seg'+str(i)+'.fits',overwrite=True)

        command = SE_executable+' '+temp_dir+'temp_det.fits'+' -c '+str(external_dir)+'default.sex -CATALOG_NAME '+temp_dir+'temp_sex_cat.fits'+str(i)+'.fits '+ \
        '-PARAMETERS_NAME '+str(external_dir)+'sex_default.param -DETECT_MINAREA 5 -DETECT_MAXAREA 200 -DETECT_THRESH 2.0 -ANALYSIS_THRESH 2.0 ' + \
        '-DEBLEND_NTHRESH 16 -DEBLEND_MINCONT 0.005 ' + weight_command + \
        '-FILTER_NAME  '+str(external_dir)+'default.conv -STARNNW_NAME '+str(external_dir)+'default.nnw -PIXEL_SCALE ' + str(PIXEL_SCALES[filters[0]]) + ' ' \
        '-BACK_SIZE '+ str(backsize)+' -BACK_FILTERSIZE '+ str(backfiltersize)+' -CHECKIMAGE_TYPE BACKGROUND,-BACKGROUND,APERTURES ' +  \
        '-CHECKIMAGE_NAME '+temp_dir+'temp_back'+str(i)+'.fits,'+temp_dir+'temp_-back'+str(i)+'.fits,'+temp_dir+'temp_aper'+str(i)+'.fits'+' -VERBOSE_TYPE NORMAL'
        #print (command)
        os.system(command)

        #if i > 0 :
        #    attach_sex_tables([temp_dir+'temp_sex_cat.fits'+str(i-1)+'.fits',temp_dir+'temp_sex_cat.fits'+str(i)+'.fits'],temp_dir+'temp_sex_cat.fits'+str(i)+'.fits')

        make_mask(input_frame,temp_dir+'temp_sex_cat.fits'+str(i)+'.fits',weight_frame,temp_dir+'temp_seg'+str(i)+'.fits',\
            temp_dir+'temp_mask'+str(i+1)+'.fits',temp_dir+'temp_mask+'+str(i+1)+'.fits',temp_dir+'temp_weight'+str(i+1)+'.fits',\
            temp_dir+'temp_flag.fits'+str(i)+'.fits', ra, dec)

    img1 = fits.open(input_frame)
    img2 = fits.open(temp_dir+'temp_back'+str(iteration-1)+'.fits')
    data1 = img1[0].data
    data2 = img2[0].data
    data1 = data1-data2
    img1[0].data = data1
    img1.writeto(output_frame,overwrite=True)

    #os.system('rm temp*.fits')
    os.system('cp '+temp_dir+'temp_mask'+str(iteration-1)+'.fits'+' '+sex_dir+gal_name+'_'+fn+'_'+'mask'+'_cropped.fits')

############################################################

def make_source_cat(gal_id):
    print ('- Making Source Catalogues ... ')
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    data_name = gal_data_name[gal_id]
    methods = gal_methods[gal_id]

    i = -1
    for fn in filters:
        i=i+1
        os.system('cp '+external_dir+'sex_default.param '+external_dir+'default.param')
        detection_frame = detection_dir+gal_name+'_'+fn+'_'+'detection'+'_cropped.fits'

        main_frame = data_dir+gal_name+'_'+fn+'_cropped.fits'
        weight_frame = data_dir+gal_name+'_'+fn+'_cropped.weight.fits'

        #print ('DET', main_frame, weight_frame)
        psf_frame = psf_dir+data_name+'_psf_'+fn+'.inst.fits'
        make_detection_frame(gal_id,main_frame, weight_frame,fn,output_frame=detection_frame)
        make_fancy_png(detection_frame,detection_frame+'.jpg',zoom=2)
        if fn == filters[0]:
            detection_frame_main = detection_dir+gal_name+'_'+'detection'+'_cropped.fits'
            os.system('cp '+detection_frame+' '+detection_frame_main)

        frame = main_frame
        weight_map = data_dir+gal_name+'_'+fn+'_cropped.weight.fits'
        source_cat_name = sex_dir+gal_name+'_'+fn+'_source_cat.fits'
        source_cat_name_proc = sex_dir+gal_name+'_'+fn+'_source_cat_proc.fits'
        check_image_aper = sex_dir+gal_name+'_'+fn+'_check_image_apertures.fits'
        check_image_noback = sex_dir+gal_name+'_'+fn+'_check_image_-background.fits'
        check_image_back = sex_dir+gal_name+'_'+fn+'_check_image_background.fits'
        check_image_filtered = sex_dir+gal_name+'_'+fn+'_check_image_filtered.fits'
        check_image_segm = sex_dir+gal_name+'_'+fn+'_check_image_segm.fits'
        check_image_back_rms = sex_dir+gal_name+'_'+fn+'_check_image_back_rms.fits'

        source_cat_name_lsb = sex_dir+gal_name+'_'+fn+'_source_cat_lsb.fits'
        source_cat_name_proc = sex_dir+gal_name+'_'+fn+'_source_cat_proc_lsb.fits'
        check_image_aper_lsb = sex_dir+gal_name+'_'+fn+'_check_image_apertures_lsb.fits'
        check_image_noback_lsb = sex_dir+gal_name+'_'+fn+'_check_image_-background_lsb.fits'
        check_image_back_lsb = sex_dir+gal_name+'_'+fn+'_check_image_background_lsb.fits'
        check_image_filtered_lsb = sex_dir+gal_name+'_'+fn+'_check_image_filtered_lsb.fits'
        check_image_segm_lsb = sex_dir+gal_name+'_'+fn+'_check_image_segm_lsb.fits'
        check_image_back_rms_lsb = sex_dir+gal_name+'_'+fn+'_check_image_back_rms_lsb.fits'

        zp = ZPS[fn]
        gain = GAIN[fn]
        pix_size = PIXEL_SCALES[fn] #automatic


        weight_command = '-WEIGHT_TYPE  MAP_WEIGHT -WEIGHT_IMAGE '+weight_map+' -WEIGHT_THRESH 0.001'

        psf_dia_ref_pixel = 2*(APERTURE_SIZE[fn])/PIXEL_SCALES[fn]

        psf_dia_ref_pixel = int(psf_dia_ref_pixel*100+0.4999)/100
        apertures = (str(psf_dia_ref_pixel)+','+PHOTOM_APERS).split(',')
        n = len(apertures)
        shutil.copy(external_dir+'sex_default.param',external_dir+'default.param')
        params = open(external_dir+'default.param','a')
        params.write('MAG_APER('+str(n)+') #Fixed aperture magnitude vector [mag]\n')
        params.write('MAGERR_APER('+str(n)+') #RMS error vector for fixed aperture mag [mag]\n')
        params.write('FLUX_APER('+str(n)+') # Flux within a Kron-like elliptical aperture [count]\n')
        params.write('FLUXERR_APER('+str(n)+') #RMS error for AUTO flux [count]\n')
        #params.write('XPSF_IMAGE #X coordinate from PSF-fitting [pixel]\n')
        #params.write('YPSF_IMAGE #Y coordinate from PSF-fitting [pixel]\n')
        #params.write('ALPHAPSF_SKY #Right ascension of the fitted PSF (native) [deg]\n')
        #params.write('DELTAPSF_SKY #Declination of the fitted PSF (native) [deg]\n')
        #params.write('MAG_PSF #Magnitude from PSF-fitting [mag]\n')
        #params.write('MAGERR_PSF #RMS error vector for Magnitude from PSF-fitting [mag]\n')

        params.close()

        ####
        print (f"{bcolors.OKCYAN}- making source catalogs for point-sources"+ bcolors.ENDC)
        command = SE_executable+' '+detection_frame+','+frame+' -c '+external_dir+'default.sex -CATALOG_NAME '+source_cat_name+' '+ \
        '-PARAMETERS_NAME '+external_dir+'default.param -DETECT_MINAREA 5 -DETECT_THRESH 2.0 -ANALYSIS_THRESH 2.0 ' + \
        '-DEBLEND_NTHRESH 16 -DEBLEND_MINCONT 0.005 ' + weight_command + ' -PHOT_APERTURES '+str(psf_dia_ref_pixel)+','+str(PHOTOM_APERS)+' -GAIN ' + str(gain) + ' ' \
        '-MAG_ZEROPOINT ' +str(zp) + ' -BACKPHOTO_TYPE GLOBAL '+\
        '-FILTER Y -FILTER_NAME  '+external_dir+'tophat_1.5_3x3.conv -STARNNW_NAME '+external_dir+'default.nnw -PIXEL_SCALE ' + str(pix_size) + ' ' \
        '-BACK_SIZE 32 -BACK_FILTERSIZE 1 -CHECKIMAGE_TYPE APERTURES,FILTERED,BACKGROUND,-BACKGROUND,SEGMENTATION,BACKGROUND_RMS ' +  \
        '-CHECKIMAGE_NAME '+check_image_aper+','+check_image_filtered+','+check_image_back+','+check_image_noback+','+check_image_segm+\
        ','+check_image_back_rms+' -VERBOSE_TYPE NORMAL'+' -PSF_NAME '+psf_frame
        #print (command)
        os.system(command)

        ### galaxy mode
        print (f"{bcolors.OKCYAN}- making source catalogs for extended sources and LSBs"+ bcolors.ENDC)
        command_lsb = SE_executable+' '+frame+' -c '+external_dir+'default.sex -CATALOG_NAME '+source_cat_name_lsb+' '+ \
        '-PARAMETERS_NAME '+external_dir+'default.param -DETECT_MINAREA 50 -DETECT_THRESH 1.0 -ANALYSIS_THRESH 1.0 ' + \
        '-DEBLEND_NTHRESH 16 -DEBLEND_MINCONT 0.005 ' + weight_command + ' -PHOT_APERTURES '+str(psf_dia_ref_pixel)+','+str(PHOTOM_APERS)+' -GAIN ' + str(gain) + ' ' \
        '-MAG_ZEROPOINT ' +str(zp) + ' -BACKPHOTO_TYPE GLOBAL '+\
        '-FILTER Y -FILTER_NAME  '+external_dir+'tophat_1.5_3x3.conv -STARNNW_NAME '+external_dir+'default.nnw -PIXEL_SCALE ' + str(pix_size) + ' ' \
        '-BACK_SIZE 256 -BACK_FILTERSIZE 3 -CHECKIMAGE_TYPE APERTURES,FILTERED,BACKGROUND,-BACKGROUND,SEGMENTATION,BACKGROUND_RMS ' +  \
        '-CHECKIMAGE_NAME '+check_image_aper_lsb+','+check_image_filtered_lsb+','+check_image_back_lsb+','+check_image_noback_lsb+','+check_image_segm_lsb+\
        ','+check_image_back_rms_lsb+' -VERBOSE_TYPE NORMAL'+' -PSF_NAME '+psf_frame
        #print (command)
        os.system(command_lsb)

        ####

        make_fancy_png(check_image_aper,check_image_aper+'.jpg',zoom=2)
        make_fancy_png(check_image_back,check_image_back+'.jpg',zoom=2)
        make_fancy_png(check_image_filtered,check_image_filtered+'.jpg',zoom=2)
        make_fancy_png(check_image_segm,check_image_segm+'.jpg',zoom=2)

        prepare_sex_cat(source_cat_name,source_cat_name_proc,gal_name,fn,distance)
        prepare_sex_cat(source_cat_name_lsb,source_cat_name_proc,gal_name,fn,distance)


############################################################

def crossmatch(cat1,cat2,ra_param1,dec_param1,ra_param2,dec_param2,max_sep_arcsec,filter_name2,output_cat):

    cat = fits.open(cat1, ignore_missing_end=True)
    cat1_data = cat[1].data
    ra1 = cat1_data[ra_param1]
    dec1 = cat1_data[dec_param1]


    cat = fits.open(cat2, ignore_missing_end=True)
    cat2_data = cat[1].data
    ra2 = cat2_data[ra_param2]
    dec2 = cat2_data[dec_param2]

    c1 = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)
    c2 = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)
    join_func = join_skycoord(max_sep_arcsec * u.arcsec)
    j = join_func(c1, c2)

    os.system('cp '+cat1+' '+temp_dir+'temp1.fits')
    os.system('cp '+cat2+' '+temp_dir+'temp2.fits')

    expand_fits_table(temp_dir+'temp1.fits','JOIN_ID_'+filter_name2,j[0])
    expand_fits_table(temp_dir+'temp2.fits','JOIN_ID_'+filter_name2,j[1])

    t1 = Table.read(temp_dir+'temp1.fits',format='fits')
    t2 = Table.read(temp_dir+'temp2.fits',format='fits')
    t12 = table.join(t1, t2, keys='JOIN_ID_'+filter_name2)
    print (t12)
    t12.write(output_cat,overwrite=True)
    #print(t12)

    #cat2.writeto(output_cat,overwrite=True)

############################################################

def make_multiwavelength_cat(gal_id, mode='forced-photometry'):
    print ('- Making the MASTER Source Catalogue ')
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    data_name = gal_data_name[gal_id]

    if mode=='cross-match' or mode=='forced-photometry':
        main_cat = sex_dir+gal_name+'_'+filters[0]+'_source_cat_proc.fits'
        os.system('cp '+main_cat+' join.fits')
        for fn in filters[1:2]:
            cat = sex_dir+gal_name+'_'+fn+'_source_cat_proc.fits'
            crossmatch('join.fits',cat,'RA','DEC','RA_'+fn,'DEC_'+fn,CROSS_MATCH_RADIUS_ARCSEC,fn,'join.fits')

        os.system('mv join.fits '+cats_dir+gal_name+'_master_cat.fits')

    if mode=='forced-photometry':
        det_cat = sex_dir+gal_name+'_'+filters[0]+'_source_cat_proc_lsb.fits'
        #det_cat = cats_dir+gal_name+'_master_cat.fits'
        output = temp_dir+'join.fits'
        os.system('rm '+output)
        shutil.copy(det_cat,output)
        for fn in filters:
            photom_frame = data_dir+gal_name+'_'+fn+'_cropped.fits'
            mask_frame = sex_dir+gal_name+'_'+fn+'_'+'mask'+'_cropped.fits'
            back_rms_frame = sex_dir+gal_name+'_'+fn+'_check_image_back_rms.fits'
            print ("- Force photometry of frame in filter "+fn)
            forced_photometry(det_cat, photom_frame, mask_frame, back_rms_frame, fn,  output, mode='aperture-corr')
            #forced_photometry(output, photom_frame, mask_frame, back_rms_frame, fn,  output, mode='circular-aperture', aper_size_arcsec=0.5) #radius
            #forced_photometry(output, photom_frame, mask_frame, back_rms_frame, fn,  output, mode='circular-aperture', aper_size_arcsec=1)
            #forced_photometry(output, photom_frame, mask_frame, back_rms_frame, fn,  output, mode='circular-aperture', aper_size_arcsec=2)
            #forced_photometry(output, photom_frame, mask_frame, back_rms_frame, fn,  output, mode='circular-aperture', aper_size_arcsec=5)
            #forced_photometry(det_cat, photom_frame, mask_frame, back_rms_frame, fn,  output, mode='elliptical-aperture')

        os.system('mv '+output+' '+cats_dir+gal_name+'_master_cat_forced.fits')

        ####### lsb

        det_cat = sex_dir+gal_name+'_'+filters[0]+'_source_cat_proc.fits'
        #det_cat = cats_dir+gal_name+'_master_cat.fits'
        output = temp_dir+'join.fits'
        os.system('rm '+output)
        shutil.copy(det_cat,output)
        for fn in filters:
            photom_frame = data_dir+gal_name+'_'+fn+'_cropped.fits'
            mask_frame = sex_dir+gal_name+'_'+fn+'_'+'mask'+'_cropped.fits'
            back_rms_frame = sex_dir+gal_name+'_'+fn+'_check_image_back_rms.fits'
            print ("- Force photometry of frame in filter "+fn)
            #forced_photometry(det_cat, photom_frame, mask_frame, back_rms_frame, fn,  output, mode='aperture-corr')
            #forced_photometry(output, photom_frame, mask_frame, back_rms_frame, fn,  output, mode='circular-aperture', aper_size_arcsec=0.5) #radius
            #forced_photometry(output, photom_frame, mask_frame, back_rms_frame, fn,  output, mode='circular-aperture', aper_size_arcsec=1)
            forced_photometry(output, photom_frame, mask_frame, back_rms_frame, fn,  output, mode='circular-aperture', aper_size_arcsec=2)
            forced_photometry(output, photom_frame, mask_frame, back_rms_frame, fn,  output, mode='circular-aperture', aper_size_arcsec=5)
            forced_photometry(output, photom_frame, mask_frame, back_rms_frame, fn,  output, mode='circular-aperture', aper_size_arcsec=10)
            forced_photometry(output, photom_frame, mask_frame, back_rms_frame, fn,  output, mode='circular-aperture', aper_size_arcsec=20)
            #forced_photometry(det_cat, photom_frame, mask_frame, back_rms_frame, fn,  output, mode='elliptical-aperture')

        os.system('mv '+output+' '+cats_dir+gal_name+'_lsb_master_cat_forced.fits')

############################################################

def crop_fits_data(fits_file,ra,dec,crop_size):

    fits_data = fits_file[0].data
    fits_header = fits_file[0].header

    w = WCS(fits_header,fix=True)
    radius_pix = int(crop_size/2)
    x_center,y_center = w.all_world2pix(ra, dec,0)
    #print (x_center,y_center,radius_pix)
    llx = int(x_center - radius_pix)
    lly = int(y_center - radius_pix)
    urx = int(x_center + radius_pix)
    ury = int(y_center + radius_pix)
    #print (np.shape(hdu[0].data))
    dimx,dimy= np.shape(fits_data)[0],np.shape(fits_data)[1]
    #print (dimx,dimy)
    if llx<0:llx=0
    if lly<0:lly=0
    if urx>=dimx:urx=dimx-1
    if ury>=dimy:ury=dimy-1
    fits_header['NAXIS1'] = urx - llx
    fits_header['NAXIS2'] = ury - lly
    fits_header['CRPIX1'] = fits_header['CRPIX1']-llx
    fits_header['CRPIX2'] = fits_header['CRPIX2']-lly
    fits_data = fits_data[lly:ury,llx:urx]
    return fits_data, fits_header, lly, ury, llx, urx

############################################################

def forced_photometry(det_cat, photom_frame, mask_frame, back_rms_frame, fn, output, mode='aperture-corr', aper_size_arcsec=1):

    cat = fits.open(det_cat)
    cat_data = cat[1].data
    fits_file = fits.open(photom_frame)

    try:
        header = fits_file[0].header
        header['RADESYSa'] = header['RADECSYS']
        del header['RADECSYS']
        fits_file[0].header = header
        fits_file.writeto(photom_frame,overwrite=True)
    except:
        donothing=1

    fits_file = fits.open(photom_frame)
    fits_data = fits_file[0].data
    fits_header = fits_file[0].header

    mask_file = fits.open(mask_frame)
    mask_data = mask_file[0].data

    back_rms = fits.open(back_rms_frame)
    back_rms_data = back_rms[0].data

    #mask_data = np.array(1-mask_data)
    #print (mask_data[2000:2010,2000:2010])
    mask_data_bool = mask_data#.astype(np.bool_)
    mask_data_bool[mask_data>0.5] = 1
    mask_data_bool[mask_data<0.5] = 0
    mask_data_bool = mask_data_bool.astype(np.bool_)
    #print (mask_data_bool[2000:2010,2000:2010])
    #print (fits_data)

    zp = ZPS[fn]
    exptime = EXPTIME[fn]
    #zp = zp - 2.5*np.log10(exptime)
    error_data = (back_rms_data**2+fits_data/GAIN[fn])
    #error_data[error_data<=0] = 99999
    error_data = np.sqrt(error_data)/np.sqrt(exptime)
    #(np.sqrt((flux_err**2)*exptime+(flux_sky_sub*exptime / GAIN[fn] / (PSF_REF_RAD_FRAC[fn]))))/exptime

    RA = cat_data['RA']
    DEC = cat_data['DEC']
    fn_det = filters[0]
    MAG_det = cat_data['MAG_AUTO_'+fn_det]
    FLUX, FLUX_ERR, MAG, MAG_ERR, BACK_FLUX, BACK_FLUX_ERR = [], [], [], [], [], []
    #mag_ref = cat_data['MAG_APER_CORR_F606W']
    N_objects = len(RA)
    w=WCS(photom_frame)
    X = get_header(photom_frame,keyword='NAXIS1')
    Y = get_header(photom_frame,keyword='NAXIS2')

    if mode=='aperture-corr':
        aper_size = (APERTURE_SIZE[fn])/(PIXEL_SCALES[fn])
        sky_aper_size_1 = int(BACKGROUND_ANNULUS_START*FWHMS_ARCSEC[fn]/PIXEL_SCALES[fn])
        sky_aper_size_2 = int(sky_aper_size_1+BACKGROUND_ANNULUS_TICKNESS)

    elif mode=='circular-aperture':
        aper_size = 2*aper_size_arcsec/PIXEL_SCALES[fn]
        sky_aper_size_1 = int(BACKGROUND_ANNULUS_START*aper_size)
        sky_aper_size_2 = int(sky_aper_size_1+sky_aper_size_1)

    print ("- apreture radius for forced photometry is (pixels):", aper_size)
    print ("- sky annulus radi for forced photometry are (pixels):", sky_aper_size_1, sky_aper_size_2)
    a = 0
    for i in range(N_objects):
        ra = RA[i]
        dec = DEC[i]
        mag_det = MAG_det[i]
        #y, x, = w.all_world2pix(ra, dec, 0)

        crop_size = 5*sky_aper_size_2+10
        fits_file = fits.open(photom_frame)
        fits_data_cropped, data_header_cropped, lly, ury, llx, urx = crop_fits_data(fits_file, ra, dec, crop_size)
        error_data_cropped = error_data[lly:ury,llx:urx]
        mask_data_bool_cropped = mask_data_bool[lly:ury,llx:urx]

        w_cropped = WCS(data_header_cropped)
        y, x, = w_cropped.all_world2pix(ra, dec, 0)

        #fits_data_cropped = CutoutImage(fits_data,(x,y),(dx,dy))#[int(x)-sky_aper_size_2-2:int(x)+sky_aper_size_2+2,int(y)-sky_aper_size_2-2:int(y)+sky_aper_size_2+2]
        #mask_data_bool_cropped =  CutoutImage(mask_data_bool,(x,y),(dx,dy))#[int(x)-sky_aper_size_2-2:int(x)+sky_aper_size_2+2,int(y)-sky_aper_size_2-2:int(y)+sky_aper_size_2+2]
        #error_data_cropped = CutoutImage(error_data,(x,y),(dx,dy))#[int(x)-sky_aper_size_2-2:int(x)+sky_aper_size_2+2,int(y)-sky_aper_size_2-2:int(y)+sky_aper_size_2+2]

        #fits_data_cropped = fits_data_cropped.data
        #mask_data_bool_cropped = mask_data_bool_cropped.data
        #error_data_cropped = error_data_cropped.data

        #print (np.shape(fits_data_cropped))

        #N_check = 0
        #if (mag_det) < 22 and (mag_det > 18) :
        #    N_check = N_check + 1
        #    if N_check <= 10 :
        #        fig, ax = plt.subplots(1, 3, figsize=(10,3))
        #        ax[0].imshow(fits_data_cropped)
        #        ax[1].imshow(error_data_cropped)
        #        mask_data_bool_cropped_int = mask_data_bool_cropped
        #        mask_data_bool_cropped_int[mask_data_bool_cropped_int==True]=1
        #        mask_data_bool_cropped_int[mask_data_bool_cropped_int==False]=0
        #        ax[2].imshow(mask_data_bool_cropped_int)
        #        plt.savefig(check_plots_dir+'object_'+str(i)+'_'+fn+'.png')
        #        plt.close()

        aper = CircularAperture((x, y), aper_size)
        sky_aper = CircularAnnulus((x, y), sky_aper_size_1, sky_aper_size_2)
        aper_area = aper.area_overlap(data=fits_data_cropped,method='exact')
        sky_area = sky_aper.area_overlap(data=fits_data_cropped,mask=mask_data_bool_cropped,method='exact')
        #print (aper_area, sky_area, sky_area/aper_area)
        #print (error_data_cropped)

        flux, flux_err = aper.do_photometry(data=fits_data_cropped,error=error_data_cropped,method='exact')
        sky_flux, sky_flux_err = sky_aper.do_photometry(data=fits_data_cropped,mask=mask_data_bool_cropped,error=error_data_cropped,method='exact')
        #print (flux, sky_flux)
        #print (flux_err, sky_flux_err)
        flux = flux[0]
        flux_err = flux_err[0]
        sky_flux = sky_flux[0]
        sky_flux_err = sky_flux_err[0]
        flux_sky_sub = float(flux)-float(sky_flux)/(sky_area/aper_area)
        if mode == 'aperture-corr':
            flux_total = ((flux_sky_sub) / (PSF_REF_RAD_FRAC[fn]))[0]
        elif mode == 'circular-aperture':
             flux_total = (flux_sky_sub)
        #flux_err = (np.sqrt((flux_err**2)*exptime+(flux_sky_sub*exptime / GAIN[fn] / (PSF_REF_RAD_FRAC[fn]))))/exptime
        #print (flux, sky_flux)
        #print (flux_total)
        if flux_total <= 0 :
            FLUX.append(-99)
            FLUX_ERR.append(-99)
            MAG.append(-99)
            MAG_ERR.append(-99)
            BACK_FLUX.append(-99)
            BACK_FLUX_ERR.append(-99)
        else:
            mag = -2.5*np.log10((flux_total))+zp
            mag_err = 2.5*np.log10(((flux_total+flux_err)/flux_total))
            #print (flux, sky_flux, mag)
            FLUX.append(flux_total)
            FLUX_ERR.append(flux_err)
            MAG.append(mag)
            MAG_ERR.append(mag_err)
            BACK_FLUX.append(float(sky_flux))
            BACK_FLUX_ERR.append(float(sky_flux_err))
            #print ('- flux, sky-flux and magnitude for object are:', flux_err, sky_flux_err, mag_err)

        text = "+ Photometry for " + str(N_objects) + " sources in filter "+\
            fn+" in progress: " + str(int((i+1)*100/N_objects)) + "%"
        print ("\r" + text + " ", end='')
        #TIME.sleep (1)
        a = a + 1

    print('')

    #print (N_objects,len(MAG))
    if mode=='aperture-corr':
        expand_fits_table(output, 'F_FLUX_APER_CORR_'+fn,np.array(FLUX))
        expand_fits_table(output, 'F_FLUXERR_APER_CORR_'+fn,np.array(FLUX_ERR))
        expand_fits_table(output, 'F_MAG_APER_CORR_'+fn,np.array(MAG))
        expand_fits_table(output, 'F_MAGERR_APER_CORR_'+fn,np.array(MAG_ERR))
        expand_fits_table(output, 'F_BACK_FLUX_'+fn,np.array(BACK_FLUX))
        expand_fits_table(output, 'F_BACK_FLUX_ERR_'+fn,np.array(BACK_FLUX_ERR))

    elif mode=='circular-aperture':
        expand_fits_table(output, 'F_FLUX_APER_'+str(aper_size_arcsec)+'ARCSEC_'+fn,np.array(FLUX))
        expand_fits_table(output, 'F_FLUXERR_APER_'+str(aper_size_arcsec)+'ARCSEC_'+fn,np.array(FLUX_ERR))
        expand_fits_table(output, 'F_MAG_APER_'+str(aper_size_arcsec)+'ARCSEC_'+fn,np.array(MAG))
        expand_fits_table(output, 'F_MAGERR_APER_'+str(aper_size_arcsec)+'ARCSEC_'+fn,np.array(MAG_ERR))
        expand_fits_table(output, 'F_BACK_FLUX_'+str(aper_size_arcsec)+'ARCSEC_'+fn,np.array(BACK_FLUX))
        expand_fits_table(output, 'F_BACK_FLUX_ERR_'+str(aper_size_arcsec)+'ARCSEC_'+fn,np.array(BACK_FLUX_ERR))

############################################################

def copy_sims(gal_id):

    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    data_name = gal_data_name[gal_id]
    methods = gal_methods[gal_id]

    for fn in filters:
        for n in range(N_SIM_GCS) :

            gal_name_orig = gal_name
            gal_name = gal_name+'_SIM_'+str(n)
            (gal_params[gal_id])[0] = gal_name

            if 'USE_SUB_GAL' in methods:
                shutil.copy(art_dir+gal_name_orig+'_'+fn+'_ART_'+str(n)+'.science.fits',\
                    art_dir+gal_name+'_'+fn+'_cropped.fits')
                shutil.copy(data_dir+gal_name_orig+'_'+fn+'_cropped.weight.fits',\
                    art_dir+gal_name+'_'+fn+'_cropped.weight.fits')

            else :
                shutil.copy(art_dir+gal_name_orig+'_'+fn+'_ART_'+str(n)+'.science.fits',\
                    art_dir+gal_name+'_'+fn+'_cropped.fits')
                shutil.copy(data_dir+gal_name_orig+'_'+fn+'_cropped.weight.fits',\
                    art_dir+gal_name+'_'+fn+'_cropped.weight.fits')

            gal_name = gal_name_orig
            (gal_params[gal_id])[0] = gal_name

############################################################

def make_source_cat_for_sim(gal_id):

    print (f"{bcolors.OKCYAN}- source detection for the simulated data"+ bcolors.ENDC)

    copy_sims(gal_id)
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    data_name = gal_data_name[gal_id]
    methods = gal_methods[gal_id]

    global data_dir
    data_dir = art_dir
    tables = []
    tables2 = []

    for n in range(N_SIM_GCS) :

        gal_name_orig = gal_name
        gal_name = gal_name+'_SIM_'+str(n)
        (gal_params[gal_id])[0] = gal_name

        make_source_cat(gal_id)
        make_multiwavelength_cat(gal_id, mode='forced-photometry')

        fn_det = filters[0]
        art_cat_name_csv = art_dir+gal_name_orig+'_'+fn_det+'_ART'+str(n)+'_'+str(N_ART_GCS)+'GCs.full_info.csv'
        art_cat_name = art_dir+gal_name_orig+'_'+fn_det+'_ART'+str(n)+'_'+str(N_ART_GCS)+'GCs.full_info.fits'
        os.system('rm '+art_cat_name)
        csv_to_fits(art_cat_name_csv,art_cat_name)

        source_cat_with_art = cats_dir+gal_name+'_master_cat_forced.fits'
        crossmatch(art_cat_name,source_cat_with_art,'RA_GC','DEC_GC','RA','DEC',2.*PIXEL_SCALES[fn_det],fn_det,\
            art_dir+'temp'+str(n)+'.fits')
        tables.append(art_dir+'temp'+str(n)+'.fits')
        tables2.append(art_cat_name)

        (gal_params[gal_id])[0] = gal_name_orig
        gal_name = gal_name_orig


    attach_sex_tables(tables,art_dir+gal_name+'_'+fn_det+'_ALL_DET_ART_GCs.fits')
    attach_sex_tables(tables2,art_dir+gal_name+'_'+fn_det+'_ALL_ART_GCs.fits')

    data_dir = data_dir_orig
    #print (data_dir)

############################################################
