import os, sys
import math
import numpy as np
import matplotlib.pyplot as plt
from astroquery.mast import Observations
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from astropy.stats import sigma_clip
from astropy.visualization import *
from astropy.visualization import make_lupton_rgb
from astropy.table import Table, join_skycoord
from astropy import table
import photutils
from photutils.aperture import CircularAperture
from photutils.aperture import CircularAnnulus

from modules.pipeline_functions import *

def prepare_sex_cat(source_cat_name_input,source_cat_name_output,gal_name,filter_name,distance):
    main = fits.open(source_cat_name_input)
    sex_cat_data = main[1].data
    fn = filter_name
    #print (len(sex_cat_data))
    mask = ((sex_cat_data['FLAGS'] < 4) & \
    (sex_cat_data ['ELLIPTICITY'] < 0.9) & \
    (sex_cat_data ['MAG_AUTO'] > 20) & \
    (sex_cat_data ['MAG_AUTO'] < 27) & \
    (sex_cat_data ['FWHM_IMAGE'] < 999) & \
    (sex_cat_data ['FWHM_IMAGE'] > 0.5) )
    sex_cat_data = sex_cat_data[mask]
    #print (len(sex_cat_data))
    hdul = fits.BinTableHDU(data=sex_cat_data)
    hdul.writeto(source_cat_name_output,overwrite=True)

    mag_apers = np.array(sex_cat_data['MAG_APER'])
    magerr_apers = np.array(sex_cat_data['MAGERR_APER'])
    flux_apers = np.array(sex_cat_data['FLUX_APER'])
    fluxerr_apers = np.array(sex_cat_data['FLUXERR_APER'])
    psf_dia_ref_pixel = 2*(PSF_REF_RAD_ARCSEC[filter_name])[0]/PIXEL_SCALES[filter_name]
    psf_dia_ref_pixel = int(psf_dia_ref_pixel*1000+0.4999)/1000
    apertures = (str(psf_dia_ref_pixel)+','+PHOTOM_APERS).split(',')
    expand_fits_table(source_cat_name_output,'MAG_APER_REF',mag_apers[:,0])
    expand_fits_table(source_cat_name_output,'MAGERR_APER_REF',magerr_apers[:,0])

    expand_fits_table(source_cat_name_output,'FLUX_APER_REF',flux_apers[:,0])
    expand_fits_table(source_cat_name_output,'FLUXERR_APER_REF',fluxerr_apers[:,0])

    expand_fits_table(source_cat_name_output,'APER_RAD_REF_PIXEL',mag_apers[:,0]*0+float(apertures[0]))

    fraction_corr = 2.512*np.log10(1./(PSF_REF_RAD_FRAC[fn])[0])
    flux_fraction_corr = 1./(PSF_REF_RAD_FRAC[fn])[0]
    expand_fits_table(source_cat_name_output,'MAG_APER_CORR',mag_apers[:,0]-fraction_corr)
    expand_fits_table(source_cat_name_output,'MAGERR_APER_CORR',magerr_apers[:,0])

    expand_fits_table(source_cat_name_output,'FLUX_APER_CORR',flux_apers[:,0]*flux_fraction_corr)
    expand_fits_table(source_cat_name_output,'FLUXERR_APER_CORR',fluxerr_apers[:,0])

    # add size
    fwhm_obs = np.array(sex_cat_data['FWHM_WORLD'])*3600.0
    fwhm = (FWHMS_ARCSEC[filter_name])[0]
    fwhm_int = fwhm_obs**2-fwhm**2
    #print (fwhm_obs, fwhm, fwhm_int)
    fwhm_int[fwhm_int<0] = 0
    fwhm_int = np.sqrt(fwhm_int)
    Re = fwhm_int/2
    expand_fits_table(source_cat_name_output,'FWHM_INT',fwhm_int)
    expand_fits_table(source_cat_name_output,'Re_arcsec',Re)
    expand_fits_table(source_cat_name_output,'Re_pc',Re*distance*0.206265)

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

def make_detection_frame(gal_id, input_frame, weight_frame, fn, output_frame, backsize=16, backfiltersize=1, iteration=3):
    print ('- Making the detection frame for source detection ... ')
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    os.system('cp '+input_frame+' '+temp_dir+'temp_det.fits')
    for i in range(0,iteration):
        print ('+ iteration '+str(i))
        #os.system('mv temp.check.fits temp0.fits')
        if i == 0 :
            weight_command = '-WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE '+weight_frame+' -WEIGHT_THRESH 1 '
        elif i > 0 :
            weight_command = '-WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE '+temp_dir+'temp_weight'+str(i)+'.fits -WEIGHT_THRESH 1 '

        #print (weight_command)

        # make segmentation map
        #if i == 0 :
        command = 'sex '+temp_dir+'temp_det.fits'+' -c '+str(input_dir)+'default.sex -CATALOG_NAME '+temp_dir+'temp_sex_cat.fits'+str(i)+'.fits '+ \
        '-PARAMETERS_NAME '+str(input_dir)+'default.param -DETECT_MINAREA 3 -DETECT_THRESH 1.0 -ANALYSIS_THRESH 1.0 ' + \
        '-DEBLEND_NTHRESH 4 -DEBLEND_MINCONT 0.1 ' + weight_command + \
        '-FILTER_NAME  '+str(input_dir)+'default.conv -STARNNW_NAME '+str(sex_dir)+'default.nnw -PIXEL_SCALE ' + str(PIXEL_SCALES[filters[0]]) + ' ' \
        '-BACK_SIZE 256 -BACK_FILTERSIZE 3 -CHECKIMAGE_TYPE SEGMENTATION ' +  \
        '-CHECKIMAGE_NAME '+temp_dir+'temp_seg'+str(i)+'.fits'
        os.system(command)

        if i>0:
            img1 = fits.open(temp_dir+'temp_seg'+str(i)+'.fits')
            img2 = fits.open(temp_dir+'temp_seg'+str(i-1)+'.fits')
            data1 = img1[0].data
            data2 = img2[0].data
            data1 = data1+data2
            img1[0].data = data1
            img1.writeto(temp_dir+'temp_seg'+str(i)+'.fits',overwrite=True)

        command = 'sex '+temp_dir+'temp_det.fits'+' -c '+str(input_dir)+'default.sex -CATALOG_NAME '+temp_dir+'temp_sex_cat.fits'+str(i)+'.fits '+ \
        '-PARAMETERS_NAME '+str(input_dir)+'default.param -DETECT_MINAREA 4 -DETECT_THRESH 2.0 -ANALYSIS_THRESH 2.0 ' + \
        '-DEBLEND_NTHRESH 64 -DEBLEND_MINCONT 0.0001 ' + weight_command + \
        '-FILTER_NAME  '+str(input_dir)+'default.conv -STARNNW_NAME '+str(sex_dir)+'default.nnw -PIXEL_SCALE ' + str(PIXEL_SCALES[filters[0]]) + ' ' \
        '-BACK_SIZE '+ str(backsize)+' -BACK_FILTERSIZE '+ str(backfiltersize)+' -CHECKIMAGE_TYPE BACKGROUND,-BACKGROUND,APERTURES ' +  \
        '-CHECKIMAGE_NAME '+temp_dir+'temp_back'+str(i)+'.fits,'+temp_dir+'temp_-back'+str(i)+'.fits,'+temp_dir+'temp_aper'+str(i)+'.fits'
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

    i = -1
    for fn in filters:
        i=i+1

        detection_frame = detection_dir+gal_name+'_'+fn+'_'+'detection'+'_cropped.fits'
        main_frame = data_dir+gal_name+'_'+fn+'_cropped.fits'
        weight_frame = data_dir+gal_name+'_'+fn+'_cropped.weight.fits'
        make_detection_frame(gal_id,main_frame, weight_frame,fn,output_frame=detection_frame)
        make_fancy_png(detection_frame,detection_frame+'.jpg',zoom=2)
        if fn == filters[0]:
            detection_frame_main = detection_dir+gal_name+'_'+'detection'+'_cropped.fits'
            os.system('cp '+detection_frame+' '+detection_frame_main)

        frame = data_dir+gal_name+'_'+fn+'_cropped.fits'
        weight_map = data_dir+gal_name+'_'+fn+'_cropped.weight.fits'
        source_cat_name = sex_dir+gal_name+'_'+fn+'_source_cat.fits'
        source_cat_name_proc = sex_dir+gal_name+'_'+fn+'_source_cat_proc.fits'
        check_image_aper = sex_dir+gal_name+'_'+fn+'_check_image_apertures.fits'
        check_image_noback = sex_dir+gal_name+'_'+fn+'_check_image_-background.fits'
        check_image_back = sex_dir+gal_name+'_'+fn+'_check_image_background.fits'
        check_image_filtered = sex_dir+gal_name+'_'+fn+'_check_image_filtered.fits'
        check_image_segm = sex_dir+gal_name+'_'+fn+'_check_image_segm.fits'
        check_image_back_rms = sex_dir+gal_name+'_'+fn+'_check_image_back_rms.fits'
        zp = ZPS[fn]
        gain = GAIN[fn]
        pix_size = PIXEL_SCALES[fn] #automatic

        weight_command = '-WEIGHT_TYPE  MAP_WEIGHT -WEIGHT_IMAGE '+weight_map+' -WEIGHT_THRESH 0.001'

        psf_dia_ref_pixel = 2*(PSF_REF_RAD_ARCSEC[fn])[0]/PIXEL_SCALES[fn]
        psf_dia_ref_pixel = int(psf_dia_ref_pixel*100+0.4999)/100
        apertures = (str(psf_dia_ref_pixel)+','+PHOTOM_APERS).split(',')
        n = len(apertures)
        params = open(input_dir+'default.param','a')
        params.write('MAG_APER('+str(n)+') #Fixed aperture magnitude vector [mag]\n')
        params.write('MAGERR_APER('+str(n)+') #RMS error vector for fixed aperture mag [mag]\n')
        params.write('FLUX_APER('+str(n)+') # Flux within a Kron-like elliptical aperture [count]\n')
        params.write('FLUXERR_APER('+str(n)+') #RMS error for AUTO flux [count]\n')
        params.close()

        # 2, 1, 1 for JWST
        command = 'sex '+detection_frame+','+frame+' -c '+input_dir+'default.sex -CATALOG_NAME '+source_cat_name+' '+ \
        '-PARAMETERS_NAME '+input_dir+'default.param -DETECT_MINAREA 4 -DETECT_THRESH 1.5 -ANALYSIS_THRESH 1.0 ' + \
        '-DEBLEND_NTHRESH 32 -DEBLEND_MINCONT 0.0001 ' + weight_command + ' -PHOT_APERTURES '+str(psf_dia_ref_pixel)+','+str(PHOTOM_APERS)+' -GAIN ' + str(gain) + ' ' \
        '-MAG_ZEROPOINT ' +str(zp) + ' -BACKPHOTO_TYPE GLOBAL '+\
        '-FILTER Y -FILTER_NAME  '+input_dir+'tophat_1.5_3x3.conv -STARNNW_NAME '+input_dir+'default.nnw -PIXEL_SCALE ' + str(pix_size) + ' ' \
        '-BACK_SIZE 32 -BACK_FILTERSIZE 1 -CHECKIMAGE_TYPE APERTURES,FILTERED,BACKGROUND,-BACKGROUND,SEGMENTATION,BACKGROUND_RMS ' +  \
        '-CHECKIMAGE_NAME '+check_image_aper+','+check_image_filtered+','+check_image_back+','+check_image_noback+','+check_image_segm+\
        ','+check_image_back_rms
        #print (command)
        os.system(command)

        ####

        make_fancy_png(check_image_aper,check_image_aper+'.jpg',zoom=2)
        make_fancy_png(check_image_back,check_image_back+'.jpg',zoom=2)
        make_fancy_png(check_image_filtered,check_image_filtered+'.jpg',zoom=2)
        make_fancy_png(check_image_segm,check_image_segm+'.jpg',zoom=2)

        prepare_sex_cat(source_cat_name,source_cat_name_proc,gal_name,fn,distance)

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
    t12.write(output_cat,overwrite=True)
    #print(t12)

    #cat2.writeto(output_cat,overwrite=True)

############################################################

def make_multiwavelength_cat(gal_id, mode='forced-photometry'):
    print ('- Making the MASTER Source Catalogue ')
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]

    if mode=='cross-match' or mode=='forced-photometry':
        main_cat = sex_dir+gal_name+'_'+filters[0]+'_source_cat_proc.fits'
        os.system('cp '+main_cat+' join.fits')
        for fn in filters[1:2]:
            cat = sex_dir+gal_name+'_'+fn+'_source_cat_proc.fits'
            crossmatch('join.fits',cat,'RA','DEC','RA_'+fn,'DEC_'+fn,0.25,fn,'join.fits')

        os.system('mv join.fits '+cats_dir+gal_name+'_master_cat.fits')

    if mode=='forced-photometry':
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
            forced_photometry(det_cat, photom_frame, mask_frame, back_rms_frame, fn, output)

        os.system('mv '+output+' '+cats_dir+gal_name+'_master_cat_forced.fits')

############################################################

def forced_photometry(det_cat, photom_frame, mask_frame, back_rms_frame, fn, output):

    cat = fits.open(det_cat)
    cat_data = cat[1].data
    fits_file = fits.open(photom_frame)
    fits_data = fits_file[0].data
    mask_file = fits.open(mask_frame)
    mask_data = mask_file[0].data

    back_rms = fits.open(back_rms_frame)
    back_rms_data = back_rms[0].data

    error_data = (back_rms_data)
    #mask_data = np.array(1-mask_data)
    #print (mask_data[2000:2010,2000:2010])
    mask_data_bool = mask_data#.astype(np.bool_)
    mask_data_bool[mask_data>0.5] = 1
    mask_data_bool[mask_data<0.5] = 0
    mask_data_bool = mask_data_bool.astype(np.bool_)
    #print (mask_data_bool[2000:2010,2000:2010])
    #print (fits_data)
    zp = ZPS[fn]
    exptime = fits_file[0].header['EXPTIME']
    #zp = zp - 2.5*np.log10(exptime)
    RA = cat_data['RA']
    DEC = cat_data['DEC']
    FLUX, FLUX_ERR, MAG, MAG_ERR, BACK_FLUX, BACK_FLUX_ERR = [], [], [], [], [], []
    #mag_ref = cat_data['MAG_APER_CORR_F606W']
    N_objects = len(RA)
    w=WCS(photom_frame)
    X = get_header(photom_frame,keyword='NAXIS1')
    Y = get_header(photom_frame,keyword='NAXIS2')
    aper_size = ((PSF_REF_RAD_ARCSEC[fn])[0])/PIXEL_SCALES[fn]
    sky_aper_size_1 = int(3*((FWHMS_ARCSEC[fn])[0])/PIXEL_SCALES[fn]+0.5)
    sky_aper_size_2 = int(sky_aper_size_1+20)
    print ("- apreture radius for forced photometry is (pixels):", aper_size)
    print ("- sky annulus radi for forced photometry are (pixels):", sky_aper_size_1, sky_aper_size_2)

    for i in range(N_objects):
        ra = RA[i]
        dec = DEC[i]
        x, y, = w.all_world2pix(ra, dec,0)
        #print (x,y)
        #if (x<1) or (y<1) or (x>X-1) or (y>Y-1):
        #    print (' *** object with RA and Dec ',str(ra),', ',str(dec), 'is outside of the frame.')
        #    mag = -99
        #    mag_err = -99
        #    MAG.append(mag)
        #    MAG_ERR.append(mag_err)
        #    continue

        aper = CircularAperture((x, y), aper_size)
        sky_aper = CircularAnnulus((x, y), sky_aper_size_1, sky_aper_size_2)
        aper_area = aper.area_overlap(data=fits_data,method='exact')
        sky_area = sky_aper.area_overlap(data=fits_data,mask=mask_data_bool,method='exact')
        #print (aper_area, sky_area, sky_area/aper_area)

        #try :
        perform_photom = 1
        if perform_photom == 1:
            flux, flux_err = aper.do_photometry(data=fits_data,method='exact',error=error_data)
            sky_flux, sky_flux_err = sky_aper.do_photometry(data=fits_data,mask=mask_data_bool,method='exact',error=error_data)
            flux = flux[0]
            flux_err = flux_err[0]
            sky_flux = sky_flux[0]
            sky_flux_err = sky_flux_err[0]

            flux_sky_sub = float(flux)-float(sky_flux)/(sky_area/aper_area)
            flux_total = (flux_sky_sub) / (PSF_REF_RAD_FRAC[fn])[0]
            flux_err = np.sqrt(flux_err**2+(flux_sky_sub / GAIN[fn] / (PSF_REF_RAD_FRAC[fn])[0]))

            #print (flux, sky_flux)

            mag = -2.5*np.log10((flux_total))+zp
            mag_err = 2.5*np.log10(((flux+flux_err)/flux))

            #print (flux, sky_flux, mag)

            FLUX.append(flux_total)
            FLUX_ERR.append(flux_err)
            MAG.append(mag)
            MAG_ERR.append(mag_err)
            BACK_FLUX.append(float(sky_flux_err))
            BACK_FLUX_ERR.append(sky_flux_err)
            print ('\n- Photometry for object with ID:',i)
            #print ('- flux, sky-flux and magnitude for object are:', flux, sky_flux, mag)
            #print ('- errors om flux, sky-flux and magnitude for object are:', flux_err, sky_flux_err, mag_err)

    #print (N_objects,len(MAG))
    expand_fits_table(output, 'F_FLUX_APER_CORR_'+fn,np.array(FLUX))
    expand_fits_table(output, 'F_FLUXERR_APER_CORR_'+fn,np.array(FLUX_ERR))
    expand_fits_table(output, 'F_MAG_APER_CORR_'+fn,np.array(MAG))
    expand_fits_table(output, 'F_MAGERR_APER_CORR_'+fn,np.array(MAG_ERR))
    expand_fits_table(output, 'F_BACK_FLUX_'+fn,np.array(BACK_FLUX))
    expand_fits_table(output, 'F_BACK_FLUX_ERR_'+fn,np.array(BACK_FLUX_ERR))
