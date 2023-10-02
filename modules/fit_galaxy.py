import sys, os
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from astropy.wcs import WCS
from astropy import wcs
from astropy.io import fits
import os.path
from scipy import signal
from scipy import ndimage
from pylab import *
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.image as mpimg
from matplotlib.ticker import ScalarFormatter
from astropy.io import fits
from matplotlib.ticker import StrMethodFormatter
import matplotlib.ticker as ticker
from astropy.stats import sigma_clip
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from scipy import ndimage, misc
import os.path
import random
from astropy.visualization import *
from modules.pipeline_functions import *
from modules.initialize import *

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


def fit_galaxy_sersic_all_filters(gal_id):

    print (f"{bcolors.OKCYAN}- Sersic modeling of the host galaxy"+ bcolors.ENDC)
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    data_name = gal_data_name[gal_id]

    gal_cat = cats_dir+gal_name+'_sersic_params.csv'
    cat = open(gal_cat, 'w')

    RA, DEC, Re, mag, n, PA, q = {}, {}, {}, {}, {}, {}, {}

    for fn in filters:
        if fn == filters[0]: cat.write('RA_'+fn+', Dec'+fn+', Re_'+fn+', mag_'+fn+', n_'+fn+', PA_'+fn+', q_'+fn)
        else: cat.write(', Re_'+fn+', mag_'+fn+', n_'+fn+', PA_'+fn+', q_'+fn)
    cat.write('\n')

    for fn in filters:

        if fn == filters[0]:
            constraint = 'None'
            constraint_from_det_filter = 'None'
        else:
            constraint = constraint_from_det_filter

        print ('- constraints on fitting for filter ', fn, 'are: ', constraint_from_det_filter)
        main_data = data_dir+gal_name+'_'+fn+'_gal_cropped.fits'
        #back_data = data_dir+gal_name+'_'+fn+'_cropped.fits'
        weight_data = data_dir+gal_name+'_'+fn+'_gal_cropped.weight.fits'
        pix_size = PIXEL_SCALES[fn]
        zp = ZPS[fn]
        gain = GAIN[fn]

        update_header(main_data,'GAIN',gain)
        #update_header(back_data,'GAIN',gain)
        update_header(weight_data,'GAIN',gain)

        scale = 0.005*distance #arcsec/kpc
        try :\
        #if True:
            ra_, dec_, Re_, mag_, n_, PA_, q_ = fit_galaxy_sersic(main_data,weight_data,ra,dec,gal_name,fn,pix_size,fit_dir,zp,plotting=True,\
            r_cut=GAL_FRAME_SIZE[fn], r_cut_fit=GAL_FRAME_SIZE[fn], scale=scale, constraint=constraint, data_name=data_name)
            ra_ = float(ra_)
            dec_ = float(dec_)
            #if USE_GAL_SUB_IMG == True:

        except Exception as error:
            # handle the exception
            print("An exception occurred:", error)
            print (f"{bcolors.FAIL}*** Fitting failed."+ bcolors.ENDC)
            ra_, dec_, Re_, mag_, n_, PA_, q_ = -99, -99, -99, -99, -99, -99, -99

        Re_ = round(Re_, 2)
        mag_ = round(mag_,2)
        n_ = round(n_, 2)
        PA_ = round(PA_, 2)
        q_ = round(q_, 2)
        ra_ = round(ra_, 6)
        dec_ = round(dec_, 6)

        if (fn == filters[0]) and (Re_ > 0) and (mag_ > 0) and (n_ > 0):
            constraint_from_det_filter = [ra_, dec_, Re_, mag_, n_, PA_, q_]

        RA[fn] = ra_
        DEC[fn] = dec_
        Re[fn] = Re_
        mag[fn] = mag_
        n[fn] = n_
        PA[fn] = PA_
        q[fn] = q_

        if fn == filters[0]: cat.write(str(ra_)+', '+str(dec_)+', '+str(Re_)+', '+str(mag_)+', '+str(n_)+', '+str(PA_)+', '+str(q_))
        else: cat.write(', '+str(Re_)+', '+str(mag_)+', '+str(n_)+', '+str(PA_)+', '+str(q_))

    cat.write('\n')
    cat.close()

############################################################

def get_ellipse(xc,yc,a,b,pa,res=360):
    '''
    Returns an ellipse with the given parameters.
    pa in degrees and context of astronomy.
    '''
    #Function is developed by Aku Venhola
    try:
        dum=len(xc)
    except:
        xc,yc,a,b,pa=np.array([xc]),np.array([yc]),np.array([a]),np.array([b]),np.array([pa])
    pa = (pa+90.) / 360. *2.*np.pi
    theta = np.arange(res+1)/float(res)*2.*np.pi
    theta = np.array([theta for i in range(len(xc))]).T
    xc = np.array([xc for i in range(res+1)])
    yc = np.array([yc for i in range(res+1)])
    a = np.array([a for i in range(res+1)])
    b = np.array([b for i in range(res+1)])
    pa = np.array([pa for i in range(res+1)])
    r = a*b / np.sqrt((b*np.cos(theta-pa))**2. + (a*np.sin(theta-pa))**2.)
    elx = xc + r * np.cos(theta)
    ely = yc + r * np.sin(theta)
    return elx,ely

############################################################

def cut(fitsfile, ra, dec, radius_pix, objectname='none', filtername='none',  back=0, overwrite=False, \
    blur=0, label=''):
    '''
    Cuts the images from the coadds.
    '''
    #Original function is developed by Aku Venhola, modified by Teymoor Saifollahi
    hdu = fits.open(fitsfile)
    hdu2 = fits.open(fitsfile)
    w=WCS(fitsfile)

    radius_pix = int(radius_pix * 1)
    x_center,y_center = w.all_world2pix(ra, dec,0)
    #print (x_center,y_center,radius_pix)
    llx = int(x_center - radius_pix)
    lly = int(y_center - radius_pix)
    urx = int(x_center + radius_pix)
    ury = int(y_center + radius_pix)
    dimx,dimy= len(hdu[0].data[0]),len(hdu[0].data)

    if llx<0:llx=0
    if lly<0:lly=0
    if urx>=dimx:urx=dimx-1
    if ury>=dimy:ury=dimy-1
    #print ('+++ Cropping area',llx,lly,'(x0,y0)',urx,ury,'(x1,y1)')
    #if (urx - llx != ury - lly) :
    #    return 0
    #print (lly,ury,llx,urx)
    object = hdu[0].data[lly:ury,llx:urx]*(1.0)
    #object2 = np.log10(hdu[0].data[lly:ury,llx:urx]*(1.0e15))  #scaling
    template = hdu
    if blur > 0 :
        #object = gaussian_filter(object,sigma=blur)
        object = ndimage.median_filter(object, size=blur)
    #template2 = hdu2
    template[0].header['NAXIS1'] = urx - llx
    template[0].header['NAXIS2'] = ury - lly
    #template[0].header['EXPTIME'] = 1.0
    #template[0].header['GAIN'] = 1.0
    #print (urx - llx,ury - lly)
    template[0].header['CRPIX1'] = hdu[0].header['CRPIX1'] -llx
    template[0].header['CRPIX2'] = hdu[0].header['CRPIX2'] -lly

    object = object-back
    where_are_NaNs = isnan(object)
    object[where_are_NaNs] = 99
    template[0].data = object
    #template2[0].data = object2
    template.writeto(objectname+'_'+filtername+'_cropped'+str(label)+'.fits', overwrite=True)
    #template2.writeto(objectname+'_'+filtername+'_scaled'+'_'+filtername_+'.fits', clobber = overwrite)
    #print ('Saved postage stamp to '+objectname+'_'+filtername+'_cropped'+str(label)+'.fits')
    output_frame = objectname+'_'+filtername+'_cropped.fits'
    # no jpg conversion
    #print 'DONE!'
    del template,hdu
    #del template2,hdu2
    w=WCS(objectname+'_'+filtername+'_cropped'+str(label)+'.fits')
    x_gal_center,y_gal_center = w.all_world2pix(ra, dec,0)

    return output_frame, x_gal_center, y_gal_center


############################################################

def mask_stars (frame, weight_frame, ra, dec, objname, filtername, zp, q=1, pa=0, blurred=0, label='', type='LSB') :
    #frame = main_data
    #print ('+++ Making mask frame for '+frame)
    X = get_header(frame,keyword='NAXIS1')
    Y = get_header(frame,keyword='NAXIS2')

    os.system('rm '+fit_dir+objname+'_'+filtername+'.sex_cat.fits')

    """
    os.system(SE_executable+' '+frame+' -c '+external_dir+'default.sex -PARAMETERS_NAME '+external_dir+'sex_default.param -CATALOG_NAME '+\
    fit_dir+objname+'_'+filtername+'.sex_cat.fits -DETECT_MINAREA 20 -DETECT_THRESH 1.5 -ANALYSIS_THRESH 1.5 -MAG_ZEROPOINT '+str(zp)+' '+\
    '-DEBLEND_NTHRESH 8 -DEBLEND_MINCONT 0.005 -FILTER_NAME '+external_dir+'default.conv '+' -STARNNW_NAME '+external_dir+'default.nnw '+\
    '-BACK_SIZE 256 -BACK_FILTERSIZE 3  -CHECKIMAGE_TYPE APERTURES,SEGMENTATION -CHECKIMAGE_NAME '+\
    fit_dir+objname+'_'+filtername+'.check_aper_2.fits,'+fit_dir+objname+'_'+filtername+'_galfit_seg_map_1.fits')
    """

    os.system(SE_executable+' '+frame+' -c '+external_dir+'default.sex -PARAMETERS_NAME '+external_dir+'sex_default.param -CATALOG_NAME '+\
    fit_dir+objname+'_'+filtername+'.sex_cat.fits -DETECT_MINAREA 8 -DETECT_MAXAREA 10000 -DETECT_THRESH 1.5 -ANALYSIS_THRESH 1.5 -MAG_ZEROPOINT '+str(zp)+' '+\
    '-DEBLEND_NTHRESH 16 -DEBLEND_MINCONT 0.005 -FILTER_NAME '+external_dir+'default.conv '+' -STARNNW_NAME '+external_dir+'default.nnw '+\
    '-BACK_SIZE 32 -CHECKIMAGE_TYPE APERTURES,SEGMENTATION -CHECKIMAGE_NAME '+\
    fit_dir+objname+'_'+filtername+'.check_aper_2.fits,'+fit_dir+objname+'_'+filtername+'_galfit_seg_map.fits')

    """
    img1 = fits.open(fit_dir+objname+'_'+filtername+'_galfit_seg_map_1.fits')
    img2 = fits.open(fit_dir+objname+'_'+filtername+'_galfit_seg_map_2.fits')
    data1 = img1[0].data
    data2 = img2[0].data
    data1 = data1+data2
    img1[0].data = data1
    img1.writeto(fit_dir+objname+'_'+filtername+'_galfit_seg_map.fits',overwrite=True)
    """

    cat = fits.open(fit_dir+objname+'_'+filtername+'.sex_cat.fits')
    img = fits.open(frame)
    weight = fits.open(weight_frame)
    img2 = fits.open(fit_dir+objname+'_'+filtername+'_galfit_seg_map.fits')
    table = cat[1].data
    data = img[0].data
    data2 = img2[0].data
    weight_data = weight[0].data
    #data = data*0
    N = len(table)

    #print (image)
    for i in range (0,N) :
        params = table[i]
        mag = params['MAG_AUTO']
        ra_star = params['ALPHA_SKY']
        dec_star = params['DELTA_SKY']
        x = params['X_IMAGE']
        y = params['Y_IMAGE']
        fwhm = params['FWHM_IMAGE']
        flag = params['FLAGS']
        A = params['A_IMAGE']
        B = params['B_IMAGE']
        #print (ra_star, dec_star, mag)
        r = math.sqrt((ra-ra_star)**2+(dec-dec_star)**2)
        if flag <= 9999 and r >= -0.01/3600. and mag < 18:
            #print (r*3600., ra_star, dec_star, mag)
            if x >= X or y >= Y :
                continue
            x = int(x)
            y = int(y)
            #fwhm = int(fwhm+0.5)
            #mask_size = int(2.0*(fwhm))
            #if blurred==0 :
            #    mask_size = int(15*B)**1
            #if blurred==1 :
            mask_size = int(2*(26-mag)*B)

            if mask_size > 20:
                mask_size = 20

            for i in range(0,mask_size+1) :
                for j in range(0,mask_size+1) :
                    rr = math.sqrt((i)**2+(j)**2)
                    if rr <= mask_size :
                        if x+i < X and y+j < Y :
                            data2[y+j][x+i] = 1
                            #data2[y+j][x+i] = min_value
                        if x+i < X and y-j > 0 :
                            data2[y-j][x+i] = 1
                            #data2[y-j][x+i] = min_value
                        if x-i > 0 and y+j < Y :
                            data2[y+j][x-i] = 1
                            #data2[y+j][x-i] = min_value
                        if x-i > 0 and y-j > 0 :
                            data2[y-j][x-i] = 1
                            #data2[y-j][x-i] = min_value

    where_are_NaNs = isnan(data)
    data2[where_are_NaNs] = 1
    data2[data2>0.5] = 1
    data2[weight_data<1e-9] = 1
    img2[0].data = (data2)
    img2.writeto(fit_dir+objname+'_'+filtername+'_masked.fits',overwrite=True)

    if type == 'LSB':
        median_filter_data = abs(median_filter_array(data,fsize=7))
        sigma_data = sqrt(1./weight_data+0*median_filter_data/GAIN[filtername])
        where_are_NaNs = isnan(sigma_data)
        sigma_data[where_are_NaNs] = 10e+8
        weight[0].data = (sigma_data)
        weight.writeto(fit_dir+objname+'_'+filtername+'.sigma.fits',overwrite=True)

    else:
        median_filter_data = abs(median_filter_array(data,fsize=7))
        sigma_data = sqrt(1./weight_data+1*median_filter_data/GAIN[filtername])
        where_are_NaNs = isnan(sigma_data)
        sigma_data[where_are_NaNs] = 10e+8
        weight[0].data = (sigma_data)
        weight.writeto(fit_dir+objname+'_'+filtername+'.sigma.fits',overwrite=True)

    mask = 1 - data2
    data = data * mask
    #data2 = np.log10(data2)
    #data2 = 10**(data2)
    img[0].data = (data)
    img.writeto(fit_dir+objname+'_'+filtername+'_masked+.fits',overwrite=True)

    return fit_dir+objname+'_'+filtername+'_masked.fits',\
           fit_dir+objname+'_'+filtername+'_masked+.fits',\
           fit_dir+objname+'_'+filtername+'.check.fits',\
           fit_dir+objname+'_'+filtername+'.sigma.fits'

############################################################

def sersic(r, ne, re, n, bkg) :
    return ne * np.exp(-1*(1.9992*n-0.3271)*((r/re)**(1./n)-1)) + bkg

############################################################

def make_weight_map(frame,weight_file,back,back_std) :
    main = fits.open(frame)
    data = main[0].data
    weight = data
    data = data-back
    data[data < 0] = 0
    weight[weight > 1] = 0
    gain = get_header(frame,'CCDGAIN')
    sigma = (back_std**2+data)
    weight = sigma #1./(sigma)
    #weight = (weight*1000)
    #weight = weight**2
    main[0].data = weight
    main.writeto(weight_file,overwrite=True)

############################################################

def change_weight_map(weight_in,weight_out,make_weight) :
    main = fits.open(weight_in)
    weight = main[0].data
    #weight = weight**0.1
    weight[weight>0] = weight[weight>0]**make_weight#**0.05#**0.5
    #median = np.median(weight[weight>0])
    #weights = weight-1
    #weight[weight<0] = 0
    #weight[weight<median] = 0
    #weight = weight**(1./3.)
    main[0].data = weight
    main.writeto(weight_out,overwrite=True)

############################################################

def estimate_frame_back(frame,gal_name,filtername) :

    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    data_name = gal_data_name[gal_id]

    X = get_header(frame,keyword='NAXIS1')
    Y = get_header(frame,keyword='NAXIS2')

    main = fits.open(frame)
    fits_data = main[0].data

    median_list = list()
    i = 0
    pixel_values=[]

    fits_data[int(X/2.-X/4.):int(X/2.+X/4.),int(Y/2.-Y/4.):int(Y/2.+Y/4.)] = -999999
    fits_data = fits_data[abs(fits_data)>-1]
    back_data = fits_data[fits_data>-99999]
    back_data = back_data[abs(back_data)>0]
    max_plot = sqrt(len(back_data))*10

    #print (len(fits_data), len(back_data), len(fits_data)/len(back_data))

    pixel_values = np.array(back_data)
    plt.figure()
    pixel_values = sigma_clip(pixel_values,3,maxiters=3)
    #plt.hist(median_list,bins=100,color='blue')
    #pixel_values = sigma_clip(pixel_values,2,maxiters=3)
    #median_list = sigma_clip(median_list,2,maxiters=10)
    plt.hist(pixel_values,bins='auto',color='green')
    median = np.nanmedian(pixel_values)
    std = np.nanstd(pixel_values)
    plt.xlim([median-1*std,median+1*std])
    plt.plot([median,median],[0,max_plot],'k-')
    plt.plot([median+1*std,median+1*std],[0,max_plot],'k--')
    plt.plot([median-1*std,median-1*std],[0,max_plot],'k--')

    plt.savefig(check_plots_dir+gal_name+'_'+filtername+'_hist_back.png',\
        bbox_inches='tight', pad_inches = 0, dpi=100)
    plt.close()

    print ('- inital sky background value and its RMS are: '+ str(median) + ', '+str(std))

    return median, std

############################################################

def make_galfit_feedme_file_sersic(main_file, mask_file, sigma_file, psf_file, objname,\
    ra, dec, reff, filtername, fit_x1, fit_x2, fit_y1, fit_y2, r_mag, sersic_index, pos_angel, axis_ratio, nuc, sky,\
        zp, pix_size, x_gal_center, y_gal_center) :
    print ('- Making galfit configuration file for galaxy ' + str(objname))
    #print (fit_x1,fit_x2, fit_y1, fit_y2)
    os.system('rm '+fit_dir+objname+'_galfit.conf')
    x = get_header(main_file,keyword='NAXIS1')
    y = get_header(main_file,keyword='NAXIS2')
    fits_file = fits.open(main_file)
    exptime = EXPTIME[filtername]
    gain = GAIN[filtername]
    zp = zp - 2.5*np.log10(exptime)
    galfit_conf = open(fit_dir+objname+'_galfit.conf','w')
    galfit_conf.write('# IMAGE and GALFIT CONTROL PARAMETERS\n')
    galfit_conf.write('A) '+main_file+'\n')
    galfit_conf.write('B) '+fit_dir+objname+'_'+filtername+'_galfit_imgblock'+'.fits\n')
    galfit_conf.write('C) '+sigma_file+'\n')
    #galfit_conf.write('C) None'+'\n')
    galfit_conf.write('D) '+psf_file+'\n')
    galfit_conf.write('E) 1\n')
    galfit_conf.write('F) '+mask_file+'\n')
    galfit_conf.write('G) none\n')
    galfit_conf.write('H) '+str(fit_x1)+' '+str(fit_x2)+' '+str(fit_y1)+' '+str(fit_y2)+'\n')
    galfit_conf.write('I) '+str(100)+' '+str(100)+'\n')
    galfit_conf.write('J) ' + str(zp) +'\n')
    galfit_conf.write('K) '+str(pix_size)+' '+str(pix_size)+'\n')
    galfit_conf.write('O) regular \n')
    galfit_conf.write('P) 0\n\n')
    galfit_conf.write('# Sersic\n')
    galfit_conf.write('0) sersic\n')
    galfit_conf.write('1) '+str(x_gal_center)+' '+str(y_gal_center)+' 1 1\n')
    galfit_conf.write('3) '+str(r_mag)+' 1\n')
    galfit_conf.write('4) '+str(reff)+' 1\n')
    galfit_conf.write('5) '+str(sersic_index)+' 1\n')
    galfit_conf.write('6) '+str(0.0000)+' 0\n')
    galfit_conf.write('7) '+str(0.0000)+' 0\n')
    galfit_conf.write('8) '+str(0.0000)+' 0\n')
    galfit_conf.write('9) '+str(axis_ratio)+' 1\n')
    galfit_conf.write('10) '+str(pos_angel)+' 1\n')
    galfit_conf.write('Z) 0\n\n')

    if nuc != -1 :
        print ('- galfit model: *sersic + psf + sky')
        galfit_conf.write('# psf\n')
        galfit_conf.write('0) psf\n')
        galfit_conf.write('1) '+str(x/2)+' '+str(y/2)+' 0 0\n')
        galfit_conf.write('3) '+str(float(nuc)+2.)+' 1\n')
        galfit_conf.write('Z) 0\n\n')
    else :
        print ('- galfit model: *sersic + sky')

    galfit_conf.write('# sky\n')
    galfit_conf.write('0) sky\n')
    galfit_conf.write('1) '+str(sky)+' 1\n')
    galfit_conf.write('2) '+str(0.000)+' 1\n')
    galfit_conf.write('3) '+str(0.000)+' 1\n')
    galfit_conf.write('Z) 0\n\n')

    galfit_conf.close()

############################################################

def make_galfit_feedme_file_sersic_constrained(main_file, mask_file, sigma_file, psf_file, objname,\
    ra, dec, reff, filtername, fit_x1, fit_x2, fit_y1, fit_y2, r_mag, sersic_index, pos_angel, axis_ratio, nuc,  sky,\
    zp, pix_size, x_gal_center, y_gal_center) :
    print ('- Making galfit configuration file for galaxy ' + str(objname))
    #print (fit_x1,fit_x2, fit_y1, fit_y2)
    os.system('rm '+fit_dir+objname+'_galfit.conf')
    x = get_header(main_file,keyword='NAXIS1')
    y = get_header(main_file,keyword='NAXIS2')
    fits_file = fits.open(main_file)
    exptime = EXPTIME[filtername]
    gain = GAIN[filtername]
    zp = zp - 2.5*np.log10(exptime)
    galfit_conf = open(fit_dir+objname+'_galfit.conf','w')
    galfit_conf.write('# IMAGE and GALFIT CONTROL PARAMETERS\n')
    galfit_conf.write('A) '+main_file+'\n')
    galfit_conf.write('B) '+fit_dir+objname+'_'+filtername+'_galfit_imgblock'+'.fits\n')
    galfit_conf.write('C) '+sigma_file+'\n')
    #galfit_conf.write('C) None'+'\n')
    galfit_conf.write('D) '+psf_file+'\n')
    galfit_conf.write('E) 1\n')
    galfit_conf.write('F) '+mask_file+'\n')
    galfit_conf.write('G) none\n')
    galfit_conf.write('H) '+str(fit_x1)+' '+str(fit_x2)+' '+str(fit_y1)+' '+str(fit_y2)+'\n')
    galfit_conf.write('I) '+str(100)+' '+str(100)+'\n')
    galfit_conf.write('J) ' + str(zp) +'\n')
    galfit_conf.write('K) '+str(pix_size)+' '+str(pix_size)+'\n')
    galfit_conf.write('O) regular \n')
    galfit_conf.write('P) 0\n\n')
    galfit_conf.write('# Sersic\n')
    galfit_conf.write('0) sersic\n')
    galfit_conf.write('1) '+str(x_gal_center)+' '+str(y_gal_center)+' 0 0\n')
    galfit_conf.write('3) '+str(r_mag)+' 1\n')
    galfit_conf.write('4) '+str(reff)+' 1\n')
    galfit_conf.write('5) '+str(sersic_index)+' 1\n')
    galfit_conf.write('6) '+str(0.0000)+' 0\n')
    galfit_conf.write('7) '+str(0.0000)+' 0\n')
    galfit_conf.write('8) '+str(0.0000)+' 0\n')
    galfit_conf.write('9) '+str(axis_ratio)+' 0\n')
    galfit_conf.write('10) '+str(pos_angel)+' 0\n')
    galfit_conf.write('Z) 0\n\n')

    if nuc != -1 :
        print ('- galfit model: *sersic + psf + sky')
        galfit_conf.write('# psf\n')
        galfit_conf.write('0) psf\n')
        galfit_conf.write('1) '+str(x/2)+' '+str(y/2)+' 0 0\n')
        galfit_conf.write('3) '+str(float(nuc)+2.)+' 1\n')
        galfit_conf.write('Z) 0\n\n')
    else :
        print ('- galfit model: *sersic + sky')

    galfit_conf.write('# sky\n')
    galfit_conf.write('0) sky\n')
    galfit_conf.write('1) '+str(sky)+' 1\n')
    galfit_conf.write('2) '+str(0.000)+' 1\n')
    galfit_conf.write('3) '+str(0.000)+' 1\n')
    galfit_conf.write('Z) 0\n\n')

    galfit_conf.close()

############################################################

def run_galfit_sersic (main_file, objname, filter_name) :

    print ('- Running galfit for ' + str(objname))
    #os.system('rm '+'fit.log')
    #os.system('rm '+fit_dir+'galfit.0*')
    os.system('rm '+fit_dir+objname+'_'+filter_name+'_galfit_imgblock*fits')
    os.system(galfit_executable+' '+fit_dir+objname+'_galfit.conf')

    if os.path.isfile(fit_dir+objname+'_'+str(filter_name)+'galfit_imgblock_model.fits') :
        os.system('rm '+fit_dir+objname+'_'+str(filter_name)+'galfit_imgblock_model.fits')
    if os.path.isfile(fit_dir+objname+'_'+str(filter_name)+'galfit_imgblock_res.fits') :
        os.system('rm '+fit_dir+objname+'_'+str(filter_name)+'galfit_imgblock_res.fits')
    if os.path.isfile(fit_dir+objname+'_'+str(filter_name)+'galfit_imgblock_data.fits') :
        os.system('rm '+fit_dir+objname+'_'+str(filter_name)+'galfit_imgblock_data.fits')

    model_frame = fit_dir+objname+'_'+filter_name+'_'+'galfit_imgblock_model.fits'
    res_frame = fit_dir+objname+'_'+filter_name+'_'+'galfit_imgblock_res.fits'
    data_frame = fit_dir+objname+'_'+filter_name+'_'+'galfit_imgblock_data.fits'

    if os.path.isfile(fit_dir+objname+'_'+filter_name+'_galfit_imgblock'+'.fits') :
        imgblock = fits.open(fit_dir+objname+'_'+filter_name+'_galfit_imgblock'+'.fits')
        model = fits.open(main_file)
        res = fits.open(main_file)
        data = fits.open(main_file)
        data[0].data = imgblock[1].data
        model[0].data = imgblock[2].data
        res[0].data = imgblock[3].data
        model.writeto(model_frame, overwrite=True)
        res.writeto(res_frame, overwrite=True)
        data.writeto(data_frame, overwrite=True)
    else :
        print (fit_dir+objname+'_'+filter_name+'_galfit_imgblock'+'.fits does not exist.')

    return model_frame, res_frame, data_frame

def read_numbers(input_string):
    accepted_chars = ['.','1','2','3','4','5','6','7','8','9','0','-','+']
    output_string = ''
    for c in input_string :
        if c in accepted_chars:
            output_string = output_string+c
        elif c == 'e':
            output_string = output_string+'e'
        else :
            output_string = output_string +' '
    return output_string


def read_galfit_sersic (main_file, objname, filtername) :
    #print ('--- Reading galfit output values from fit.log')
    log = open('fit.log','r')
    frame = main_file
    i = 0
    n = 0
    X = get_header(main_file,keyword='NAXIS1')
    Y = get_header(main_file,keyword='NAXIS2')

    mag_1, reff_1, d_mag_1, d_reff_1, pa, d_pa, axis_ratio, d_axis_ratio, \
    sky, sky_dx, sky_dy, d_sky, d_sky_dx, d_sky_dy = \
    (-99,-99,-99,-99,-99,-99,-99,-99,-99,-99,-99,-99,-99,-99)
    sersic_index_1, d_sersic_index_1 = -99,-99
    x, y = -99,-99

    for lines in log :
        i = i + 1
        if frame in lines :
            n = 1
            continue
        if n > 0 :
            n = n + 1

        if n == 6 :
            if ('*' in str(lines)) :
                conv = 0
                continue
                print ('--- *warning: galfit solution has not been converged.')
            elif n == 6 :
                conv = 1

        lines_numerics = read_numbers(lines[8:])
        params = lines_numerics.split()
        #print (lines_numerics)
        #print (params)

        if n == 6:
            x = float(params[0])
            y = float(params[1])
            mag_1 = float(params[2])
            reff_1 = float(params[3])
            sersic_index_1 = float(params[4])
            axis_ratio = float(params[5])
            pa = float(params[6])

        if n == 7 and conv == 1 :
            d_mag_1 = float(params[2])
            d_reff_1 = float(params[3])
            d_sersic_index_1 = float(params[4])
            d_axis_ratio = float(params[5])
            d_pa = float(params[6])

        if n == 8 :
            sky = float(params[2])
            sky_dx = float(params[3])
            sky_dy = float(params[4])

        if n == 9 :
            d_sky = float(params[0])
            d_sky_dx = float(params[1])
            d_sky_dy = float(params[2])

    return mag_1, reff_1, d_mag_1, d_reff_1, sersic_index_1, d_sersic_index_1, X, Y, \
        x, y, pa, d_pa, axis_ratio, d_axis_ratio, sky, sky_dx, sky_dy, d_sky, d_sky_dx, d_sky_dy


############################################################

def make_radial_profile(data,ellipse,exptime=1,mask=None,sn_stop=1./3.,rad_stop=None,binsize=5,sky=False) :
    '''
    Makes a radial profile using the input paramateres and outputs
    the (masked) clipped values for each bin.

    input:

    ellipse     :    [xc,yc,a,b,pa] (pa should be in rads fron x axis)
    sn_stop     :    at whic signal to noise the profile should stop
    sky         :    it True, measures the sky level and variations at three reffs in bins

    returns:
    rad, flux, errors, (skyval, skynoise)
    '''
    #Function is developed by Aku Venhola

    min_step   = 5
    n_sky_bins = 12
    dist_sky   = 4. #REFFS
    nx,ny = len(data[0]),len(data)
    xc,yc,A,B,PA = ellipse
    fluxbins = []
    errors   = []
    rs = []
    #Define elliptical coordinates
    x,y = np.ogrid[-yc:float(ny)-yc,-xc:float(nx)-xc]
    #print (nx, ny)
    angles = np.arctan(y/x)
    angles = np.pi/2.-angles
    angles[:yc,:] = angles[:yc,:]+np.pi
    angles[yc,:xc] = np.pi
    distance = np.sqrt(x**2.+y**2.)
    d, a = distance , angles
    edist = d / ( B / np.sqrt((B*np.cos(a-PA))**2.+(A*np.sin(a-PA))**2.))
    edist[yc,xc] = 0.
    #Iterate through the bins
    #r1,r2 = 0., 0.+binsize
    r1, r2 = 0, binsize
    step = 1
    while (step >= 0):
        uplim  = edist < r2
        lowlim = edist >= r1
        ind = uplim*lowlim
        bin_pix = data[ind]
        #if mask != None:
        bin_pix = bin_pix[ mask[ind] == 0  ]
        error = np.std(bin_pix)/np.sqrt(float(len(bin_pix)))
        if len(bin_pix)>5:
            bin_pix = sigma_clip(bin_pix,3,maxiters=3)
        level = np.mean(bin_pix)
        #store the bins values
        errors.append(error)
        fluxbins.append(level)
        rs.append((r1+r2)/2.)
        #Check if the time to stop has come
        #r1 += binsize
        #r2 += binsize
        r1 = r2
        r2 += binsize
        area = np.pi * ((r2*r2*(B/A))-(r1*r1*(B/A)))
        if rad_stop==None:
            if ( (level / error < sn_stop ) & (step>min_step) ) | (step*binsize>0.9*len(data)/2.):
                step=-10
        else:
            if r2 > rad_stop:
                step=-10
        step+=1
    # Calculate reff:
    fluxbins,errors,rs = np.array(fluxbins),np.array(errors),np.array(rs)
    reff = A

    # Deal with the sky (if asked):
    if sky:
        dthet      = 2.*np.pi/n_sky_bins
        sky_bins   = []
        sky_bins_values = []
        sky_radius = int( dist_sky * reff )
        if (sky_radius+binsize >= len(data)/2):
            sky_radius = len(data)/2-binsize
        max_r = sky_radius+binsize > edist
        min_r = sky_radius-binsize < edist
        rind = min_r*max_r
        for i in range(n_sky_bins):
            min_the =    i*dthet   < angles
            max_the = (i+1.)*dthet > angles
            ind_the = min_the*max_the
            ind     = ind_the*rind
            #if mask.any() == None:
            sky_pix = data[ind][mask[ind] == 0 ]
            #else:
            #sky_pix = data[ind][mask[ind] == 0 ]

            sky_bins_values.append('None')
            not_nan = ~np.isnan(sky_pix)
            sky_pix = sigma_clip(sky_pix[not_nan],3,maxiters=5)
            sky_bins.append(np.median(sky_pix))

        not_nan = ~np.isnan(sky_bins)
        sky_bin = sigma_clip(np.array(sky_bins)[not_nan],3,maxiters=5)
        sky_noise = np.std(np.array(sky_bin))
        sky_level = np.median(np.array(sky_bin))

        return rs,fluxbins,errors,sky_level,sky_noise,sky_bin, sky_bins_values, area
    else:
        return rs,fluxbins,errors, 0, 0, 0, 0, area

############################################################

def fit_galaxy_sersic(main_data,weight_data,ra,dec,obj_name,filter_name,pix_size,fit_dir,zp,\
    r_cut,r_cut_fit,scale,constraint='None',blur_frame=0,plotting=False,data_name='') :

    print ('\n+ Sersic fitting for the galaxy : '+str(obj_name))

    Re, n, PA, q, Ie = [], [], [], [], []
    e_Re, e_n, e_PA, e_q, e_Ie = [], [], [], [], []

    #rint ('- cropping ')
    cropped_frame = main_data
    weight_frame = weight_data
    #print (cropped_frame)
    w=WCS(main_data)
    c = ((pix_size)**2)
    fn = filter_name

    ##########################

    if os.path.exists(psf_dir+data_name+'_psf_'+fn+'.inst.fits'):
        psf_frame = psf_dir+data_name+'_psf_'+fn+'.inst.fits'
    else:
        psf_frame = 'None'

    ##########################
    print ('- estimating sky flux')
    back_average, back_std = estimate_frame_back(cropped_frame, obj_name, filter_name)

    print ('- masking ')
    if 'LSB' in comments:
        fit_type = 'LSB'
    else:
        fit_type = 'Normal'
        #fit_type = 'LSB'

    mask_frame, masked_frame, check_frame, sigma_frame = mask_stars(cropped_frame, weight_frame, ra, dec, obj_name, filter_name, zp, type=fit_type)
    #sigma_frame='None'

    print ('- fitting ')
    #running galfit
    os.system('rm fit.log')
    os.system('rm galfit.0*')
    fit_x1 = 1 # int(x_gal_center-r_cut_fit/2.)
    fit_y1 = 1 #int(y_gal_center-r_cut_fit/2.)
    fit_x2 = r_cut_fit #int(x_gal_center+r_cut_fit/2.)
    fit_y2 = r_cut_fit #int(y_gal_center+r_cut_fit/2.)

    if constraint == 'None':
        if ('LSB' in comments) or ('DWARF' in comments) :
            ra_c = ra
            dec_c = dec
            re_c = 1.5/scale/pix_size
            mag_c = -16 + 5*np.log10(distance*1e+5)
            n_c = 1
            pa_c = 0
            q_c = 1
            x_gal_center,y_gal_center = w.all_world2pix(ra, dec,0)

        elif ('MASSIVE' in comments) :
            ra_c = ra
            dec_c = dec
            re_c = 4.0/scale/pix_size
            mag_c = -20 + 5*np.log10(distance*1e+5)
            n_c = 1
            pa_c = 0
            q_c = 1
            x_gal_center,y_gal_center = w.all_world2pix(ra, dec,0)

        else:
            print ('*** galaxy type is not specified in the comments. Assuming a dwarf galaxy ... ')
            ra_c = ra
            dec_c = dec
            re_c = 1.5/scale/pix_size
            mag_c = -16 + 5*np.log10(distance*1e+5)
            n_c = 1
            pa_c = 0
            q_c = 1
            x_gal_center,y_gal_center = w.all_world2pix(ra, dec,0)

        make_galfit_feedme_file_sersic(cropped_frame,mask_frame,sigma_frame,psf_frame,\
        obj_name, ra_c, dec_c, re_c, filter_name, fit_x1, fit_x2, fit_y1, fit_y2, \
        mag_c, n_c, pa_c, q_c, -1, back_average, zp, pix_size, x_gal_center, y_gal_center)

    else:
        ra_c = float(constraint[0])
        dec_c = float(constraint[1])
        re_c = float(constraint[2])/scale/pix_size
        mag_c = float(constraint[3])
        n_c = float(constraint[4])
        pa_c = float(constraint[5])
        q_c = float(constraint[6])
        x_gal_center,y_gal_center = w.all_world2pix(ra_c, dec_c,0)

        make_galfit_feedme_file_sersic_constrained(cropped_frame,mask_frame,sigma_frame,psf_frame,\
        obj_name, ra_c, dec_c, re_c, filter_name, fit_x1, fit_x2, fit_y1, fit_y2, \
        mag_c, n_c, pa_c, q_c, -1, back_average, zp, pix_size, x_gal_center, y_gal_center)

    model_frame, res_frame, data_frame = run_galfit_sersic(cropped_frame, obj_name, filter_name)

    mag_best, re_best, d_mag_best, d_re_best, n_best, d_n_best, X, Y, \
    x_best, y_best, pa_best, d_pa_best, axis_ratio_best, d_axis_ratio_best, sky_best, \
    sky_dx_best, sky_dy_best, d_sky_best, d_sky_dx_best, d_sky_dy_best = \
    read_galfit_sersic(cropped_frame,obj_name,filter_name)

    print ('- fiting paremeters are (X, Y, Mag, Re, n, PA, q, Sky):')
    print ('-', x_best, y_best, mag_best, re_best, n_best, pa_best, axis_ratio_best, sky_best)
    #print (d_mag_best, d_re_best, d_n_best, d_pa_best, d_axis_ratio_best, d_sky_best)

    re_best_arcsec = re_best * pix_size
    re_best_kpc = re_best * pix_size * scale

    d_re_best_arcsec = d_re_best * pix_size
    d_re_best_kpc = d_re_best * pix_size * scale

    pa_best_corr = pa_best

    sersic_cat = open(fit_dir+obj_name+'_'+filter_name+'_sersic_params.csv','w')
    #sersic_cat.write('Re,n,PA,q,Ie,err_Re,err_n,err_PA,err_q,err_Ie\n')
    sersic_cat.write(str(re_best_kpc)+','+str(n_best)+','+str(pa_best_corr)+','+str(axis_ratio_best)+','+str(mag_best)+','+\
        str(d_re_best_kpc)+','+str(d_n_best)+','+str(d_pa_best)+','+str(d_axis_ratio_best)+','+str(d_mag_best)+','+\
        str(x_best)+','+str(y_best))
    sersic_cat.close()

    if plotting == True :
        x_best = int(x_best+0.5)
        y_best = int(y_best+0.5)

        ### plotting galaxy light profile + sersci fit

        plt.rc('axes', labelsize=36)
        plt.rc('xtick', labelsize=36)
        plt.rc('ytick', labelsize=36)

        plt.rc('axes', linewidth=1.7)

        plt.rcParams['xtick.major.width']=1.7
        plt.rcParams['xtick.major.size']=10
        plt.rcParams['xtick.minor.size']=6

        plt.rcParams['ytick.major.width']=1.7
        plt.rcParams['ytick.major.size']=10
        plt.rcParams['ytick.minor.size']=6
        plt.rc('text', usetex=True)

        bin_size=int(5)

        img = fits.open(data_frame)
        data = img[0].data

        fig, ax = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [2.0, 1]})
        fig1, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4,figsize=(12, 4), \
            gridspec_kw={'wspace':0.05, 'hspace':0}, squeeze=True) #figsize=(11, 4),
        fig.subplots_adjust(hspace=0)
        fig2, ax5 = plt.subplots(1,1,figsize=(8,8),frameon=False)

        img2 = fits.open(model_frame)
        model = img2[0].data

        img3 = fits.open(mask_frame)
        mask = img3[0].data

        img4 = fits.open(res_frame)
        res = img4[0].data

        img5 = fits.open(masked_frame)
        masked = img5[0].data

        exptime = EXPTIME[fn]

        ellipse = [x_best,y_best,re_best,axis_ratio_best*re_best,(90+pa_best)/360.*2.*np.pi]
        rs,fluxbins,errors,sky_level,sky_noise,sky_bin, sky_bins_values, area\
            = make_radial_profile(data,ellipse,exptime,mask=mask,rad_stop=GAL_FRAME_SIZE[fn],binsize=bin_size,sky=True)
        fluxbins0 = fluxbins
        fluxbins = (fluxbins) - sky_best
        e = np.sqrt(errors**2 + sky_noise**2)

        plt.figure()
        plt.plot(rs,fluxbins0,'k.')
        plt.plot([0,np.max(rs)],[back_average,back_average],'r--',label='sky (initial guess)')
        plt.plot([0,np.max(rs)],[sky_best, sky_best],'b--', label='sky estimated by galfit')
        plt.legend(loc='upper right')
        plt.xlabel('radius [pixels]')
        plt.ylabel('flux')
        plt.savefig(check_plots_dir+obj_name+'_'+filter_name+'_flux_profile.png',bbox_inches='tight', pad_inches = 0, dpi=100)
        plt.close()

        rs_m,fluxbins_m,errors_m,sky_level_m,sky_noise_m,sky_bin_m, sky_bins_values_m, area_m\
            = make_radial_profile(model,ellipse,exptime,mask=mask*0,rad_stop=GAL_FRAME_SIZE[fn],binsize=bin_size,sky=True)
        fluxbins0_m = fluxbins
        fluxbins_m = (fluxbins_m) - sky_best
        e_m = np.sqrt(errors_m**2 + sky_noise_m**2)

        rs_res,fluxbins_res,errors_res,sky_level_res,sky_noise_res,sky_bin_res, sky_bins_values_res, area_res\
            = make_radial_profile(res,ellipse,exptime,mask=mask,rad_stop=GAL_FRAME_SIZE[fn],binsize=bin_size,sky=True)
        fluxbins0_res = fluxbins
        fluxbins_res = (fluxbins_res)
        e_res = np.sqrt(errors_res**2 + sky_noise_res**2)

        #####

        magbins = -2.5*np.log10((fluxbins)/c)+zp
        magbins_up = -2.5*np.log10((fluxbins+e)/c)+zp
        magbins_down = -2.5*np.log10((fluxbins-e)/c)+zp

        magbins_m = -2.5*np.log10((fluxbins_m)/c)+zp
        magbins_res = -2.5*np.log10((fluxbins_res)/c)+zp

        #####

        l = 'R$_e$ = ' + str(re_best_kpc)[:4] +' kpc, n = '+str(n_best)[:4]

        #m_half_flux = mag_best+0.756
        flux_within_re = (10.**((mag_best-zp)*-0.4))/2.
        Ie_best = flux_within_re / (3.141592*re_best*re_best*axis_ratio_best)

        f = sersic(rs,Ie_best,re_best,n_best,0)
        m = -2.5*np.log10(f/c)+zp
        #ax[0].plot(rs*pix_size*scale,m,color='red',markersize=3, lw=6, label='best-fit: '+l,zorder=3)
        #ax[1].plot(rs*pix_size*scale,magbins-m,color='red',markersize=1, lw=5)

        ax[0].plot(rs*pix_size*scale,magbins,color='black',markersize=1, lw=5, label=obj_name+' ('+str(filter_name)+')')
        ax[0].plot(rs*pix_size*scale,magbins_up,'k--',markersize=1, lw=3)
        ax[0].plot(rs*pix_size*scale,magbins_down,'k--',markersize=1, lw=3)

        ax[0].plot(rs_m*pix_size*scale,magbins_m,color='red',markersize=3, label='best-fit: '+l, lw=5,zorder=3)
        ax[1].plot(rs*pix_size*scale,magbins-magbins_m,color='black',markersize=1, lw=5)

        ax[0].legend(loc='upper right', fontsize=30)
        ax[0].yaxis.labelpad = 26
        ax[1].plot([-2,20],[0,0],color='red',lw=5)

        #ax[0].set_xlim([-0.001,scale*r_cut_fit*pix_size+0.05])
        ax[0].set_xlim([-0.001,10.001])
        #plt.xlim([0,10])
        ##ax[0].set_ylim(29.45,23.0)
        ax[0].set_ylim(np.nanmax(magbins)+0.25,np.nanmin(magbins)-0.25)
        ax[1].set_ylim(-0.55,0.55)
        ax[1].set_xlabel('R [kpc]')
        ax[0].set_ylabel('$\mu$ (mag/arcsec$^2$)')
        ax[1].set_ylabel('$\Delta \mu$ (mag/arcsec$^2$)')

        ax[1].xaxis.set_minor_locator(ticker.FixedLocator(np.arange(0,20,0.5)))
        ax[1].xaxis.set_major_locator(ticker.FixedLocator(np.arange(0,20,2)))
        ax[1].xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

        ax[0].yaxis.set_minor_locator(ticker.FixedLocator(np.arange(-10.0,30.,0.2)))
        ax[0].yaxis.set_major_locator(ticker.FixedLocator(np.arange(-10.0,30.,2.0)))
        ax[0].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

        ax[1].yaxis.set_major_locator(ticker.FixedLocator(np.arange(-10.0,30.,0.5)))
        ax[1].yaxis.set_minor_locator(ticker.FixedLocator(np.arange(-10.0,30.,0.05)))
        ax[1].yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

        #plt.show()
        ax[0].tick_params(which='both',direction='in')
        ax[1].tick_params(which='both',direction='in')
        #ax[2].tick_params(which='both',direction='in')
        plt.tight_layout()
        fig.subplots_adjust(hspace=0)
        ax[0].legend(loc='upper right', fontsize=28)
        fig.savefig(plots_dir+obj_name+'_'+filter_name+'_sersic_profile.png',dpi=150)

        res[res<0]=0

        #data[y0-100:y0+100,x0-100:x0+100] = 0
        min_ = np.nanmedian(model)-2*np.nanstd(model)
        max_ = np.nanmedian(model)+8*np.nanstd(model)

        res = sigma_clip(res,sigma=3,maxiters=1)
        min_2 = np.nanmedian(res)-0.5*np.nanstd(res)
        max_2 = np.nanmedian(res)+1*np.nanstd(res)

        s = r_cut
        #norm1 = LogStretch()
        thumbnail_size = int(60/PIXEL_SCALES[fn]/2)
        x_best_int = int(x_best+0.5)
        y_best_int = int(y_best+0.5)
        data_zoom = data[x_best_int-thumbnail_size:x_best_int+thumbnail_size,\
                    y_best_int-thumbnail_size:y_best_int+thumbnail_size]

        model_zoom = model[x_best_int-thumbnail_size:x_best_int+thumbnail_size,\
                    y_best_int-thumbnail_size:y_best_int+thumbnail_size]

        res_zoom = res[x_best_int-thumbnail_size:x_best_int+thumbnail_size,\
                    y_best_int-thumbnail_size:y_best_int+thumbnail_size]

        mask_zoom = mask[x_best_int-thumbnail_size:x_best_int+thumbnail_size,\
                    y_best_int-thumbnail_size:y_best_int+thumbnail_size]


        ax1.imshow((data_zoom),cmap='gist_gray',vmin=min_, vmax=max_)
        ax2.imshow((mask_zoom),cmap='gist_gray',vmin=0, vmax=1)
        ax3.imshow((model_zoom),cmap='gist_gray',vmin=min_, vmax=max_)
        ax4.imshow((res_zoom),cmap='gist_gray',vmin=min_2, vmax=max_2)

        ax5.imshow(data_zoom,cmap='gist_gray',vmin=min_, vmax=max_)

        ax1.set_title('Main Frame',color='black',fontsize=20)
        ax2.set_title('Mask',color='black',fontsize=20)
        ax3.set_title('Model',color='black',fontsize=20)
        ax4.set_title('Residuals',color='black',fontsize=20)

        #ax1.plot(x_best,y_best,'r+',markersize=20)
        #ax2.plot(x_best,y_best,'r+',markersize=20)
        #ax3.plot(x_best,y_best,'r+',markersize=20)
        #ax4.plot(x_best,y_best,'r+',markersize=20)

        s = 60/PIXEL_SCALES[fn]
        ax1.text(s/20,s-s/12,obj_name,color='red',fontsize=20)
        ax1.arrow(s/20,s/25,2/scale/pix_size,0,head_width=0, head_length=20, color='gold',lw=2)
        ax1.text(s/20,s/15,'2 kpc',fontsize=32, color='gold')
        #ax1.arrow(r_cut_fit-50,50,0,100,color='gold',head_width=10, head_length=10,lw=2)
        #ax1.text(r_cut_fit-200+90,150,'N',fontsize=14, color='gold')
        #ax1.arrow(r_cut_fit-50,50,-100,0,color='gold',head_width=10, head_length=10,lw=2)
        #ax1.text(r_cut_fit-100-75,75,'E',fontsize=14, color='gold')

        ax5.text(s/20,s-s/12,obj_name,color='red',fontsize=20)
        ax5.arrow(s/20,s/25,2/scale/pix_size,0,head_width=0, head_length=20, color='gold',lw=4)
        ax5.text(s/20,s/15,'2 kpc',fontsize=32, color='gold')
        #ax5.arrow(r_cut_fit-50,50,0,100,color='gold',head_width=10, head_length=20,lw=4)
        #ax5.text(r_cut_fit-200+110,150,'N',fontsize=32, color='gold')
        #ax5.arrow(r_cut_fit-50,50,-100,0,color='gold',head_width=10, head_length=20,lw=4)
        #ax5.text(r_cut_fit-100-75,75,'E',fontsize=32, color='gold')

        ax1.axis('off')
        ax1.invert_yaxis()
        ax2.axis('off')
        ax2.invert_yaxis()
        ax3.axis('off')
        ax3.invert_yaxis()
        ax4.axis('off')
        ax4.invert_yaxis()

        ax5.axis('off')
        ax5.invert_yaxis()

        fig1.savefig(plots_dir+obj_name+'_'+filter_name+'_sersic_model.png',bbox_inches='tight', pad_inches = 0, dpi=100)
        fig2.savefig(plots_dir+obj_name+'_'+filter_name+'_fancy_frame.png',bbox_inches='tight', pad_inches = 0, dpi=100)
        plt.close()

        os.system('mv fit*log '+fit_dir)

    ra_best, dec_best = w.all_pix2world(x_best, y_best,0)
    return ra_best, dec_best, re_best_kpc, mag_best, n_best, pa_best_corr, axis_ratio_best#,Ie_best,d_re_best,d_n_best,d_pa_best,d_axis_ratio_best,d_Ie_best
