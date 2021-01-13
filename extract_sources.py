import sys, os
import pyfits
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from astropy.wcs import WCS
from astropy import wcs
from astropy.io import fits
import os.path
import scipy
from scipy import signal
from scipy import ndimage
from pylab import *
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.image as mpimg
from matplotlib.ticker import ScalarFormatter
import astropy.io.fits
from matplotlib.ticker import StrMethodFormatter
import matplotlib.ticker as ticker
from functions import *
from astropy.stats import sigma_clip
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from scipy import ndimage, misc
import os.path
import pyfits

def make_detection_frame(frame, udg, filter, data_dir, sex_dir, detection_dir, ra, dec, pix_size, backsize=16, backfiltersize=1, iteration=3):
    os.system('cp '+frame+' temp0.fits')

    for i in range(0,iteration):
        print ('+ iteration '+str(i))
        #os.system('mv temp.check.fits temp0.fits')
        if i == 0 :
            weight_command = ''
        if i > 0 :
            weight_command = '-WEIGHT_TYPE  MAP_VAR -WEIGHT_IMAGE temp.weight'+str(i)+'.fits -WEIGHT_THRESH 0.005 '

        command = 'sextractor '+frame+' -c '+str(sex_dir)+'default.sex -CATALOG_NAME '+'temp.sex_cat'+str(i)+'.fits '+ \
        '-PARAMETERS_NAME '+str(sex_dir)+'default.param -DETECT_MINAREA 4 -DETECT_THRESH 2 -ANALYSIS_THRESH 2 ' + weight_command + \
        '-FILTER_NAME  '+str(sex_dir)+'default.conv -STARNNW_NAME '+str(sex_dir)+'default.nnw -PIXEL_SCALE ' + str(pix_size) + ' ' \
        '-BACK_SIZE '+ str(backsize)+' -BACK_FILTERSIZE '+ str(backfiltersize)+' -CHECKIMAGE_TYPE BACKGROUND,-BACKGROUND,APERTURES ' +  \
        '-CHECKIMAGE_NAME temp.back-check'+str(i)+'.fits,temp.-back-check'+str(i)+'.fits,temp.aper-check'+str(i)+'.fits'
        #print (command)
        os.system(command)
        if i > 0 :
            attach_sex_tables(['temp.sex_cat'+str(i-1)+'.fits','temp.sex_cat'+str(i)+'.fits'],'temp.sex_cat'+str(i)+'.fits')
        make_mask(frame,'temp.sex_cat'+str(i)+'.fits',None,'temp.mask'+str(i+1)+'.fits','temp.mask+'+str(i+1)+'.fits','temp.weight'+str(i+1)+'.fits',\
            'temp.flag'+str(i)+'.fits', ra, dec)


    img1 = pyfits.open(frame)
    img2 = pyfits.open('temp.back-check'+str(iteration-1)+'.fits')
    data1 = img1[0].data
    data2 = img2[0].data

    data1 = data1-data2
    #print (data1)
    img1[0].data = data1
    img1.writeto(udg+'_'+filter+'.detection_frame.fits',clobber=True)
    
    os.system('rm temp*.fits')
    return udg+'_'+filter+'.detection_frame.fits'

def make_mask(frame, sex_source_cat, weight, mask_out, mask_out2, weight_out, flag_out, ra, dec, mode='None'):
    X = get_header(frame,keyword='NAXIS1')
    Y = get_header(frame,keyword='NAXIS2')
    img = pyfits.open(frame)
    data = img[0].data
    data2 = data
    if mode == 'cat' :
        cat = pyfits.open(sex_source_cat)
        table = cat[2].data
        N = len(table)
        for i in range (0,N) :
            params = table[i]
            ra_star = params[27]
            dec_star = params[28]
            x = params[25]
            y = params[26]
            fwhm = params[34]
            flag = params[33]
            A = params[29]
            B = params[30]
            r = math.sqrt((ra-ra_star)**2+(dec-dec_star)**2)
            if flag <= 9999 and r >= 0.1/3600.:
                if x >= X or y >= Y :
                    continue
                x = int(x)
                y = int(y)
                #fwhm = int(fwhm+0.5)
                mask_size = int(10*B)
                #print (i, ra, dec, x, y, fwhm, flag)
                for i in range(0,mask_size+1) :
                    for j in range(0,mask_size+1) :

                        rr = math.sqrt((i)**2+(j)**2)
                        if rr <= mask_size :
                            if x+i < X and y+j < Y :
                                data[y+j][x+i] = 1
                                #data2[y+j][x+i] = min_value
                            if x+i < X and y-j > 0 :
                                data[y-j][x+i] = 1
                                #data2[y-j][x+i] = min_value
                            if x-i > 0 and y+j < Y :
                                data[y+j][x-i] = 1
                                #data2[y+j][x-i] = min_value
                            if x-i > 0 and y-j > 0 :
                                data[y-j][x-i] = 1
                                #data2[y-j][x-i] = min_value
    if mode == 'weight' or mode == 'cat' :
        wimg = pyfits.open(weight)
        weight_data = wimg[0].data
        data[weight_data<2500] = 1
        data[weight_data>2500-1] = 0.001
                    
    where_are_NaNs = isnan(data)
    data[where_are_NaNs] = 1
    data[abs(data)<0.000000001] = 1
    data = data.astype(int)
    img[0].data = (data)
    #print (data)
    img.writeto(mask_out,clobber=True)

    data_flag = data*64
    data_flag = data_flag.astype(np.int16)
    img[0].data = data_flag
    img.writeto(flag_out,clobber=True)

    w = 1 - data
    data2 = data2*w
    #data2 = np.log10(data2)
    #data2 = 10**(data2)
    img[0].data = (data2)
    img.writeto(mask_out2,clobber=True)

    w = data + 0.001#(data[:,:]*99)+1
    #data3 = data3*w
    img[0].data = (w)
    img.writeto(weight_out,clobber=True)

def make_sex_cats(frames, udg, filters, weights, detection_frames, sex_dir, detection_dir, ra, dec, pix_size, zp, aper_corr_value, maglimit) :
    for i in range(len(frames)):
        frame = frames[i]
        filter = filters[i]
        detection_frame = detection_frames[i]
        weight = weights[i]

        make_mask(frame, None, weight, 'temp.mask'+str(i)+'.fits','temp.mask+'+str(i)+'.fits','temp.weight'+str(i)+'.fits', \
        'temp.flag'+str(i)+'.fits', ra, dec, mode='weight')
        weight_command = '-WEIGHT_TYPE  MAP_VAR -WEIGHT_IMAGE temp.weight'+str(i)+'.fits -WEIGHT_THRESH 0.005 '+\
        '-FLAG_IMAGE '+'temp.flag'+str(i)+'.fits -FLAG_TYPE MAX'

        command = 'sextractor '+detection_frame+','+frame+' -c '+str(sex_dir)+'default.sex -CATALOG_NAME '+'temp.sex_cat'+str(i)+'.fits '+ \
        '-PARAMETERS_NAME '+str(sex_dir)+'default+.param -DETECT_MINAREA 5 -DETECT_THRESH 0.25 -ANALYSIS_THRESH 0.25 ' + \
        '-DEBLEND_NTHRESH 4 -DEBLEND_MINCONT 0.1 ' + weight_command + ' -PHOT_APERTURES 2,4,6,8,10,15,20,25,30,40 ' + \
        '-MAG_ZEROPOINT ' +str(zp) + ' -BACKPHOTO_TYPE LOCAL -BACKPHOTO_THICK 20 '+\
        '-FILTER_NAME  '+str(sex_dir)+'tophat_2.0_3x3.conv -STARNNW_NAME '+str(sex_dir)+'default.nnw -PIXEL_SCALE ' + str(pix_size) + ' ' \
        '-BACK_SIZE 32 -BACK_FILTERSIZE 1 -CHECKIMAGE_TYPE APERTURES,FILTERED,-BACKGROUND ' +  \
        '-CHECKIMAGE_NAME temp.aper-check'+str(i)+'.fits,temp.filtered-check'+str(i)+'.fits,temp.-back-check'+str(i)+'.fits'

        os.system(command)
        prepare_cat('temp.sex_cat'+str(i)+'.fits',udg,filter,udg+'_'+filter+'_'+'catalogue.fits',aper_corr_value, maglimit) 
        #os.system('rm temp*.fits')


def prepare_cat(cat_name,udg,filter,out_name, aper_corr_value, maglimit) :
    os.system('rm temp.fits')
    os.system('rm temp+.fits')
    os.system('rm '+out_name)

    with pyfits.open(cat_name) as hdul:
        fwhm_upper = 20
        fwhm_lower = 0
        data = hdul[2].data
        mask = ( (data['FLAGS'] < 4) & \
        #(data['FWHM_IMAGE'] > fwhm_lower) & \
        #(data['MAG_PETRO'] > 5) & \
        #(data['MAG_PETRO'] < 25) & \
        #(data['MAG_APER'] > 5) & \
        (data['ELLIPTICITY'] < 0.8) & \
        (data['MAG_AUTO'] > 20) & \
        (data['MAG_AUTO'] < 30) & \
        (data['FWHM_IMAGE'] < fwhm_upper) & \
        (data['FWHM_IMAGE'] > fwhm_lower) )
        newdata = data[mask]

        mg = newdata['MAG_APER']
        mg = np.array(mg)
        emg = newdata['MAGERR_APER']
        emg = np.array(emg)

        c28 = mg[:,0]-mg[:,3]
        c48 = mg[:,1]-mg[:,3]
        c68 = mg[:,2]-mg[:,3]
        c810 = mg[:,3]-mg[:,4]
        c1015 = mg[:,4]-mg[:,5]
        c1520 = mg[:,5]-mg[:,6]
        c2025 = mg[:,6]-mg[:,7]
        c2530 = mg[:,7]-mg[:,8]
        c3040 = mg[:,8]-mg[:,9]

        ec28 = np.sqrt(emg[:,0]**2+emg[:,3]**2)
        ec48 = np.sqrt(emg[:,1]**2+emg[:,3]**2)
        ec68 = np.sqrt(emg[:,2]**2+emg[:,3]**2)
        ec810 = np.sqrt(emg[:,3]**2+emg[:,4]**2)
        ec1015 = np.sqrt(emg[:,4]**2+emg[:,5]**2)
        ec1520 = np.sqrt(emg[:,5]**2+emg[:,6]**2)
        ec2025 = np.sqrt(emg[:,6]**2+emg[:,7]**2)
        ec2530 = np.sqrt(emg[:,7]**2+emg[:,8]**2)
        ec3040 = np.sqrt(emg[:,8]**2+emg[:,9]**2)

        m = mg[:,3] - aper_corr_value
        em = emg[:,3]

        hdu = pyfits.BinTableHDU(data=newdata)
        #hdu.writeto(filtername+field+'.sex_stars_cat.filtered.fits')
        hdu.writeto('temp.fits')

        expand_fits_table('temp.fits','c28'+'_'+str(filter),c28)
        expand_fits_table('temp.fits','c48'+'_'+str(filter),c48)
        expand_fits_table('temp.fits','c68'+'_'+str(filter),c68)
        expand_fits_table('temp.fits','c810'+'_'+str(filter),c810)
        expand_fits_table('temp.fits','c1015'+'_'+str(filter),c1015)
        expand_fits_table('temp.fits','c1520'+'_'+str(filter),c1520)
        expand_fits_table('temp.fits','c2025'+'_'+str(filter),c2025)
        expand_fits_table('temp.fits','c2530'+'_'+str(filter),c2530)
        expand_fits_table('temp.fits','c3040'+'_'+str(filter),c3040)

        expand_fits_table('temp.fits','ec28'+'_'+str(filter),ec28)
        expand_fits_table('temp.fits','ec48'+'_'+str(filter),ec48)
        expand_fits_table('temp.fits','ec68'+'_'+str(filter),ec68)
        expand_fits_table('temp.fits','ec810'+'_'+str(filter),ec810)
        expand_fits_table('temp.fits','ec1015'+'_'+str(filter),ec1015)
        expand_fits_table('temp.fits','ec1520'+'_'+str(filter),ec1520)
        expand_fits_table('temp.fits','ec2025'+'_'+str(filter),ec2025)
        expand_fits_table('temp.fits','ec2530'+'_'+str(filter),ec2530)
        expand_fits_table('temp.fits','ec3040'+'_'+str(filter),ec3040)

        expand_fits_table('temp.fits','mag'+'_'+str(filter),m)
        expand_fits_table('temp.fits','emag'+'_'+str(filter),em)

    ttypes = list()
    for i in range (0,57) :
        i = i+1
        ttypes.append('TTYPE'+str(i))

    cat = get_fits('temp.fits')
    for ttype in ttypes :
        #print (cat[1].header[ttype])
        if cat[1].header[ttype] == 'ALPHA_SKY' :
            cat[1].header[ttype] = 'RA'
        if cat[1].header[ttype] == 'DELTA_SKY' :
            cat[1].header[ttype] = 'DEC'
        cat[1].header[ttype] = cat[1].header[ttype]+'_'+str(filter)
        labels = cat[1].header

    cat.writeto('temp+.fits')

    ### other criteria
    with pyfits.open('temp+.fits') as hdul:
        data = hdul[1].data

        mask = ( (data['c48'+'_'+filter] < 5) & \
        (data['c28'+'_'+filter] > -5) & \
        (data['c48'+'_'+filter] > -5) & \
        (data['c68'+'_'+filter] > -5) & \
        (data['mag'+'_'+filter] < maglimit) & \
        (data['NIMAFLAGS_ISO'+'_'+filter] < 2) )
        #(data['FWHM_IMAGE'] < fwhm_upper) & \
        #(data['FWHM_IMAGE'] > fwhm_lower) )
        newdata = data[mask]
        hdu = pyfits.BinTableHDU(data=newdata)
        hdu.writeto(out_name)

    os.system('rm temp.fits')
    os.system('rm temp+.fits')

