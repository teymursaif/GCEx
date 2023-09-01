import os, sys
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from astropy.io import fits
from astropy.wcs import WCS
import photutils
from photutils.aperture import CircularAperture
from photutils.aperture import CircularAnnulus
from modules.pipeline_functions import *
from modules.initialize import *
import random

############################################################

def estimate_aper_corr(gal_id):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    for fn in filters:
        psf_file = psf_dir+'psf_'+fn+'.fits'
        if os.path.exists(psf_file):
            aper_size_arcsec = (PSF_REF_RAD_ARCSEC[fn])
            #aper_size_pixel = aper_size_arcsec/PIXEL_SCALES[fn]
            psf_fits_file = fits.open(psf_file)
            psf_data = psf_fits_file[0].data
            psf_pixel_scale = psf_fits_file[0].header['PIXELSCL']
            X = float(psf_fits_file[0].header['NAXIS1'])
            Y = float(psf_fits_file[0].header['NAXIS2'])
            aper_size_pixel = aper_size_arcsec/psf_pixel_scale

            total_flux = np.nansum(psf_data)

            aper = CircularAperture((X/2., Y/2.), aper_size_pixel)
            aper_area = aper.area_overlap(data=psf_data,method='exact')
            flux, flux_err = aper.do_photometry(data=psf_data,method='exact')

            flux_ratio = float(flux[0]) / total_flux
            PSF_REF_RAD_FRAC[fn] = flux_ratio
            print ('- Estimated flux correction value for filter: ', flux_ratio, fn)
        else:
            print ("* PSF file is not found. using the default flux correction for aperture photometry in filter:", PSF_REF_RAD_FRAC[fn], fn)

############################################################

def estimate_fwhm(gal_id):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    for fn in filters:
        psf_file = psf_dir+'psf_'+fn+'.fits'
        print ('- estimating psf FWHM for filter:', fn)
        if os.path.exists(psf_file):
            psf_fits_file = fits.open(psf_file)
            psf_data = psf_fits_file[0].data
            psf_pixel_scale = psf_fits_file[0].header['PIXELSCL']
            X = float(psf_fits_file[0].header['NAXIS1'])
            Y = float(psf_fits_file[0].header['NAXIS2'])
            (FWHM_x, FWHM_y) = getFWHM_GaussianFitScaledAmp(psf_data)
            #print (FWHM_x,FWHM_y)
            #print (FWHM_x*psf_pixel_scale ,FWHM_y*psf_pixel_scale)
            FWHMS_ARCSEC[fn] = np.mean([FWHM_x*psf_pixel_scale ,FWHM_y*psf_pixel_scale])
            print ('- FWHM in filter', fn, 'is', FWHMS_ARCSEC[fn], 'arcsec')

        else:
            print ("* PSF file is not found. using the default value of FWHM (arcsec)", (FWHMS_ARCSEC[fn]))

############################################################

def twoD_GaussianScaledAmp(xy, xo, yo, sigma_x, sigma_y, amplitude, offset):
    """Function to fit, returns 2D gaussian function as 1D array"""
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    g = offset + amplitude*np.exp( - (((x-xo)**2)/(2*sigma_x**2) + ((y-yo)**2)/(2*sigma_y**2)))
    return g.ravel()

def getFWHM_GaussianFitScaledAmp(img):
    """Get FWHM(x,y) of a blob by 2D gaussian fitting
    Parameter:
        img - image as numpy array
    Returns:
        FWHMs in pixels, along x and y axes.
    """
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    x, y = np.meshgrid(x, y)
    #Parameters: xpos, ypos, sigmaX, sigmaY, amp, baseline
    initial_guess = (img.shape[1]/2,img.shape[0]/2,10,10,1,0)
    # subtract background and rescale image into [0,1], with floor clipping
    bg = np.percentile(img,5)
    img_scaled = np.clip((img - bg) / (img.max() - bg),0,1)
    popt, pcov = opt.curve_fit(twoD_GaussianScaledAmp, (x, y),
                               img_scaled.ravel(), p0=initial_guess,
                               bounds = ((img.shape[1]*0.4, img.shape[0]*0.4, 1, 1, 0.5, -0.1),
                                     (img.shape[1]*0.6, img.shape[0]*0.6, img.shape[1]/2, img.shape[0]/2, 1.5, 0.5)))
    xcenter, ycenter, sigmaX, sigmaY, amp, offset = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    FWHM_x = np.abs(4*sigmaX*np.sqrt(-0.5*np.log(0.5)))
    FWHM_y = np.abs(4*sigmaY*np.sqrt(-0.5*np.log(0.5)))
    return (FWHM_x, FWHM_y)

############################################################

def simulate_GCS_all_filters(gal_id):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    for fn in filters:
        psf_file = psf_dir+'psf_'+fn+'.fits'
        print ('- Making artificial GCs for data in filter:', fn)
        if os.path.exists(psf_file):
            simualte_GCs(gal_id,psf,psf_file,fn)
        else:
            print ("* PSF file is not found. GC simulations can not be done.")


def simualte_GCs(gal_id,psf_file,fn):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    # randomly select magnitudes and sizes of GCs
    for n in range(N_SIM_GCs):
        print ('+ simulating GCs: frame ',n,'out of',N_SIM_GCs)
        GCS_MAG = np.arange(GC_SIZE_RANGE[0],GC_SIZE_RANGE[1]+0.01,0.01)
        GCS_MAG = random.sample(GCS_MAG, N_ART_GCS)
        GCS_SIZE = np.arange(GC_MAG_RANGE[0],GC_MAG_RANGE[1]+0.01,0.01)
        GCS_SIZE = random.sample(GCS_SIZE, N_ART_GCS)

        # Randomly select X and Y in data that overalps with the science pixels
        science_frame = data_dir+gal_name+'_'+fn+'_cropped.fits'
        weight_frame = data_dir+gal_name+'_'+fn+'_cropped.weight.fits'
        weight = fits.open(weight_frame)
        weight_data = weight[0].data
        weight_header = weight[0].header
        X = weight_header['NAXIS1']
        Y = weight_header['NAXIS2']
        GCS_X = []
        GCS_Y = []
        for x in range(X):
            for y in range(Y):
                if weight_data[x,y] > 0:
                    GCS_X.append(x)
                    GCS_Y.append(y)

        GCS_X = random.sample(GCS_X,N_ART_GCS)
        DX = random.sample(np.arange(0,1,1./RATIO_OVERSAMPLE_PSF),N_ART_GCS)
        #GCS_X = GCS_X + DX
        GCS_Y = random.sample(GCS_Y,N_ART_GCS)
        DY = random.sample(np.arange(0,1,1./RATIO_OVERSAMPLE_PSF),N_ART_GCS)
        #GCS_Y = GC_Y + DY

        shutil.copy(science_frame,'temp.fits')
        temp = fits.open('temp.fits')
        temp[0].data = 0
        temp.writeto('temp.fits',overwrite=True)

        for i in range(N_ART_GCS):
            size = GCS_SIZE[i]
            mag = GCS_MAG[i]
            x = GCS_X[i]
            y = GCS_Y[i]
            print (x, y, dx, dy, mag, size)
            gc_file = art_dir+gal_name+'_'+fn+'_ART_GC_'+str(i)+'.fits'
            simulate_GC(x,y,mag,size,psf_file,gc_file)
            add_GC_to_frame(gc_file,art_dir+'temp.fits',output=art_dir+'temp.fits')

        add_fits_files(science_frame,art_dir+'temp.fits',art_dir+gal_name+'_'+fn+'_ART_+'str(n)'.fits'))

############################################################

def simulate_GC(x,y,dx,dy,mag,size,psf_file,gc_file):
    return 0

############################################################

def add_GC_to_frame(gc_file, science_file, output):
    return 0

############################################################

estimate_aper_corr(gal_id)
estimate_fwhm(gal_id)
