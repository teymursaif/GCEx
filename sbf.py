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
import pyfits
from astropy.table import Table, join_skycoord
from astropy import table
from fitsio import FITS
import scipy.stats as stats

from modules.pipeline_functions import *

def normalize_res_to_sqrt_model(sbf_res, sbf_model, sbf_mask, sbf_normal):
    res = fits.open(sbf_res)
    model = fits.open(sbf_model)
    mask = fits.open(sbf_mask)
    normal = fits.open(sbf_res) ##

    res_data = res[0].data
    model_data = model[0].data
    model_data = gaussian_filter(model_data,sigma=100)
    mask_data = 1-mask[0].data
    sqrt_model_data = np.sqrt(model_data)

    normal_data = res_data#/model_data
    mask[0].data = normal_data*(1-mask_data)
    normal_data = normal_data*mask_data
    normal[0].data = normal_data
    model[0].data = model_data

    normal.writeto(sbf_normal,overwrite=True)
    model.writeto(sbf_model,overwrite=True)
    mask.writeto(sbf_mask,overwrite=True)

def perform_fft(fitsfile,label=''):
    frame = fits.open(fitsfile)
    frame_data = frame[0].data
    img = np.array(frame_data)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 1.*np.log(np.abs(fshift))
    scale = LogStretch()
    plt.subplot(121),plt.imshow(scale(img), cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.savefig(sbf_dir+'sbf_fft2d'+label+'.png')
    plt.close()

    npix = img.shape[0]
    img[img==0]=0
    image = img
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2
    kfreq = np.fft.fftfreq(npix) * npix

    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    plt.plot(kvals[10:], Abins[10:])
    plt.yscale( 'log' )
    plt.xlabel("$k$")
    plt.ylabel("$P(k)$")
    plt.xlim([0,100])
    #plt.tight_layout()
    plt.savefig(sbf_dir+'sbf_power_spectrum'+label+'.png')
    plt.close()

    return (kvals[10:],Abins[10:])


def distance_sbf(gal_id,filter_sbf=None):
    sbf_distance = -99
    gal_name, ra, dec, distance, filters = gal_params[gal_id]
    print ('- Estimating the SBF distance of the galaxy '+gal_name+' in filter '+str(filter_sbf))
    if filter_sbf == None:
        print ("Filter for SBF is not defined. Skipping measuring SBF distance (the pipeline will use the initial input distance) ...")

    fn = filter_sbf
    check_image_noback = sex_dir+gal_name+'_'+fn+'_check_image_-background.fits'
    check_image_back = sex_dir+gal_name+'_'+fn+'_check_image_background.fits'
    image_mask = sex_dir+gal_name+'_'+'mask'+'_cropped.fits'

    #sbf_psf = sbf_dir+gal_name+'_'+fn+'_sbf_psf.fits'
    sbf_mask = sbf_dir+gal_name+'_'+fn+'_sbf_mask.fits'
    sbf_model = sbf_dir+gal_name+'_'+fn+'_sbf_model.fits'
    sbf_res = sbf_dir+gal_name+'_'+fn+'_sbf_res.fits'
    sbf_normal = sbf_dir+gal_name+'_'+fn+'_sbf_normal.fits'

    frame_size = GAL_FRAME_SIZE
    crop_frame(check_image_back,gal_name,frame_size,fn,ra,dec,output=sbf_model)
    crop_frame(check_image_noback,gal_name,frame_size,fn,ra,dec,output=sbf_res)
    crop_frame(image_mask,gal_name,frame_size,fn,ra,dec,output=sbf_mask)

    normalize_res_to_sqrt_model(sbf_res, sbf_model, sbf_mask, sbf_normal)
    sbf_normal_fft = perform_fft(sbf_normal,label='_normal')
    sbf_diffuse_fft = perform_fft(sbf_model,label='_diffuse')

    f = sbf_normal_fft[0]
    plt.plot(f,sbf_normal_fft[1],'k')
    plt.plot(f,4*10e2*sbf_diffuse_fft[1]+0.8*10e4,'r')

    plt.savefig(sbf_dir+'sbf_power_spectrum.png')
