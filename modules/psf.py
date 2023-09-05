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
from modules.initialize import *
from modules.pipeline_functions import *
import random
from scipy import signal

############################################################

def estimate_aper_corr(gal_id):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    for fn in filters:
        psf_file = psf_dir+'psf_'+fn+'.fits'
        if os.path.exists(psf_file):
            aper_size_arcsec = (APERTURE_SIZE[fn])
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
            APERTURE_SIZE[fn] = 2*FWHMS_ARCSEC[fn]
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

############################################################

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

def simulate_GCs_all(gal_id):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    for n in range(N_SIM_GCs):
        simualte_GCs(gal_id,n)

############################################################

def simualte_GCs(gal_id,n):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    print ('- simulating GCs: frame ',n+1,'out of',N_SIM_GCs)
    fn_det = filters[0]
    science_frame = data_dir+gal_name+'_'+fn_det+'_cropped.fits'
    weight_frame = data_dir+gal_name+'_'+fn_det+'_cropped.weight.fits'
    weight = fits.open(weight_frame)
    weight_data = weight[0].data
    weight_header = weight[0].header
    X = weight_header['NAXIS1']
    Y = weight_header['NAXIS2']
    GC_RA = []
    GC_DEC = []
    m = 0

    X_list = np.arange(1,X,1)
    Y_list = np.arange(1,Y,1)
    X_random = X_list.copy()
    Y_random = Y_list.copy()
    np.random.shuffle(X_random)
    np.random.shuffle(Y_random)

    m = 0
    #print (X_random)
    for i in range(10*N_ART_GCS):
        x = X_random[i]
        y = Y_random[i]
        if weight_data[x,y] > 0:
            w=WCS(science_frame)
            dx = np.arange(-0.5,0.51,0.1)
            dx = random.sample(list(dx),1)
            dy = np.arange(-0.5,0.51,0.1)
            dy = random.sample(list(dy),1)
            ra, dec = w.all_pix2world(x+dx, y+dy,0)
            GC_RA.append(ra[0])
            GC_DEC.append(dec[0])
            m = m+1
            if m == N_ART_GCS:
                break

    plt.plot(GC_RA,GC_DEC,'r.')
    plt.savefig(check_plots_dir+gal_name+'_ART_GCs_RA_DEC_'+str(n)+'.png')

    for fn in filters:
        psf_file = psf_dir+'psf_'+fn+'.fits'
        print ('- Making artificial GCs for data in filter:', fn)
        if os.path.exists(psf_file):
            donothing = 1
        else:
            print ("* PSF file is not found. GC simulations can not be done.")
            continue

        zp = ZPS[fn]

        GC_MAG = np.arange(GC_MAG_RANGE[0],GC_MAG_RANGE[1]+0.01,1./N_ART_GCS)
        GC_MAG = random.sample(list(GC_MAG), N_ART_GCS)
        GC_SIZE = np.arange(GC_SIZE_RANGE[0],GC_SIZE_RANGE[1]+0.01,1./N_ART_GCS)
        GC_SIZE = random.sample(list(GC_SIZE), N_ART_GCS)

        GC_RA = random.sample(GC_RA,N_ART_GCS)
        GC_DEC = random.sample(GC_DEC,N_ART_GCS)

        psf_fits_file = fits.open(psf_file)
        psf_pixel_scale = psf_fits_file[0].header['PIXELSCL']
        X_psf = float(psf_fits_file[0].header['NAXIS1'])
        Y_psf = float(psf_fits_file[0].header['NAXIS2'])

        RATIO_OVERSAMPLE_PSF = int(PIXEL_SCALES[fn]/psf_pixel_scale+0.5)

        shutil.copy(science_frame,art_dir+'temp.fits')
        temp = fits.open(art_dir+'temp.fits')
        X_oversampled = temp[0].header['NAXIS1']*RATIO_OVERSAMPLE_PSF
        Y_oversampled = temp[0].header['NAXIS2']*RATIO_OVERSAMPLE_PSF
        temp[0].data = np.zeros((X_oversampled,Y_oversampled))
        temp[0].header['NAXIS1'] = X_oversampled
        temp[0].header['NAXIS2'] = Y_oversampled
        temp.writeto(art_dir+'temp.fits',overwrite=True)

        img1 = fits.open(art_dir+'temp.fits')
        data1 = img1[0].data

        for i in range(N_ART_GCS):
            print ('- making psf:',str(i+1),'out of',N_ART_GCS)
            size = GC_SIZE[i]
            size_arcsec = GC_SIZE[i] / distance  * 0.206265
            abs_mag = GC_MAG[i]
            mag = GC_MAG[i] + 5*np.log10(distance*1e+5)

            ra = GC_RA[i]
            dec = GC_DEC[i]
            x, y = w.all_world2pix(ra, dec, 0)
            xos = x * RATIO_OVERSAMPLE_PSF
            yos = y * RATIO_OVERSAMPLE_PSF

            print (mag, abs_mag, size, size_arcsec)
            print (xos, yos, x, y)

            gc_file = art_dir+gal_name+'_'+fn+'_ART_GC_'+str(i)+'.fits'
            simulate_GC(mag,size_arcsec,zp,psf_pixel_scale,psf_file,gc_file)

            # add GCs to frame

            swarp_cmd = swarp_executable+' '+gc_file+' -c '+external_dir+'default.swarp -IMAGEOUT_NAME '+gc_file+'.resampled.fits'+\
                ' -IMAGE_SIZE 0 -PIXELSCALE_TYPE MANUAL -PIXEL_SCALE 1'+\
                ' -RESAMPLE Y -CENTER_TYPE MANUAL -CENTER '+str(x)+','+str(y)+' -SUBTRACT_BACK N'
            os.system(swarp_cmd)

            x0 = int(xos)
            y0 = int(yos)
            img2 = fits.open(gc_file+'.resampled.fits')
            data2 = img2[0].data
            x_psf = img2[0].header['NAXIS1']
            y_psf = img2[0].header['NAXIS2']
            dx = int(x_psf)
            dy = int(y_psf)
            data1[x0-dx:x0-dx+x_psf,y0-dy:y0-dy+y_psf] = data1[x0-dx:x0-dx+x_psf,y0-dy:y0-dy+y_psf]+data2

        img1[0].data = data1
        img1.writeto(art_dir+gal_name+'_'+fn+'_ART_'+str(n)+'_resampled.fits',overwrite=True)


        # undersample the simulated frame
        swarp_cmd = swarp_executable+' '+art_dir+gal_name+'_'+fn+'_ART_'+str(n)+'.fits'+\
            ' -c '+external_dir+'default.swarp -IMAGEOUT_NAME '+art_dir+gal_name+'_'+fn+'_ART_'+str(n)+'.fits'+\
            ' -IMAGE_SIZE 0 -PIXELSCALE_TYPE MANUAL -PIXEL_SCALE '+str(RATIO_OVERSAMPLE_PSF)+\
            ' -RESAMPLE Y -CENTER_TYPE MANUAL -CENTER '+str(x)+','+str(y)+' -SUBTRACT_BACK N'
        os.system(swarp_cmd)

        add_fits_files(science_frame,art_dir+gal_name+'_'+fn+'_ART_'+str(n)+'.fits',art_dir+gal_name+'_'+fn+'_ART_'+str(n)+'.science.fits')

############################################################

def simulate_GC(mag,size_arcsec,zp,pixel_size,psf_file,gc_file):
    #print (mag,size_arcsec,zp,pixel_size)
    stamp = makeKing2D(1, size_arcsec, mag, zp, 1, pixel_size)
    n = np.arange(100.0)
    hdu = fits.PrimaryHDU(n)
    hdul = fits.HDUList([hdu])
    #hdul[0].header = header
    hdul[0].data = stamp
    hdul.writeto(gc_file+'.temp.fits', overwrite=True)

    # convolving with psf
    psf_fits_file = fits.open(psf_file)
    psf_data = psf_fits_file[0].data
    stamp = signal.convolve2d(stamp, psf_data, boundary='symm', mode='same')
    #stamp = numpy.random.poisson(stamp)
    n = np.arange(100.0)
    hdu = fits.PrimaryHDU(n)
    hdul = fits.HDUList([hdu])
    #hdul[0].header = header
    hdul[0].data = stamp
    hdul.writeto(gc_file, overwrite=True)
    #return 0


############################################################

def makeKing2D(cc, rc, mag, zeropoint, exptime, pixel_size):
    '''
    A simple funtion to model a KING globular cluster. It is based on the inputs by Ariane Lancon:
    See Redmine issue

    https://euclid.roe.ac.uk/issues/16801?issue_count=9&issue_position=8&next_issue_id=16182&prev_issue_id=16802

    A.nucita

    :param cc: truncation parameter
    :param rc: core radius in arcseconds
    :param mag: integrated magnitude
    :param zeropoint: zeropoint magnitude to convert into
    :param exptime: seconds
    :param pixel_size in arcsec

    :return:
    '''
    # Calculate truncation radius in arcsec
    trunc_radius = rc * (10 ** (cc))  # arcsec
    # print(trunc_radius)

    # get stamp size: we require that the galaxy is in the exact center of the matrix. Therefore we set even size always.

    # Size: double of truncation radius + 2 pixel as a border
    Size = int(2 * round(trunc_radius / float(pixel_size))) + 2
    # make even
    if (Size % 2) != 0:
        Size = Size + 1

    stamp = np.zeros((Size, Size))

    # print(Size)
    # Exact center of the image
    # xc = (Size / 2.0 +0.5)*pixel_size
    # yc = (Size / 2.0 +0.5)*pixel_size

    imin = 0
    imax = Size  # -1  # Primo Cambio -1 commentato
    xc = ((imax + imin) / 2.0) * pixel_size
    yc = ((imax + imin) / 2.0) * pixel_size
    # Fill the matrix by requiring that the center of the galaxy is in xc, yc.
    # i --->  x  in f(x,y)
    # j ----> y  in f(x,y)
    # stamp[j, i] = flux

    # Total Flux in ADU
    totalflux = exptime * 10 ** (-(mag - zeropoint) / 2.5)

    # get normalization
    x = 10 ** (-cc)
    psi1 = np.log((1 + x ** 2) / (x ** 2))
    psi2 = -4 * (np.sqrt(1 + x ** 2) - x) / (np.sqrt(1 + x ** 2))
    psi3 = 1 / (1 + x ** 2)
    psi = np.pi * (psi1 + psi2 + psi3)
    A = (totalflux / psi) * pixel_size ** 2

    for i in range(0, Size):
        for j in range(0, Size):
            # Secondo Cambio aggiunto 0.5
            xi = (i) * pixel_size + 0.5 * pixel_size
            yi = (j) * pixel_size + 0.5 * pixel_size
            '''
            if ((i == 124) and (j == 124)):
                print(i, j, xi, yi, xc, yc)
                # stop()
            '''
            r2 = (xi - xc) ** 2 + (yi - yc) ** 2
            if r2 < trunc_radius ** 2:
                f1 = 1 / np.sqrt(r2 + rc ** 2)
                f2 = 1 / np.sqrt(trunc_radius ** 2 + rc ** 2)
                flux = A * (f1 - f2) ** 2
            else:
                flux = 0.0
            # print(i,j,flux)
            stamp[j, i] = flux
    '''
    hdu = fits.PrimaryHDU(data=stamp)
    file_name = os.path.join('output', 'king.fits')
    hdu.writeto(file_name, overwrite=True)
    '''
    return stamp

############################################################

estimate_fwhm(gal_id)
estimate_aper_corr(gal_id)
