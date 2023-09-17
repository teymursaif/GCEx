import os, sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy
from astropy.io import fits
from astropy.wcs import WCS
import photutils
from photutils.aperture import CircularAperture
from photutils.aperture import CircularAnnulus
from modules.initialize import *
from modules.pipeline_functions import *
from modules.source_det import *
import random
from scipy import signal
from scipy.ndimage import gaussian_filter

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

def estimate_aper_corr(gal_id):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    for fn in filters:
        psf_file = psf_dir+'psf_'+fn+'.fits'
        if os.path.exists(psf_file):
            aper_size_arcsec = (APERTURE_SIZE[fn])
            #aper_size_pixel = aper_size_arcsec/PIXEL_SCALES[fn]
            psf_fits_file = fits.open(psf_file)
            psf_data = psf_fits_file[0].data
            try :
                psf_pixel_scale = psf_fits_file[0].header[PSF_PIXELSCL_KEY]
            except :
                psf_pixel_scale = PSF_PIXEL_SCALE
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
            try :
                psf_pixel_scale = psf_fits_file[0].header[PSF_PIXELSCL_KEY]
            except :
                psf_pixel_scale = PSF_PIXEL_SCALE
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
    print (f"{bcolors.OKCYAN}- making artificial globular clusters"+ bcolors.ENDC)
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    for n in range(N_SIM_GCS):
        simualte_GCs(gal_id,n)

############################################################

def simualte_GCs(gal_id,n):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    ra_gal = ra
    dec_gal = dec
    fn_det = filters[0]
    print ('- simulating GCs: frame ',n+1,'out of',N_SIM_GCS)
    science_frame = data_dir+gal_name+'_'+fn_det+'_cropped.fits'
    w=WCS(science_frame)
    y_gal, x_gal= w.all_world2pix(ra_gal,dec_gal,0)
    weight_frame = data_dir+gal_name+'_'+fn_det+'_cropped.weight.fits'
    weight = fits.open(weight_frame)
    weight_data = weight[0].data
    weight_header = weight[0].header
    X = weight_header['NAXIS1']
    Y = weight_header['NAXIS2']
    x_gal = X-x_gal
    y_gal = Y-y_gal
    art_coords_cat_name = art_dir+gal_name+'_'+fn_det+'_ART'+str(n)+'_'+str(N_ART_GCS)+'GCs_coords.csv'

    if os.path.exists(art_coords_cat_name):
        print ('+ Coordinates of the simulated GCs are:')
        donothing = 1
        df = pd.read_csv(art_coords_cat_name, sep=',')
        coords = df
        print (df)
        #print (coords.values)
        GC_X = list(coords['X_GC' ])
        GC_Y = list(coords['Y_GC' ])
        GC_RA = list(coords['RA_GC' ])
        GC_DEC = list(coords['DEC_GC' ])
        GC_ABS_MAG = list(coords['GC_ABS_MAG'])
        GC_MAG = list(coords['GC_MAG'])
        GC_SIZE_ARCSEC = list(coords['GC_SIZE_ARCSEC'])
        GC_SIZE_PC = list(coords['GC_SIZE_PC'])

    else :
    #if True:
        coords = []
        GC_X = []
        GC_Y = []
        GC_RA = []
        GC_DEC = []
        GC_ABS_MAG = []
        GC_MAG = []
        GC_SIZE_ARCSEC = []
        GC_SIZE_PC = []

        MAG = np.arange(GC_MAG_RANGE[0],GC_MAG_RANGE[1]+0.01,abs(GC_MAG_RANGE[1]-GC_MAG_RANGE[0])/N_ART_GCS)
        MAG = random.sample(list(MAG), N_ART_GCS)
        SIZE = np.arange(GC_SIZE_RANGE[0],GC_SIZE_RANGE[1]+0.01,abs(GC_SIZE_RANGE[1]-GC_SIZE_RANGE[0])/N_ART_GCS)
        #print (SIZE)
        SIZE = random.sample(list(SIZE), N_ART_GCS)

        #X_list = np.arange(1,X,1)
        #Y_list = np.arange(1,Y,1)
        #X_random = X_list.copy()
        #Y_random = Y_list.copy()
        #np.random.shuffle(X_random)
        #np.random.shuffle(Y_random)
        X1_random = x_gal + np.random.normal(0,X/5,2000*N_ART_GCS)
        Y1_random = y_gal + np.random.normal(0,Y/5,2000*N_ART_GCS)
        X2_random = x_gal + np.random.normal(0,X/10,2000*N_ART_GCS)
        Y2_random = y_gal + np.random.normal(0,Y/10,2000*N_ART_GCS)
        X3_random = x_gal + np.random.normal(0,X/20,500*N_ART_GCS)
        Y3_random = y_gal + np.random.normal(0,Y/20,500*N_ART_GCS)
        X1_random = np.append(X1_random,X2_random)
        X1_random = np.append(X1_random,X3_random)
        Y1_random = np.append(Y1_random,Y2_random)
        Y1_random = np.append(Y1_random,Y3_random)
        #print (len(X1_random))
        X_random = X1_random.copy()
        Y_random = Y1_random.copy()
        np.random.shuffle(X_random)
        np.random.shuffle(Y_random)

        m = 0
        i = -1
        #print (X_random)
        while m < N_ART_GCS :
        #for i in range(N_ART_GCS):
            i = i+1
            x = int(X_random[i])
            y = int(Y_random[i])

            if (x<1) or (x>(X-1)) or (y<1) or (y>(Y-1)):
                continue

            if weight_data[x,y] > 0:
                dx = np.arange(-0.5,0.51,0.1)
                dx = random.sample(list(dx),1)
                dy = np.arange(-0.5,0.51,0.1)
                dy = random.sample(list(dy),1)
                dx = dx[0]
                dy = dy[0]
                ra, dec = w.all_pix2world(y+dy,x+dx,0)
                GC_X.append(x+dx)
                GC_Y.append(y+dy)
                GC_RA.append(ra)
                GC_DEC.append(dec)
                GC_ABS_MAG.append(MAG[m])
                mag = MAG[m] + 5*np.log10(distance*1e+5)
                GC_MAG.append(MAG[m] + 5*np.log10(distance*1e+5))
                GC_SIZE_PC.append(SIZE[m])
                size_arcsec = SIZE[m] / distance  * 0.206265
                GC_SIZE_ARCSEC.append(size_arcsec)
                coords.append([x+dx,y+dy,ra,dec,MAG[m],mag,SIZE[m],size_arcsec])
                #print (x+dx,y+dy)
                m = m+1
                if m == N_ART_GCS:
                    break

        coords = np.array(coords)
        #print (coords)
        print ('+ Coordinates of the simulated GCs are:')
        df = pd.DataFrame(coords, columns=['X_GC' ,'Y_GC' ,'RA_GC' ,'DEC_GC' ,'GC_ABS_MAG','GC_MAG','GC_SIZE_PC','GC_SIZE_ARCSEC'])
        df.to_csv(art_coords_cat_name, header=True, sep=',', index=False)
        print (df)

    plt.plot(GC_RA,GC_DEC,'r.')
    plt.savefig(check_plots_dir+gal_name+'_ART_GCs_RA_DEC_'+str(n)+'.png')
    plt.close()

    plt.hist(GC_X)
    plt.savefig(check_plots_dir+gal_name+'_ART_GCs_X_HIST_'+str(n)+'.png')
    plt.close()

    plt.hist(GC_Y)
    plt.savefig(check_plots_dir+gal_name+'_ART_GCs_Y_HIST_'+str(n)+'.png')
    plt.close()

    df = pd.read_csv(art_coords_cat_name)
    art_cat_name = art_dir+gal_name+'_'+fn_det+'_ART'+str(n)+'_'+str(N_ART_GCS)+'GCs.full_info.csv'

    for fn in filters:

        science_frame = data_dir+gal_name+'_'+fn+'_cropped.fits'
        w=WCS(science_frame)

        weight_frame = data_dir+gal_name+'_'+fn+'_cropped.weight.fits'
        weight = fits.open(weight_frame)
        weight_data = weight[0].data
        mask = weight_data
        mask[mask>0]=1

        psf_file = psf_dir+'psf_'+fn+'.fits'
        print ('- Making artificial GCs for data in filter:', fn)
        if os.path.exists(psf_file):
            donothing = 1
        else:
            print ("* PSF file is not found. GC simulations can not be done.")
            continue

        zp = ZPS[fn]
        exptime = EXPTIME[fn]
        ART_GC_FLAG = []
        GC_ABS_MAG_FILTER = []
        GC_MAG_FILTER = []

        psf_fits_file = fits.open(psf_file)

        try :
            psf_pixel_scale = psf_fits_file[0].header[PSF_PIXELSCL_KEY]
        except :
            psf_pixel_scale = PSF_PIXEL_SCALE

        global X_psf, Y_psf
        X_psf = float(psf_fits_file[0].header['NAXIS1'])
        Y_psf = float(psf_fits_file[0].header['NAXIS2'])

        global RATIO_OVERSAMPLE_PSF 
        RATIO_OVERSAMPLE_PSF = int(PIXEL_SCALES[fn]/psf_pixel_scale+0.5)
        #print (RATIO_OVERSAMPLE_PSF)

        shutil.copy(science_frame,art_dir+'temp.fits')
        temp = fits.open(art_dir+'temp.fits')
        X_oversampled = temp[0].header['NAXIS1']*1#RATIO_OVERSAMPLE_PSF
        Y_oversampled = temp[0].header['NAXIS2']*1#RATIO_OVERSAMPLE_PSF
        temp[0].data = np.zeros((X_oversampled,Y_oversampled))
        temp[0].header['NAXIS1'] = X_oversampled
        temp[0].header['NAXIS2'] = Y_oversampled
        temp.writeto(art_dir+'temp.fits',overwrite=True)

        img1 = fits.open(art_dir+'temp.fits')
        data1 = img1[0].data

        for i in range(N_ART_GCS):

            text = "+ Making " + str(N_ART_GCS) + " artificial GCs"+\
                " in progress: " + str(int((i+1)*100/N_ART_GCS)) + "%"
            print ("\r" + text + " ", end='')

            size_arcsec = GC_SIZE_ARCSEC[i]
            mag = GC_MAG[i] + color(fn,fn_det)
            GC_ABS_MAG_FILTER.append(GC_MAG[i]-5*np.log10(distance*1e+5))
            GC_MAG_FILTER.append(mag)
            #print (len(GC_RA), i)
            ra = GC_RA[i]
            dec = GC_DEC[i]
            y, x = w.all_world2pix(ra, dec, 0)
            x0 = int(x+0.5)
            y0 = int(y+0.5)
            #xos = x * RATIO_OVERSAMPLE_PSF
            #yos = y * RATIO_OVERSAMPLE_PSF
            print (' -> GC properties:',' mag=',mag,'size=',size_arcsec,'RA=',ra,'Dec=',dec,'X=',x,'Y=',y)
            #rint (x0,y0)
            gc_file = art_dir+gal_name+'_'+fn+'_ART_'+str(n)+'_GC_'+str(i)+'.fits'
            simulate_GC(mag,size_arcsec,zp,psf_pixel_scale,exptime,psf_file,gc_file)

            # add GCs to frame
            gc_fits_file = fits.open(gc_file)
            xc = (gc_fits_file[0].header['NAXIS1']+0.5)/2
            yc = (gc_fits_file[0].header['NAXIS2']+0.5)/2
            xos = xc #+ (x-x0)
            yos = yc #+ (y-y0)

            swarp_cmd = swarp_executable+' '+gc_file+' -c '+external_dir+'default.swarp -IMAGEOUT_NAME '+gc_file+'.resampled.fits'+\
                ' -IMAGE_SIZE 0 -PIXELSCALE_TYPE MANUAL -PIXEL_SCALE '+str(RATIO_OVERSAMPLE_PSF)+\
                ' -RESAMPLE Y -CENTER_TYPE ALL -SUBTRACT_BACK N -VERBOSE_TYPE QUIET'
            #print (swarp_cmd)
            os.system(swarp_cmd)

            #swarp_cmd = swarp_executable+' '+gc_file+'.resampled.fits'+' -c '+external_dir+'default.swarp -IMAGEOUT_NAME '+gc_file+'.resampled_displaced.fits'+\
            #    ' -IMAGE_SIZE 0 -PIXELSCALE_TYPE MEDIAN -PIXEL_SCALE '+str(0)+\
            #    ' -RESAMPLE Y -CENTER_TYPE MANUAL -CENTER '+str(x-x0)+','+str(y-y0)+' -SUBTRACT_BACK N -VERBOSE_TYPE QUIET'
            #os.system(swarp_cmd)

            img2 = fits.open(gc_file+'.resampled.fits')
            data2 = img2[0].data
            x_psf = img2[0].header['NAXIS1']
            y_psf = img2[0].header['NAXIS2']
            dx = int(x_psf/2)
            dy = int(y_psf/2)
            x1 = x0-dx
            x2 = x0-dx+x_psf
            y1 = y0-dy
            y2 = y0-dy+y_psf

            if x1 < 0 or y1 < 0 or x2 > X or y2 > Y:
                print ('--- GC outside of the frame, skipping simulating the GC ...')
                ART_GC_FLAG.append(-1)
                continue
            else:
                ART_GC_FLAG.append(1)

            #print (X,Y)
            #print (x1,x2,y1,y2)
            #print (x1,x2,y1,y2)
            x1_psf = 0
            x2_psf = x_psf
            y1_psf = 0
            y2_psf = y_psf
            data1[x1:x2,y1:y2] = data1[x1:x2,y1:y2]+data2[x1_psf:x2_psf,y1_psf:y2_psf]

        df['GC_ABS_MAG_'+fn] = GC_ABS_MAG_FILTER
        df['GC_MAG_'+fn] = GC_MAG_FILTER
        df['ART_GC_FLAG_'+fn] = ART_GC_FLAG

        img1[0].data = data1*mask
        img1.writeto(art_dir+gal_name+'_'+fn+'_ART_'+str(n)+'.fits',overwrite=True)

        add_fits_files(science_frame,art_dir+gal_name+'_'+fn+'_ART_'+str(n)+'.fits',art_dir+gal_name+'_'+fn+'_ART_'+str(n)+'.science.fits')
        print ('')

    df.to_csv(art_cat_name, index=False)

############################################################

def simulate_GC(mag,size_arcsec,zp,pix_size,exptime,psf_file,gc_file):
    # Conclusion from Ariane: For c in [1.3,1.5], the half-light radius is about  2.5Rc( ±10%) and
    # Rc itself contains about 20% of the light (i.e 40% of the light within the half-light radius).
    #print (mag,size_arcsec,zp,pixel_size)
    #rh/rc ~ 2.5
    stamp = makeKing2D(1.4, size_arcsec/2.5, mag, zp, 1, pix_size)
    n = np.arange(100.0)
    hdu = fits.PrimaryHDU(n)
    hdul = fits.HDUList([hdu])
    #hdul[0].header = header
    hdul[0].data = stamp
    hdul.writeto(gc_file+'.king.fits', overwrite=True)
    #print ('king', np.sum(stamp))

    # convolving with psf
    psf_fits_file = fits.open(psf_file)
    psf_data = psf_fits_file[0].data
    #print (np.sum(psf_data))
    psf_data = psf_data/np.sum(psf_data)
    ####psf_data = gaussian_filter(psf_data,sigma=3.5)
    stamp = signal.convolve2d(stamp, psf_data, boundary='symm', mode='valid')
    #stamp = convolve2D(stamp, psf_data)
    #print ('king+psf', np.sum(stamp))
    hdul[0].data = stamp
    hdul.writeto(gc_file+'.conv.fits', overwrite=True)

    stamp_noisy = np.random.normal(stamp*exptime,1*np.sqrt(stamp*exptime)/RATIO_OVERSAMPLE_PSF)#/RATIO_OVERSAMPLE_PSF
    #print (stamp_noisy[20,20],stamp[20,20]*exptime,stamp_noisy[20,20]-stamp[20,20]*exptime)
    stamp_noisy = stamp_noisy/exptime
    hdul[0].data = stamp_noisy
    hdul.writeto(gc_file+'.noise.fits', overwrite=True)
    #print (stamp-stamp_noisy)
    stamp = stamp_noisy
    #print ('king+psf+noise', np.sum(stamp))

    n = np.arange(100.0)
    hdu = fits.PrimaryHDU(n)
    hdul = fits.HDUList([hdu])
    #hdul[0].header = header
    hdul[0].data = stamp
    hdul.writeto(gc_file, overwrite=True)
    #return 0
    if len(stamp[stamp>0]) == 0:
        print (f"{bcolors.WARNING}*** Warning: the output frame of the simulated GC looks blank."+ bcolors.ENDC)

############################################################

def makeKing2D(cc, rc, mag, zeropoint, exptime, pixel_size):
    '''
    A simple funtion to model a KING globular cluster. It is based on the inputs by Ariane Lancon:
    See Redmine issue

    https://euclid.roe.ac.uk/issues/16801?issue_count=9&issue_position=8&next_issue_id=16182&prev_issue_id=16802

    A.nucita
    (modified by Teymoor Saifollahi, September 2023)

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

    # Size: 2 times of truncation radius + 50 pixel as a border
    Size = int(int(1 * round(trunc_radius / float(pixel_size))) + 2)
    
    #if Size < X_psf:
    #    Size = int(X_psf)
    # make even
    if (Size % 2) != 0:
        Size = Size + 1

    stamp = np.zeros((Size, Size))

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

    final_flux_ratio = (np.sum(stamp)/totalflux)

    if final_flux_ratio < 0.50:
        print (f"{bcolors.FAIL}*** Serious Warning: the simulated GCs are missing a fraction of the light larger than 50%. Something is wrong!"+ bcolors.ENDC)
    elif final_flux_ratio < 0.95:
        print (f"{bcolors.WARNING}*** Warning: the simulated GCs are missing a fraction of the light larger than 5%."+ bcolors.ENDC)
    elif final_flux_ratio < 0.98:
        print (f"{bcolors.WARNING}*** Warning: the simulated GCs are missing a fraction of the light between 2% to 5%."+ bcolors.ENDC)
    elif final_flux_ratio < 0.99:
        print (f"{bcolors.WARNING}*** Warning: the simulated GCs are missing a fraction of the light between 1% to 2%."+ bcolors.ENDC)
    
    #print (final_flux_ratio)
        
    return stamp
    
############################################################

def make_psf_all_filters(gal_id):
    print (f"{bcolors.OKCYAN}- Making PSF models for all the filters"+ bcolors.ENDC)
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    
    for fn in filters:
        print ('- Making PSF for filter', fn)

        zp = ZPS[fn]
        gain = GAIN[fn]
        pix_size = PIXEL_SCALES[fn]

        main_frame = data_dir+gal_name+'_'+fn+'_gal_cropped.fits'
        weight_frame = data_dir+gal_name+'_'+fn+'_gal_cropped.weight.fits'
        source_cat = sex_dir+gal_name+'_'+fn+'_source_cat_for_psf_model.fits'
        psf_frame = psfs_dir+'psf_'+fn+'.fits'

        # run SE
        command = SE_executable+' '+main_frame+' -c '+external_dir+'default.sex -CATALOG_NAME '+source_cat+' '+ \
        '-PARAMETERS_NAME '+external_dir+'default.param -DETECT_MINAREA 8 -DETECT_THRESH 3.0 -ANALYSIS_THRESH 3.0 ' + \
        '-DEBLEND_NTHRESH 1 -DEBLEND_MINCONT 1 -MAG_ZEROPOINT ' +str(zp) + ' -BACKPHOTO_TYPE GLOBAL '+\
        '-FILTER Y -FILTER_NAME  '+external_dir+'tophat_1.5_3x3.conv -STARNNW_NAME '+external_dir+'default.nnw -PIXEL_SCALE ' + \
        str(pix_size)+ ' -BACK_SIZE 128 -BACK_FILTERSIZE 3'
  
        os.system(command)

        ###

        make_psf_for_frame(main_frame,weight_frame,source_cat,fn,psf_frame)

############################################################

def make_psf_for_frame(main_frame,weight_frame,source_cat,filtername,psf_frame):

    table_main = fits.open(source_cat)
    table_data = table_main[1].data
    sex_cat_data = table_data
    fn = filtername
    zp = ZPS[fn]
    gain = GAIN[fn]
    pix_size = PIXEL_SCALES[fn]

    mask = ((sex_cat_data['FLAGS'] < 4) & \
    (sex_cat_data ['ELLIPTICITY'] < 0.1) & \
    (sex_cat_data ['MAG_AUTO'] > 18) & \
    (sex_cat_data ['MAG_AUTO'] < 22) & \
    (sex_cat_data ['FWHM_IMAGE'] > 0.5) & \
    (sex_cat_data ['FWHM_IMAGE'] < 10))

    sex_cat_data = sex_cat_data[mask]
    fwhm = sex_cat_data['FWHM_IMAGE']
    fwhm = sigma_clip(fwhm,sigma=2)

    mask = ((sex_cat_data ['FWHM_IMAGE'] >= np.nanmin(fwhm)) & \
    (sex_cat_data ['FWHM_IMAGE'] <= np.nanmax(fwhm)))
    #print (np.min(fwhm),np.max(fwhm))

    sex_cat_data = sex_cat_data[mask]
    
    N = len(sex_cat_data)

    RA = sex_cat_data['ALPHA_SKY']
    DEC = sex_cat_data['DELTA_SKY']
    X = sex_cat_data['X_IMAGE']
    Y = sex_cat_data['Y_IMAGE']
    fwhm = sex_cat_data['FWHM_IMAGE']

    psf_frames = list()
    #os.system('rm ' + psfs_dir +'*'+'psf*')
    print ('- Number of selected stars: '+str(N))
    psfs = list()
    for i in range(N) :

        ra = RA[i]
        dec = DEC[i]
        star_fits_file = psfs_dir+gal_name+'_'+fn+'_star_'+str(i)+'.fits'
        crop_frame(main_frame, '', int(PSF_IMAGE_SIZE/2.), filtername, ra, dec, \
            output=star_fits_file)

        psf = fits.open(star_fits_file)
        image_size =  psf[0].header['NAXIS1']
        psf_data = psf[0].data
        psf_data_back = np.nanmedian(sigma_clip(psf_data,sigma=3))
        psf[0].data = psf_data - psf_data_back
        psf.writeto(star_fits_file, overwrite=True)
        
        psf_data = scipy.ndimage.zoom(psf_data, RATIO_OVERSAMPLE_PSF, order=3)
        psf_data = np.array(psf_data)
        
        x_center = int(image_size*RATIO_OVERSAMPLE_PSF/2.+0.5)
        y_center = int(image_size*RATIO_OVERSAMPLE_PSF/2.+0.5)
        x_max, y_max = np.unravel_index(psf_data.argmax(), psf_data.shape)
        #print (x_max,y_max)
        dx = int(x_max-x_center+(RATIO_OVERSAMPLE_PSF/2.))
        dy = int(y_max-y_center+(RATIO_OVERSAMPLE_PSF/2.))
        #print (dx,dy)
        psf_data = np.roll(psf_data, -1*dx, axis=0)
        psf_data = np.roll(psf_data, -1*dy, axis=1)

        psf[0].data = psf_data
        psf.writeto(star_fits_file+'.oversampled.fits', overwrite=True)
        #psf_data_min = np.sort(psf_data)[:int(len(psf_data)/2)]
        psf_data_back = np.nanmedian(sigma_clip(psf_data,sigma=3))
        #print (psf_data_back)
        psf_data = psf_data - psf_data_back
        psf_data_sum = np.nansum(psf_data)
        psf_data = psf_data / psf_data_sum
        if np.shape(psf_data) == (image_size*RATIO_OVERSAMPLE_PSF,image_size*RATIO_OVERSAMPLE_PSF) :
            psfs.append(psf_data)
            psf_frames.append(star_fits_file+'.oversampled.fits')
        #print (psf_data)
        #print (np.shape(psf_data))
        #print ('-------------')

    psf_median = np.median(psfs,axis=0)
    #print (psfs)
    psf_median[psf_median<0] = 0
    psf_median_sum = np.nansum(psf_median)
    psf_median = psf_median / psf_median_sum
    PSF = fits.open(star_fits_file+'.oversampled.fits')
    #print (psf_median)
    PSF[0].data = psf_median
    PSF.writeto(psf_frame+'.oversampled.fits', overwrite=True)
    #psf_median = rebin(psf_median, (int(image_size), int(image_size)))
    PSF[0].header['PIXELSCL'] = pix_size/RATIO_OVERSAMPLE_PSF
    PSF[0].data = psf_median
    PSF.writeto(psf_frame, overwrite=True)

    shutil.copy(psf_frame,psf_dir+'psf_'+fn+'.fits')

############################################################

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).sum(-1).sum(1)

############################################################

def initial_psf(gal_id):
    if MODEL_PSF == True:
        make_psf_all_filters(gal_id)
    estimate_fwhm(gal_id)
    estimate_aper_corr(gal_id)

############################################################
