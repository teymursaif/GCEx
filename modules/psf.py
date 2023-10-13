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
from modules.fit_galaxy import *
from modules.plots_functions import *
import random
from scipy import signal
from scipy.ndimage import gaussian_filter
from astropy.stats import sigma_clip

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

#plotting parameters
plt.style.use('tableau-colorblind10')
plt.rc('axes', labelsize=24)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('axes', linewidth=1.7)
plt.rc('font',**{'family':'serif','serif':['Times']})
plt.rcParams['hatch.linewidth'] = 2.0
plt.rc('text', usetex=True)

############################################################

def estimate_aper_corr(gal_id):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    data_name = gal_data_name[gal_id]

    for fn in filters:
        psf_file = psf_dir+data_name+'_psf_'+fn+'.fits'
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

            back = np.nanmedian(sigma_clip(psf_data,2, masked=False))
            #print (back)
            #total_flux = np.nansum(psf_data-back)
            psf_data = psf_data#-back

            aper = CircularAperture((X/2., Y/2.), aper_size_pixel)
            aper_area = aper.area_overlap(data=psf_data,method='exact')
            flux, flux_err = aper.do_photometry(data=psf_data,method='exact')

            aper_large = CircularAperture((X/2., Y/2.), X/2.)#*FWHMS_ARCSEC[fn]/psf_pixel_scale)
            aper_area_large = aper.area_overlap(data=psf_data,method='exact')
            total_flux, total_flux_err = aper_large.do_photometry(data=psf_data,method='exact')

            flux_ratio = float(flux[0]) / total_flux
            PSF_REF_RAD_FRAC[fn] = flux_ratio
            print ('- Estimated flux correction value for filter: ', flux_ratio, fn)
        else:
            print ("* PSF file is not found. using the default flux correction for aperture photometry in filter:", PSF_REF_RAD_FRAC[fn], fn)

############################################################

def estimate_fwhm_for_psf_fits(psf_fits_file_name, psf_pixel_scale=-1):

    psf_fits_file = fits.open(psf_fits_file_name)
    psf_data = psf_fits_file[0].data

    if psf_pixel_scale == -1:
        try :
            psf_pixel_scale = psf_fits_file[0].header[PSF_PIXELSCL_KEY]
        except :
            psf_pixel_scale = PSF_PIXEL_SCALE

    X = float(psf_fits_file[0].header['NAXIS1'])
    Y = float(psf_fits_file[0].header['NAXIS2'])
    (FWHM_x, FWHM_y) = getFWHM_GaussianFitScaledAmp(psf_data)
    #print (FWHM_x,FWHM_y)
    #print (FWHM_x*psf_pixel_scale ,FWHM_y*psf_pixel_scale)
    fwhm_pixel = np.mean([FWHM_x,FWHM_y])
    fwhm_arcsec = np.mean([FWHM_x*psf_pixel_scale ,FWHM_y*psf_pixel_scale])
    print ('- FWHM is', fwhm_pixel, 'pixel and', fwhm_arcsec, 'arcsec')
    return fwhm_pixel,fwhm_arcsec

############################################################

def estimate_fwhm(gal_id):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    data_name = gal_data_name[gal_id]

    for fn in filters:
        psf_file = psf_dir+data_name+'_psf_'+fn+'.fits'
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
            APERTURE_SIZE[fn] = 1.0*FWHMS_ARCSEC[fn]
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
    bg = 0 #np.percentile(img,5)
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
    data_name = gal_data_name[gal_id]
    for n in range(N_SIM_GCS):
        simualte_GCs(gal_id,n)

############################################################

def simualte_GCs(gal_id,n):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    data_name = gal_data_name[gal_id]
    methods = gal_methods[gal_id]

    ra_gal = ra
    dec_gal = dec
    fn_det = filters[0]
    print ('- simulating GCs: frame ',n+1,'out of',N_SIM_GCS)

    if 'USE_SUB_GAL' in methods:
        shutil.copy(fit_dir+gal_name+'_'+fn_det+'_galfit_imgblock_res.fits',sub_data_dir+gal_name+'_'+fn_det+'.fits')
        science_frame = sub_data_dir+gal_name+'_'+fn_det+'.fits'
    else :
        science_frame = data_dir+gal_name+'_'+fn_det+'_cropped.fits'

    #print ('GCSIM', science_frame)

    weight_frame = data_dir+gal_name+'_'+fn_det+'_cropped.weight.fits'
    w=WCS(science_frame)
    x_gal, y_gal= w.all_world2pix(ra_gal,dec_gal,0)
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

        if GC_SIM_MODE == 'UNIFORM' :
            X_list = np.arange(1,X,1)
            Y_list = np.arange(1,Y,1)
            X_random = X_list.copy()
            Y_random = Y_list.copy()
            np.random.shuffle(X_random)
            np.random.shuffle(Y_random)

        if GC_SIM_MODE == 'CONCENTRATED' :
            X1_random = x_gal + np.random.normal(0,X/2,2000*N_ART_GCS)
            Y1_random = y_gal + np.random.normal(0,Y/2,2000*N_ART_GCS)
            X2_random = x_gal + np.random.normal(0,X/10,1000*N_ART_GCS)
            Y2_random = y_gal + np.random.normal(0,Y/10,1000*N_ART_GCS)
            X3_random = x_gal + np.random.normal(0,X/20,1000*N_ART_GCS)
            Y3_random = y_gal + np.random.normal(0,Y/20,1000*N_ART_GCS)
            X1_random = np.append(X1_random,X2_random)
            X1_random = np.append(X1_random,X3_random)
            Y1_random = np.append(Y1_random,Y2_random)
            Y1_random = np.append(Y1_random,Y3_random)
            X_random = X1_random.copy()
            Y_random = Y1_random.copy()

        #print (len(X1_random))
        
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

            #print ()

            if weight_data[y,x] > 0:
                dx = np.arange(-0.5,0.51,0.1)
                dx = random.sample(list(dx),1)
                dy = np.arange(-0.5,0.51,0.1)
                dy = random.sample(list(dy),1)
                dx = dx[0]
                dy = dy[0]
                ra, dec = w.all_pix2world(x+dx,y+dy,0)
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

        if 'USE_SUB_GAL' in methods:
            shutil.copy(fit_dir+gal_name+'_'+fn+'_galfit_imgblock_res.fits',sub_data_dir+gal_name+'_'+fn+'.fits')
            science_frame = sub_data_dir+gal_name+'_'+fn+'.fits'
        else :
            science_frame = data_dir+gal_name+'_'+fn+'_cropped.fits'

        weight_frame = data_dir+gal_name+'_'+fn+'_cropped.weight.fits'
        weight = fits.open(weight_frame)
        weight_data = weight[0].data
        mask = weight_data
        mask[mask>0]=1
        wf=WCS(science_frame)

        #print ('GCSIM2',science_frame)

        psf_file = psf_dir+data_name+'_psf_'+fn+'.fits'
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
        temp[0].data = np.zeros((Y_oversampled,X_oversampled))
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
            x, y = wf.all_world2pix(ra, dec, 0)
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
                ' -RESAMPLING_TYPE LANCZOS4 -IMAGE_SIZE 0 -PIXELSCALE_TYPE MANUAL -PIXEL_SCALE '+str(RATIO_OVERSAMPLE_PSF)+\
                ' -RESAMPLE Y -CENTER_TYPE ALL -SUBTRACT_BACK N -VERBOSE_TYPE QUIET'
            #print (swarp_cmd)
            os.system(swarp_cmd)

            #swarp_cmd = swarp_executable+' '+gc_file+'.resampled.fits'+' -c '+external_dir+'default.swarp -IMAGEOUT_NAME '+gc_file+'.resampled_displaced.fits'+\
            #    ' -IMAGE_SIZE 0 -PIXELSCALE_TYPE MEDIAN -PIXEL_SCALE '+str(0)+\
            #    ' -RESAMPLE Y -CENTER_TYPE MANUAL -CENTER '+str(x-x0)+','+str(y-y0)+' -SUBTRACT_BACK N -VERBOSE_TYPE QUIET'
            #os.system(swarp_cmd)

            #print ("stamps FWHM:")
            #print ('king:')
            #estimate_fwhm_for_psf_fits(gc_file+'.king.fits',psf_pixel_scale)
            #print ('conv:')
            #estimate_fwhm_for_psf_fits(gc_file+'.conv.fits',psf_pixel_scale)
            #print ('noise:')
            #estimate_fwhm_for_psf_fits(gc_file+'.noise.fits',psf_pixel_scale)
            #print ('resampled:')
            #estimate_fwhm_for_psf_fits(gc_file+'.resampled.fits',PIXEL_SCALES[fn])

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

            X = X_oversampled
            Y = Y_oversampled

            if x1 < (dx+1) or y1 < (dy+1) or x2 > (X-dx-1) or y2 > (Y-dy-1) :#or (abs(x2-x1)<x_psf) or (abs(y2-y1)<y_psf):
                print (f"{bcolors.WARNING}--- GC outside of the frame, skipping simulating the GC ..."+ bcolors.ENDC)
                ART_GC_FLAG.append(-1)
                continue
            else:
                ART_GC_FLAG.append(1)
            
            #print (X,Y)
            print (x,y)
            print (x1,x2,y1,y2)
            #print (x_psf,y_psf)
            x1_psf = 0
            x2_psf = x_psf
            y1_psf = 0
            y2_psf = y_psf
            #data1[x1:x2,y1:y2] = data1[x1:x2,y1:y2]+data2[x1_psf:x2_psf,y1_psf:y2_psf]
            #print (np.shape(data1[y1:y2,x1:x2]))
            #print (np.shape(data2[y1_psf:y2_psf,x1_psf:x2_psf]))
            data1[y1:y2,x1:x2] = data1[y1:y2,x1:x2]+data2[y1_psf:y2_psf,x1_psf:x2_psf]

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
    n = np.arange(100.0)
    hdu = fits.PrimaryHDU(n)
    hdul = fits.HDUList([hdu])
    hdul[0].data = stamp
    hdul.writeto(gc_file+'.conv.fits', overwrite=True)

    stamp_noisy = np.random.normal(stamp*exptime,1*np.sqrt(stamp*exptime))#/RATIO_OVERSAMPLE_PSF
    #print (stamp_noisy[20,20],stamp[20,20]*exptime,stamp_noisy[20,20]-stamp[20,20]*exptime)
    stamp_noisy = stamp_noisy/exptime

    n = np.arange(100.0)
    hdu = fits.PrimaryHDU(n)
    hdul = fits.HDUList([hdu])
    hdul[0].data = stamp_noisy
    hdul.writeto(gc_file+'.noise.fits', overwrite=True)
    #print (stamp-stamp_noisy)
    #print ('king+psf+noise', np.sum(stamp))

    n = np.arange(100.0)
    hdu = fits.PrimaryHDU(n)
    hdul = fits.HDUList([hdu])
    #hdul[0].header = header
    hdul[0].data = stamp_noisy
    hdul.writeto(gc_file, overwrite=True)
    #return 0
    if len(stamp[stamp>0]) == 0:
        print (f"{bcolors.WARNING}*** Warning: the output frame of the simulated GC looks blank."+ bcolors.ENDC)

    """
    ### resampling to the nominal pixel-scale
    X = hdul[0].header['NAXIS1']
    Y = hdul[0].header['NAXIS2']
    #print (X, Y)
    resampled_image_size = int((np.shape(stamp_noisy))[0]/RATIO_OVERSAMPLE_PSF)
    res = X % resampled_image_size
    #print (res)
    dx1 = int(res/2)
    dx2 = res-dx1
    dy1 = int(res/2)
    dy2 = res-dy1
    x1 = dx1
    x2 = X-dx2
    y1 = dy1
    y2 = Y-dy2
    #print (x1, x2, y1, y2, x2-x1, y2-y1)
    #print (resampled_image_size)
    stamp_noisy_resampled = stamp_noisy[x1:x2,y1:y2]
    stamp_noisy_resampled = rebin(stamp_noisy_resampled, (int(resampled_image_size), int(resampled_image_size)))
    n = np.arange(100.0)
    hdu = fits.PrimaryHDU(n)
    hdul = fits.HDUList([hdu])
    hdul[0].data = stamp_noisy_resampled
    hdul.writeto(gc_file+'.resampled.fits', overwrite=True)
    """

############################################################

def makeKing2D(cc, rc, mag, zeropoint, exptime, pixel_size):
    '''
    A simple funtion to model a KING globular cluster. It is based on the inputs by Ariane Lancon:
    See Redmine issue

    https://euclid.roe.ac.uk/issues/16801?issue_count=9&issue_position=8&next_issue_id=16182&prev_issue_id=16802

    A.nucita
    Modified by Teymoor Saifollahi to correct for the missing light

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

    # Size: 1 time of truncation radius + 2 pixel as a border
    Size = (int(1 * round(trunc_radius / float(pixel_size))) + 2)
    #print (Size)

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

    n = 10 #this is a start...
    for i in range(0, Size):
        for j in range(0, Size):
            flux = 0
            n = 10
            #if (abs(i-Size/2) < 5) and (abs(j-Size/2) < 5):
            #    n = 50

            # Secondo Cambio aggiunto 0.5
            for ii in range(n):
                for jj in range(n):

                    xi = (i+ii/n) * pixel_size + 0.5 * pixel_size
                    yi = (j+jj/n) * pixel_size + 0.5 * pixel_size

                    r2 = (xi - xc) ** 2 + (yi - yc) ** 2
                    if r2 < trunc_radius ** 2:
                        f1 = 1 / np.sqrt(r2 + rc ** 2)
                        f2 = 1 / np.sqrt(trunc_radius ** 2 + rc ** 2)
                        f = A / (n ** 2) * (f1 - f2) ** 2
                    else:
                        f = 0.0
                    flux = flux + f
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
    data_name = gal_data_name[gal_id]

    for fn in filters:
        print ('- Making PSF for filter', fn)

        psf_file = psf_dir+data_name+'_psf_'+fn+'.fits'
        if os.path.exists(psf_file):
            print ('- A psf model is found, however the pipeline continues to make a new one.')
            #return 0

        zp = ZPS[fn]
        gain = GAIN[fn]
        pix_size = PIXEL_SCALES[fn]

        main_frame = data_dir+gal_name+'_'+fn+'_cropped.fits'
        weight_frame = data_dir+gal_name+'_'+fn+'_cropped.weight.fits'
        source_cat = sex_dir+gal_name+'_'+fn+'_source_cat_for_psf_model.fits'
        psf_frame = psfs_dir+data_name+'_psf_'+fn+'.fits'
        psf_frame_inst = psfs_dir+data_name+'_psf_'+fn+'.inst.fits'

        # run SE
        command = SE_executable+' '+main_frame+' -c '+external_dir+'default.sex -CATALOG_NAME '+source_cat+' '+ \
        '-PARAMETERS_NAME '+external_dir+'default_psf.param -DETECT_MINAREA 8 -DETECT_THRESH 5.0 -ANALYSIS_THRESH 5.0 ' + \
        '-DEBLEND_NTHRESH 1 -DEBLEND_MINCONT 1 -MAG_ZEROPOINT ' +str(zp) + ' -BACKPHOTO_TYPE GLOBAL '+\
        '-FILTER Y -FILTER_NAME  '+external_dir+'default.conv -STARNNW_NAME '+external_dir+'default.nnw -PIXEL_SCALE ' + \
        str(pix_size)+ ' -BACK_SIZE 128 -BACK_FILTERSIZE 3 -PHOT_APERTURES 10'

        shutil.copy(external_dir+'sex_default.param',external_dir+'default_psf.param')
        params = open(external_dir+'default_psf.param','a')
        params.write('MAG_APER('+str(1)+') #Fixed aperture magnitude vector [mag]\n')
        params.write('MAGERR_APER('+str(1)+') #RMS error vector for fixed aperture mag [mag]\n')
        params.write('FLUX_APER('+str(1)+') # Flux within a Kron-like elliptical aperture [count]\n')
        params.write('FLUXERR_APER('+str(1)+') #RMS error for AUTO flux [count]\n')
        params.close()

        #if os.path.exists(source_cat):
        #    donothing=1
        #else:
        os.system(command)
        ###


        make_psf_for_frame(main_frame,weight_frame,source_cat,fn,psf_frame,mode='auto')
        shutil.copy(psf_frame,psf_dir+data_name+'_psf_'+fn+'.fits')

        #psf_frame_soren = '/data/users/saifollahi/Euclid/ERO/ERO-data/PSF/psf_VIS_v3c_Soren.fits'
        #normalize_psf(psf_frame_soren)
        #psf_frames = [psf_frame, psf_frame_soren]
        make_radial_profile_for_psf([psf_frame],fn,0,zp,\
            output_png=plots_dir+data_name+'_'+fn+'_psf_radial_profiles.png')

        make_fancy_png(psf_frame,img_dir+data_name+'psf_'+fn+'.fits'+'.png',text='PSF for '+fn,zoom=2,cmap='seismic')

        ####
        #make_psf_for_frame(main_frame,weight_frame,source_cat,fn,psf_frame_inst,mode='auto',resample=False)
        #shutil.copy(psf_frame_inst,psf_dir+data_name+'_psf_'+fn+'.inst.fits')

        #make_radial_profile_for_psf([psf_frame_inst],0,zp,\
        #    output_png=check_plots_dir+data_name+'_'+fn+'_psf_inst_radial_profiles.png')

        os.system('rm '+psfs_dir+'*star*.fits')

############################################################

def normalize_psf(psf_frame):
    psf_fits_file = fits.open(psf_frame)
    psf_data = psf_fits_file[0].data
    sum_psf = np.sum(psf_data)
    psf_data = psf_data/sum_psf
    psf_fits_file[0].data = psf_data
    psf_fits_file.writeto(psf_frame, overwrite=True)

############################################################

def make_psf_for_frame(main_frame,weight_frame,source_cat,filtername,psf_frame,mode='auto',resample=True):

    table_main = fits.open(source_cat)
    table_data = table_main[1].data
    sex_cat_data = table_data
    fn = filtername
    zp = ZPS[fn]
    gain = GAIN[fn]
    pix_size = PIXEL_SCALES[fn]

    mag = sex_cat_data ['MAG_AUTO']
    fwhm = sex_cat_data['FWHM_IMAGE']
    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
    ax.plot(mag,fwhm,'k.',alpha=0.1,label='Detected sources')

    mask = ((sex_cat_data['FLAGS'] < 1) & \
    (sex_cat_data ['ELLIPTICITY'] < ELL_LIMIT_PSF) & \
    (sex_cat_data ['MAG_AUTO'] > MAG_LIMIT_SAT) & \
    (sex_cat_data ['MAG_AUTO'] < MAG_LIMIT_PSF) & \
    (sex_cat_data ['FWHM_IMAGE'] > 1) & \
    (sex_cat_data ['FWHM_IMAGE'] < 10))
    sex_cat_data = sex_cat_data[mask]

    fwhm = sex_cat_data['FWHM_IMAGE']

    if mode == 'auto':
        fwhm = sigma_clip(fwhm,sigma=2, maxiters=5, masked=False)
        fwhm_max = np.nanmax(fwhm)
        fwhm_min = np.nanmin(fwhm)

    print ('- the lower and upper limits for FWHM (for selecting stars) are:',fwhm_min,fwhm_max)
    mask = ((sex_cat_data ['FWHM_IMAGE'] >= fwhm_min) & \
    (sex_cat_data ['FWHM_IMAGE'] <= fwhm_max))
    #print (np.min(fwhm),np.max(fwhm))
    sex_cat_data = sex_cat_data[mask]

    mag = sex_cat_data ['MAG_AUTO']
    fwhm = sex_cat_data['FWHM_IMAGE']
    ax.plot(mag,fwhm,'ro',alpha=0.2,label='Selected for PSF modeling')

    ax.set_xlim([MAG_LIMIT_SAT-1,MAG_LIMIT_PSF+2])
    ax.set_ylim([0,6])
    ax.legend(loc='upper left',fontsize=20)
    ax.tick_params(which='both',direction='in')
    ax.set_xlabel('m$_{'+fn+'}$ \ [mag]')
    ax.set_ylabel('FWHM$_{'+fn+'}$ \ [pixel]')
    #plt.savefig(plots_dir+'psf_'+fn+'_selected_for_psf.png')
    output_plot = psf_frame+'.stars-selected-for-psf.png'
    plt.savefig(output_plot)
    plt.close()
    os.system('mv '+output_plot+' '+plots_dir)
    #return 0

    ###############

    N = len(sex_cat_data)
    #N_converged = 0
    N_used = 0

    RA = sex_cat_data['ALPHA_SKY']
    DEC = sex_cat_data['DELTA_SKY']
    X = sex_cat_data['X_IMAGE']
    Y = sex_cat_data['Y_IMAGE']
    FWHM = sex_cat_data['FWHM_IMAGE']
    MAG = sex_cat_data['MAG_APER']

    star_frames = ''
    star_weight_frames = ''
    print ('- Number of selected stars: '+str(N))

    psfs = list()
    #for i in range(N) :
    if N > 500 :
        N = 500
    for i in range(N):

        XC = list()
        YC = list()
        ra = RA[i]
        dec = DEC[i]
        x = X[i]
        y = Y[i]
        mag = MAG[i]
        fwhm = FWHM[i]

        star_fits_file = psfs_dir+gal_name+'_'+fn+'_star_'+str(i)+'.fits'
        star_weight_file = psfs_dir+gal_name+'_'+fn+'_star_'+str(i)+'.weight.fits'

        crop_frame(main_frame, '', int(PSF_IMAGE_SIZE), filtername, ra, dec, \
            output=star_fits_file)

        crop_frame(weight_frame, '', int(PSF_IMAGE_SIZE), filtername, ra, dec, \
            output=star_weight_file)

        output = psfs_dir+gal_name+'_'+fn+'_star_'+str(i)+'.cropped.fits'
        output_back_sub = psfs_dir+gal_name+'_'+fn+'_star_'+str(i)+'.cropped-back.fits'
        output_weight = psfs_dir+gal_name+'_'+fn+'_star_'+str(i)+'.cropped.weight.fits'
        output_back_sub_weight = psfs_dir+gal_name+'_'+fn+'_star_'+str(i)+'.cropped-back.weight.fits'

        N_iter = 1
        for iter in range(N_iter) :
            psf_frame_size_pix = int(PSF_IMAGE_SIZE)*2#RATIO_OVERSAMPLE_PSF
            psf_pixel_size = pix_size#RATIO_OVERSAMPLE_PSF
            if iter==0:
                command = swarp_executable+' '+star_fits_file+' -c '+external_dir+'default.swarp -IMAGEOUT_NAME '+output+\
                    ' -WEIGHTOUT_NAME '+output_weight+' -WEIGHT_TYPE MAP_WEIGHT '+'-WEIGHT_IMAGE '+star_weight_file+\
                    ' -IMAGE_SIZE '+str(psf_frame_size_pix)+','+str(psf_frame_size_pix)+' -PIXELSCALE_TYPE  MANUAL -CELESTIAL_TYPE EQUATORIAL'+\
                    ' -PIXEL_SCALE '+str(psf_pixel_size)+' -CENTER_TYPE MANUAL -CENTER '+str(ra)+','+str(dec)+\
                    ' -RESAMPLE N -RESAMPLING_TYPE LANCZOS4 -SUBTRACT_BACK Y -BACK_SIZE '+str(psf_frame_size_pix)+' -BACK_FILTERSIZE 1 -VERBOSE_TYPE NORMAL'

            else:
                command = swarp_executable+' '+star_fits_file+' -c '+external_dir+'default.swarp -IMAGEOUT_NAME '+output+\
                    ' -WEIGHTOUT_NAME '+output_weight+' -WEIGHT_TYPE MAP_WEIGHT '+'-WEIGHT_IMAGE '+star_weight_file+\
                    ' -IMAGE_SIZE '+str(psf_frame_size_pix)+','+str(psf_frame_size_pix)+' -PIXELSCALE_TYPE  MANUAL -CELESTIAL_TYPE EQUATORIAL'+\
                    ' -PIXEL_SCALE '+str(psf_pixel_size)+' -CENTER_TYPE MANUAL -CENTER '+str(ra_c)+','+str(dec_c)+\
                    ' -RESAMPLE N -RESAMPLING_TYPE LANCZOS4 -SUBTRACT_BACK N -BACK_SIZE '+str(psf_frame_size_pix)+' -BACK_FILTERSIZE 1 -VERBOSE_TYPE NORMAL'

            os.system(command)

            psf_data = fits.open(output)
            X_psf = psf_data[0].header['NAXIS1']
            Y_psf = psf_data[0].header['NAXIS2']
            xc = int(X_psf/2+0.5)
            yc = int(Y_psf/2+0.5)

            source_cat = temp_dir+'temp.fits'
            command = SE_executable+' '+output+' -c '+external_dir+'default.sex -CATALOG_NAME '+source_cat+' '+ \
                '-PARAMETERS_NAME '+external_dir+'sex_default.param -DETECT_MINAREA 8 -DETECT_THRESH 3.0 -ANALYSIS_THRESH 3.0 ' + \
                '-DEBLEND_NTHRESH 1 -DEBLEND_MINCONT 1 -MAG_ZEROPOINT ' +str(zp) + ' -BACKPHOTO_TYPE GLOBAL '+\
                '-FILTER Y -FILTER_NAME  '+external_dir+'default.conv -STARNNW_NAME '+external_dir+'default.nnw '+\
                '-BACK_SIZE ' + str(psf_frame_size_pix) + ' -BACK_FILTERSIZE 1 -PHOT_APERTURES 10 -CHECKIMAGE_TYPE -BACKGROUND '+\
                '-CHECKIMAGE_NAME '+output_back_sub
            os.system(command)

            #shutil.copy(temp_dir+'temp_psf_back_sub.fits',output)
            shutil.copy(output_weight,output_back_sub_weight)

            table_main = fits.open(source_cat)
            sex_cat_data = table_main[1].data
            RA0 = sex_cat_data['ALPHA_SKY']
            DEC0 = sex_cat_data['DELTA_SKY']
            X0 = sex_cat_data['X_IMAGE']
            Y0 = sex_cat_data['Y_IMAGE']
            FWHM0 = sex_cat_data['FWHM_IMAGE']
            M = len(RA0)
            det_flag = 0
            fwhm_flag = 0
            for j in range(M):
                ra0 = RA0[j]
                dec0 = DEC0[j]
                x0 = X0[j]
                y0 = Y0[j]
                fwhm0 = FWHM0[j]
                crossmatch_radius = psf_pixel_size/3600
                #print (crossmatch_radius)
                if (abs(ra0-ra)<crossmatch_radius) and (abs(ra0-ra)<crossmatch_radius):
                    #print ('*** FWHM for the not-resampled data:',fwhm)
                    #print ('*** FWHM for the resampled data:',fwhm0)
                    ra_c = ra0
                    dec_c = dec0
                    x_c = x0
                    y_c = y0
                    det_flag = 1

        psf_data = fits.open(output_back_sub)
        psf_data[0].header['PIXELSCL'] = psf_pixel_size
        psf_data[0].header['CRPIX1'] = x_c
        psf_data[0].header['CRVAL1'] = 0
        psf_data[0].header['CRPIX2'] = y_c
        psf_data[0].header['CRVAL2'] = 0
        norm = 2.512**(mag-MAG_LIMIT_PSF)
        psf_data[0].data = (psf_data[0].data*norm)
        psf_data.writeto(output_back_sub,overwrite=True)
        #normalize_psf(output_back_sub)

        #print (i, ra, dec, x, y, fwhm, x_c, y_c)


        fwhm_ = make_radial_profile_for_psf([output_back_sub],fn,pix_size,zp,output_png=None)

        if (fwhm_ > fwhm_min) and (fwhm_ < fwhm_max) and (abs(fwhm-fwhm_)<1e-1) :
            fwhm_flag = 1

        if i < 5 :
            fwhm_ = make_radial_profile_for_psf([output_back_sub],fn,pix_size,zp,\
                output_png=check_plots_dir+gal_name+'_'+fn+'_star_'+str(i)+'.cropped-back_radial_profile_'+str(fwhm_flag)+\
                '_'+str(ra0)[:9]+'_'+str(dec0)[:9]+'.png')

        #print ('*****', i, fwhm, fwhm_, fwhm_flag)

        if (det_flag == 1) and (fwhm_flag == 1):
            star_frames = star_frames+output_back_sub+','
            star_weight_frames = star_weight_frames+output_back_sub_weight+','
            N_used = N_used+1

    ### stacking all stars
    psf_weight_frame = psf_frame+'.weight.fits'
    if resample == True:
        psf_frame_size_pix = int(PSF_IMAGE_SIZE)*RATIO_OVERSAMPLE_PSF
        psf_pixel_size  = pix_size/RATIO_OVERSAMPLE_PSF

    elif resample == False:
        psf_frame_size_pix = int(PSF_IMAGE_SIZE)
        psf_pixel_size  = pix_size

    command = swarp_executable+' '+star_frames+' -c '+external_dir+'default.swarp -IMAGEOUT_NAME '+psf_frame+\
            ' -WEIGHTOUT_NAME '+psf_weight_frame+' -WEIGHT_TYPE MAP_WEIGHT -SUBTRACT_BACK Y -CELESTIAL_TYPE EQUATORIAL'+\
            ' -RESAMPLE Y -PIXELSCALE_TYPE  MANUAL -PIXEL_SCALE '+str(psf_pixel_size)+' -RESAMPLING_TYPE LANCZOS4 '+\
            ' -IMAGE_SIZE '+str(psf_frame_size_pix)+','+str(psf_frame_size_pix)+' -CENTER_TYPE MANUAL -CENTER '+str(0)+','+str(0)+\
            ' -BACK_SIZE '+str(psf_frame_size_pix)+' -BACK_FILTERSIZE 1'
            # -VERBOSE_TYPE QUIET'
            #' -PIXELSCALE_TYPE  MANUAL -PIXEL_SCALE '+str(1)+' -VERBOSE_TYPE QUIET'
            #' -IMAGE_SIZE '+str(radius_pix)+','+str(radius_pix)+' -PIXEL_SCALE '+str(pix_size/RATIO_OVERSAMPLE_PSF)+\
            #' -CENTER_TYPE MANUAL -CENTER '+str(ra)+','+str(dec)+' -SUBTRACT_BACK N -VERBOSE_TYPE QUIET'

    #print (command)
    os.system(command)

    psf_data = fits.open(psf_frame)
    psf_data[0].header['PIXELSCL'] = psf_pixel_size
    psf_data[0].header['SAMPLING'] = 1/RATIO_OVERSAMPLE_PSF
    X_psf = psf_data[0].header['NAXIS1']
    Y_psf = psf_data[0].header['NAXIS2']
    psf_data[0].header['CRPIX1'] = int(X_psf/2+0.5)
    psf_data[0].header['CRVAL1'] = 0
    psf_data[0].header['CRPIX2'] = int(Y_psf/2+0.5)
    psf_data[0].header['CRVAL2'] = 0

    back = np.nanmedian(sigma_clip(psf_data[0].data,2, masked=False))
    psf_data[0].data = psf_data[0].data - back
    #print (back)

    psf_data.writeto(psf_frame,overwrite=True)
    normalize_psf(psf_frame)
    print ('- Number of used stars for PSF modeling is: '+str(N_used))


############################################################

def make_radial_profile_for_psf(psf_frames,fn,pixelsize,zp,output_png):

    if output_png != None:
        make_plots = 1
        fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
    elif output_png == None:
        make_plots = 0

    colors = ['black','red','cyan','green','blue']
    i = -1
    for psf_frame in psf_frames:
        d_flux_apers = []
        radi_arcsec = []
        i = i + 1

        psf_data = fits.open(psf_frame)
        X_psf = psf_data[0].header['NAXIS1']
        Y_psf = psf_data[0].header['NAXIS2']

        if pixelsize == 0 :
            try :
                #psf_fits_file = fits.open(psf_frame)
                psf_pixel_size = psf_data[0].header[PSF_PIXELSCL_KEY]
            except :
                psf_pixel_size = PSF_PIXEL_SCALE

        elif pixelsize > 0 :
            psf_pixel_size = pixelsize

        radial_profile_apers_array = np.arange(0.05/psf_pixel_size,X_psf/2,0.05/psf_pixel_size)
        radial_profile_apers_values = radial_profile_apers_array
        radial_profile_apers = ''
        for rad in radial_profile_apers_array:
            radial_profile_apers = radial_profile_apers+str(rad)+','
        radial_profile_apers = radial_profile_apers[:len(radial_profile_apers)]

        #print (len(radial_profile_apers))
        #print (len(radial_profile_apers_values))
        #print (radial_profile_apers_array)


        shutil.copy(external_dir+'sex_default.param',external_dir+'default_psf.param')
        params = open(external_dir+'default_psf.param','a')
        params.write('MAG_APER('+str(len(radial_profile_apers_values))+') #Fixed aperture magnitude vector [mag]\n')
        params.write('MAGERR_APER('+str(len(radial_profile_apers_values))+') #RMS error vector for fixed aperture mag [mag]\n')
        params.write('FLUX_APER('+str(len(radial_profile_apers_values))+') # Flux within a Kron-like elliptical aperture [count]\n')
        params.write('FLUXERR_APER('+str(len(radial_profile_apers_values))+') #RMS error for AUTO flux [count]\n')
        params.close()

        psf_frame_size_pix = X_psf
        source_cat = temp_dir+'temp.fits'

        command = SE_executable+' '+psf_frame+' -c '+external_dir+'default.sex -CATALOG_NAME '+source_cat+' '+ \
            '-PARAMETERS_NAME '+external_dir+'default_psf.param -DETECT_MINAREA 8 -DETECT_THRESH 3.0 -ANALYSIS_THRESH 3.0 ' + \
            '-DEBLEND_NTHRESH 1 -DEBLEND_MINCONT 1 -MAG_ZEROPOINT ' +str(zp) + ' -BACKPHOTO_TYPE GLOBAL '+\
            '-FILTER Y -FILTER_NAME  '+external_dir+'default.conv -STARNNW_NAME '+external_dir+'default.nnw '+\
            '-BACK_SIZE '+str(psf_frame_size_pix)+' -BACK_FILTERSIZE 1 -PHOT_APERTURES '+radial_profile_apers

        os.system(command)

        table_main = fits.open(source_cat)
        sex_cat_data = table_main[1].data
        RA = sex_cat_data['ALPHA_SKY']
        DEC = sex_cat_data['DELTA_SKY']
        X = sex_cat_data['X_IMAGE']
        Y = sex_cat_data['Y_IMAGE']
        FWHM = sex_cat_data['FWHM_IMAGE']
        MAG_APER = sex_cat_data['MAG_APER']
        FLUX_APER = sex_cat_data['FLUX_APER']
        M = len(RA)
        det_flag = 0
        crossmatch_radius = 1

        while det_flag == 0 :
            crossmatch_radius = crossmatch_radius + 1
            #print (crossmatch_radius)
            for j in range(M):
                ra = RA[j]
                dec = DEC[j]
                x = X[j]
                y = Y[j]
                fwhm = FWHM[j]
                #print (crossmatch_radius)
                if (abs(x-X_psf/2)<crossmatch_radius) and (abs(y-Y_psf/2)<crossmatch_radius):
                    print ('- FWHM for the resampled PSF model is (pixel, arcsec):', fwhm, fwhm*psf_pixel_size)
                    mag_apers = MAG_APER[j]
                    flux_apers = FLUX_APER[j]
                    det_flag = 1
                    print ('- centeroid and fwhm of the PSF model are:', x, y, fwhm)

                    break

        #print (flux_apers)
        for j in range(len(radial_profile_apers_values)):
            if j == 0:
                df = flux_apers[j] - 0
                A = 3.141592 * (radial_profile_apers_values[j]**2) / 4.
            elif j > 0 :
                df = flux_apers[j] - flux_apers[j-1]
                A = 3.141592 * ( (radial_profile_apers_values[j]**2) - (radial_profile_apers_values[j-1]**2) ) / 4.

            d_flux_apers.append(df/A/(psf_pixel_size**2))
            radi_arcsec.append((radial_profile_apers_values[j]/2)*psf_pixel_size)

        if make_plots == 1 :
            ax.scatter(radi_arcsec,d_flux_apers/np.nanmax(d_flux_apers),s=40,marker='o',color=colors[i],label='PSF MODEL for '+fn+'\n(FWHM = '+\
            str(fwhm*psf_pixel_size)[:5]+' arcsec)',alpha=1)

    if make_plots == 1 :
        ax.set_xlabel('Radius [arcsec]')
        ax.set_ylabel('Normalized flux')
        #plt.xscale('log')
        #plt.yscale('log')
        ax.set_xlim([0,3*fwhm*psf_pixel_size])
        ax.set_ylim([-0.05,1.05])
        plt.legend(loc='upper right',fontsize=20)
        plt.tick_params(which='both',direction='in')
        #plt.tight_layout()
        plt.savefig(output_png,dpi=100)
        plt.close()

    return (fwhm)

############################################################

def initial_psf(gal_id):
    estimate_fwhm(gal_id)
    estimate_aper_corr(gal_id)

############################################################
