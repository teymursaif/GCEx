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
import shutil
import photutils
from photutils.aperture import CircularAperture
from photutils.aperture import CircularAnnulus
import scipy.optimize as opt
#from lacosmic import lacosmic

############################################################

def initialize_params() :
    # Defining the working directories
    global working_directory,input_dir,output_dir,data_dir,main_data_dir,clean_data_dir,img_dir,sex_dir,fit_dir,plots_dir,\
    detection_dir,cats_dir,psfs_dir,art_dir,final_cats_dir,temp_dir,sbf_dir,psf_dir

    working_directory = '/data/users/saifollahi/Euclid/ERO/'
    input_dir = '/data/users/saifollahi/Euclid/ERO/inputs/'
    output_dir = '/data/users/saifollahi/Euclid/ERO/outputs/'

    gal_input_cat = input_dir+'udg_input.csv'
    main_data_dir = input_dir+'main_data/'
    data_dir = input_dir+'data/'
    psf_dir = input_dir+'psf/'

    clean_data_dir = output_dir+'clean_data/'
    img_dir = output_dir+'img/'
    sex_dir = output_dir+'sex/'
    fit_dir = output_dir+'fit/'
    plots_dir = output_dir+'plots/'
    detection_dir = output_dir+'detection/'
    cats_dir = output_dir+'cats/'
    psfs_dir = output_dir+'psfs/'
    art_dir = output_dir+'artificial/'
    final_cats_dir = output_dir+'final_cats/'
    temp_dir = output_dir+'temp_files/'
    sbf_dir = output_dir+'sbf/'

    for dir in [working_directory,input_dir,output_dir,data_dir,main_data_dir,clean_data_dir,img_dir,sex_dir,fit_dir,plots_dir,\
    detection_dir,cats_dir,psfs_dir,art_dir,final_cats_dir,temp_dir,sbf_dir,psf_dir] :
        if not os.path.exists(dir): os.makedirs(dir)

    # Getting the objcts and data info
    global gal_id, gal_name, ra, dec, distance, filters, comments, gal_params

    gal_params = {}
    gal_cat = open(gal_input_cat,'r')
    for line in gal_cat:
        if line[0] != '#' :
            line = line.split(' ')
            gal_id, gal_name, ra, dec, distance, filters, comments = int(line[0]), str(line[1]), \
            float(line[2]), float(line[3]), float(line[4]), np.array(line[5].split(',')), np.array((line[6].split('\n')[0]).split(','))
            gal_params[gal_id] = [gal_name, ra, dec, distance, filters, comments]

    # Configuring the pipeline parameters
    global PRIMARY_FRAME_SIZE_ARCSEC, FRAME_SIZE_ARCSEC, GAL_FRAME_SIZE_ARCSEC, \
    N_ART_GCS, PSF_IMAGE_SIZE, INSTR_FOV, COSMIC_CLEAN, GAIN, \
    PHOTOM_APERS, FWHMS_ARCSEC, PSF_REF_RAD_ARCSEC, PSF_REF_RAD_FRAC

    PRIMARY_FRAME_SIZE_ARCSEC = 240 #arcsec
    FRAME_SIZE_ARCSEC = 240 #arcsec
    GAL_FRAME_SIZE_ARCSEC  = 120 #arcsec

    PSF_IMAGE_SIZE = 50 #pixelsatom
    #INSTR_FOV = 0.05 #deg
    N_ART_GCS = 1000
    COSMIC_CLEAN = False
    PHOTOM_APERS = '2,4,8,12,16,20,24,32' # the largest aperture will be used for aperture correction

    ###################33

    # if no PSF is given to the pipeline, FWHM (in arcsec) in eachfilter must be given to the pipeline:
    FWHMS_ARCSEC = {'F115W':[0.0037], 'F150W':[0.049], 'F200W':[0.064], 'F277W':[0.088], 'F356W':[0.114], 'F410M':[0.133], 'F444W':[0.140], \
    'F606W':[0.08], 'F814W':[0.09], 'F125W':[0.12], 'F160W':[0.18], 'g':[1], 'r':[1], 'i':[1]}

    # and the 50% energy radius in arcsec:
    #PSF_RAD50_ARCSEC = {'F115W':[0.025], 'F150W':[0.032], 'F200W':[0.042], 'F277W':[0.055], 'F356W':[0.073], 'F410M':[0.084], 'F444W':[0.090], \
    #'F606W':[0.076], 'F814W':[0.081], 'F125W':[0.12], 'F160W':[0.13], 'g':[0.5], 'r':[0.5], 'i':[0.5]}

    #PSF_RAD80_ARCSEC = {'F115W':[0.136], 'F150W':[0.121], 'F200W':[0.138], 'F277W':[0.179], 'F356W':[0.228], 'F410M':[0.258], 'F444W':[0.276], \
    #'F606W':[0.157], 'F814W':[0.175], 'F125W':[0.25], 'F160W':[0.35], 'g':[1], 'r':[1], 'i':[1]}

    PSF_REF_RAD_ARCSEC = {'F115W':[0.136], 'F150W':[0.121], 'F200W':[0.138], 'F277W':[0.179], 'F356W':[0.228], 'F410M':[0.258], 'F444W':[0.276], \
    'F606W':[0.157], 'F814W':[0.175], 'F125W':[0.25], 'F160W':[0.35], 'g':[1], 'r':[1], 'i':[1]}

    PSF_REF_RAD_FRAC = {'F115W':[0.8], 'F150W':[0.8], 'F200W':[0.8], 'F277W':[0.8], 'F356W':[0.8], 'F410M':[0.8], 'F444W':[0.8], \
    'F606W':[0.8], 'F814W':[0.8], 'F125W':[0.8], 'F160W':[0.8], 'g':[0.8], 'r':[0.8], 'i':[0.8]}

    GAIN = {'F606W':1.5, 'F814W':2, 'g':2.5, 'i':2.5}

############################################################

def welcome():
    print ('\n+ GCTOOLS pipeline v2.0 (2023) ')
    print ('+ Developed by: Teymoor Saifollahi ')
    print ('+ Kapteyn Astronomical Institute')

############################################################

def intro(gal_id):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    global main_filter, other_filters, n_other_filters
    print ('\n+ Analysing galaxy: '+str(gal_name))
    main_filter = str(filters[0])
    other_filters = filters[1:]
    n_other_filters = len(other_filters)
    print ('- Coordinates: RA = '+str(ra)+', DEC = '+str(dec))
    print ('- Distance: '+str(distance)+' Mpc')
    print ('- Available filters: '+str(filters))

############################################################

def copy_data(gal_id):
    print ('- Copying data to the input-data directory: '+(data_dir))
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]

    for fn in filters:
        data_path = main_data_dir+gal_name+'_'+fn+'.fits'
        weight_path = main_data_dir+gal_name+'_'+fn+'.weight.fits'

        if os.path.exists(data_path):
            crop_frame(main_data_dir+gal_name+'_'+fn+'.fits',gal_name,PRIMARY_FRAME_SIZE[fn]/2,fn,ra,dec,format='.fits')
        else:
            print ('*** File does no exists: '+str(data_path))

        if os.path.exists(weight_path):
            crop_frame(main_data_dir+gal_name+'_'+fn+'.weight.fits',gal_name,PRIMARY_FRAME_SIZE[fn]/2,fn,ra,dec,format='.weight.fits')
        else:
            print ('*** Weight does no exists: '+str(data_path)+', crating a weight-map with values of 1')
            make_weight_map(data_path,weight_path)

############################################################

def make_weight_map(data_path,weight_path):
    shutil.copy(data_path,weight_path)
    fits_file = fits.open(weight_path)
    fits_data = fits_file[0].data
    fits_data[abs(fits_data)>-1] = 1
    fits_file[0].data = fits_data
    fits_file.writeto(weight_path, overwrite=True)

############################################################

def median_filter_array(data,fsize = 3):
        from scipy import signal
        return signal.medfilt(data, kernel_size=fsize)

############################################################

def get_data_info(gal_id):
    global PIXEL_SCALES, ZPS, PRIMARY_FRAME_SIZE, FRAME_SIZE, GAL_FRAME_SIZE
    print ('- Extracting pixel-size and zero-point values from the data')
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    PIXEL_SCALES = {}
    ZPS = {}
    PRIMARY_FRAME_SIZE = {}
    FRAME_SIZE = {}
    GAL_FRAME_SIZE = {}
    for fn in filters:
        pixel_scale = get_pixel_scale(main_data_dir+gal_name+'_'+fn+'.fits')
        PIXEL_SCALES[fn] = pixel_scale
        zp = get_zp_AB(main_data_dir+gal_name+'_'+fn+'.fits')
        ZPS[fn] = zp
        PRIMARY_FRAME_SIZE[fn] = int(PRIMARY_FRAME_SIZE_ARCSEC/pixel_scale+0.5)
        FRAME_SIZE[fn] = int(FRAME_SIZE_ARCSEC/pixel_scale+0.5)
        GAL_FRAME_SIZE[fn] = int(GAL_FRAME_SIZE_ARCSEC/pixel_scale+0.5)
    print ('- Pixel-sizes are: '+str(PIXEL_SCALES))
    print ('- Zero-points are: '+str(ZPS))
    print ('- Cut-out sizes are: '+str(FRAME_SIZE))
    #PIXEL_SCALE = PIXEL_SCALES[0]
    #print ('- The adopted pixel-size for this analysis is: '+str(PIXEL_SCALE))

############################################################

def resample_data(gal_id):
    print ('- Resampling data')
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    for fn in filters:
        resample_swarp(data_dir+gal_name+'_'+fn+'.fits',data_dir+gal_name+'_'+fn+'.weight.fits',\
                    gal_name,FRAME_SIZE[fn],fn,ra,dec,pixel_size=PIXEL_SCALES[fn],format='_resampled.fits',weight_format='_resampled.weight.fits')

############################################################

def get_fits_data(fitsfile):
    hdu = fits.open(fitsfile)
    fits_data = hdu[0].data
    return fits_data

############################################################

def get_fits_header(fitsfile):
    hdu = fits.open(fitsfile)
    fits_header = hdu[0].header
    return fits_header

############################################################

def get_header(file,keyword=None):
	'''
	Reads the fits file and outputs the header dictionary.
	OR
	If a keyword is given, returns value of the keyword.
	'''
	fitsfile = fits.open(file)
	if keyword:
		return fitsfile[0].header[keyword]
	else:
		return fitsfile[0].header

############################################################

def get_pixel_scale(fitsfile):
    header = get_fits_header(fitsfile)
    pixel_scale1 = np.sqrt((abs(header['CD1_1'])*3600)**2 + (abs(header['CD2_1'])*3600)**2)
    pixel_scale2 = np.sqrt((abs(header['CD1_2'])*3600)**2 + (abs(header['CD2_2'])*3600)**2)
    pixel_scale = 0.5*(pixel_scale1+pixel_scale2)
    #print (pixel_scale,pixel_scale1,pixel_scale2)
    pixel_scale = (int(pixel_scale*100+0.49999))/100
    return pixel_scale

############################################################

def get_zp_AB(fitsfile):
    header = get_fits_header(fitsfile)
    pixel_scale = get_pixel_scale(fitsfile)
    telescope = header['TELESCOP']

    if (telescope in ['jwst','JWST']):
        PHOTFLAM = header['PHOTFLAM ']
        PHOTPLAM = header['PHOTPLAM ']
        zp = -2.5*np.log10(PHOTFLAM)-5*np.log10(PHOTPLAM)-2.408

    elif telescope in ['hst','HST']:
        PHOTFLAM = header['PHOTFLAM ']
        PHOTPLAM = header['PHOTPLAM ']
        EXPTIME = header['EXPTIME']
        zp = -2.5*np.log10(PHOTFLAM)-5*np.log10(PHOTPLAM)-2.408

    elif telescope in ['ESO-VST']:
        zp = float(header['PHOTZP'])

    zp = (int(zp*100+0.49999))/100
    return zp


############################################################

def attach_sex_tables(tables,output_table) :
    os.system('rm '+output_table)
    out_table = fits.open(tables[0])
    out_table.writeto(output_table)

    for i in range(len(tables)-1) :
        #print (i, len(tables))
        with fits.open(output_table) as hdul1:
            with fits.open(tables[i+1]) as hdul2:
                nrows1 = hdul1[1].data.shape[0]
                nrows2 = hdul2[1].data.shape[0]
                nrows = nrows1 + nrows2
                hdu1 = fits.BinTableHDU.from_columns(hdul1[1].columns, nrows=nrows)
                for colname in hdul1[1].columns.names:
                    hdu1.data[colname][nrows1:] = hdul2[1].data[colname]

                nrows1 = hdul1[1].data.shape[0]
                nrows2 = hdul2[1].data.shape[0]
                nrows = nrows1 + nrows2
                hdu2 = fits.BinTableHDU.from_columns(hdul1[1].columns, nrows=nrows)
                for colname in hdul1[1].columns.names:
                    hdu2.data[colname][nrows1:] = hdul2[1].data[colname]


        os.system('rm '+output_table)
        out_table[1] = hdu1
        out_table[1] = hdu2
        out_table.writeto(output_table)

############################################################

def find_data(gal_id):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    data_for_filters = np.empty(len(filters))
    for fn in filters:
        print ('- Searching for science frames for the galaxy '+str(gal_name)+\
        ' in '+str(fn))
        data_for_filter = list()
        os.system('ls '+data_dir+'*.science.fits > list_of_fits_files.txt')
        fits_files = open('list_of_fits_files.txt','r')
        for fits_file in fits_files:
            fits_file = fits_file.split()
            fits_file = fits_file[0]
            hdu0 = fits.open(fits_file)
            header0 = hdu0[0].header
            CRVAL1 = header0['CRVAL1']
            CRVAL2 = header0['CRVAL2']
            try: FILTER = header0['FILTER']
            except: FILTER = header0['FILTER1']

            if abs(CRVAL1-ra) < INSTR_FOV and abs(CRVAL2-dec) < INSTR_FOV and FILTER == fn:
                print ('--- Data was found for galaxy '+str(gal_name)+' and filter '+str(fn)+': '+str(fits_file))
                data_for_filter.append(fits_file)

        fits_files.close()

############################################################

def resample_swarp(fitsfile, fitsfile_weight, obj_name, radius_pix, filter_name, ra, dec, pixel_size=0, \
    format='_xxx_resampled.fits', weight_format='_xxx_resampled.weight.fits'):
                    #format='_xxx_resampled.fits', weight_format='_xxx_resampled.weight.fits'):
    output = data_dir+obj_name+'_'+filter_name+format
    output_weight = data_dir+obj_name+'_'+filter_name+weight_format

    data = get_fits_data(fitsfile)
    header = get_fits_header(fitsfile)
    n = np.arange(100.0)
    hdu = fits.PrimaryHDU(n)
    hdul = fits.HDUList([hdu])
    hdul[0].header = header
    hdul[0].data = data
    hdul.writeto(temp_dir+'temp.fits',overwrite=True)

    if pixel_size == 0 :
        command = 'swarp '+temp_dir+'temp.fits -c '+input_dir+'default.swarp -IMAGEOUT_NAME '+output+' -WEIGHTOUT_NAME '+output_weight+\
            ' -WEIGHT_TYPE MAP_WEIGHT '+'-WEIGHT_IMAGE '+fitsfile_weight+\
            ' -IMAGE_SIZE '+str(radius_pix)+','+str(radius_pix)+' -PIXEL_SCALE '+str(pixel_size)+\
            ' -CENTER_TYPE MANUAL -CENTER '+str(ra)+','+str(dec)+' -SUBTRACT_BACK N'

    else :
        command = 'swarp '+temp_dir+'temp.fits -c '+input_dir+'default.swarp -IMAGEOUT_NAME '+output+' -WEIGHTOUT_NAME '+output_weight+\
            ' -WEIGHT_TYPE MAP_WEIGHT '+'-WEIGHT_IMAGE '+fitsfile_weight+\
            ' -IMAGE_SIZE '+str(radius_pix)+','+str(radius_pix)+' -PIXELSCALE_TYPE MANUAL -PIXEL_SCALE '+str(pixel_size)+\
            ' -RESAMPLE Y -CENTER_TYPE MANUAL -CENTER '+str(ra)+','+str(dec)+' -SUBTRACT_BACK N'

    #print (command)
    os.system(command)

############################################################

def crop_frame(fitsfile, obj_name, radius_pix, filter_name, ra, dec, pixel_size=0, blur=0, back=0, format='_xxx.fits', output=None) :
    if output == None:
        output = data_dir+obj_name+'_'+filter_name+format
    hdu = fits.open(fitsfile)
    w=WCS(fitsfile)
    radius_pix = int(radius_pix * 1)
    x_center,y_center = w.all_world2pix(ra, dec,0)
    #print (x_center,y_center,radius_pix)
    llx = int(x_center - radius_pix)
    lly = int(y_center - radius_pix)
    urx = int(x_center + radius_pix)
    ury = int(y_center + radius_pix)
    dimx,dimy= len(hdu[0].data[0]),len(hdu[0].data)
    #################
    conversion_factor = 1.
    #try :
    #    unit = hdu[0].header['BUNIT']
    #except:
    #    unit = None

    #if (unit == '10.0*nanoJansky') :
    #    conversion_factor = 0.01
    ################
    if llx<0:llx=0
    if lly<0:lly=0
    if urx>=dimx:urx=dimx-1
    if ury>=dimy:ury=dimy-1
    #print ('+++ Cropping area',llx,lly,'(x0,y0)',urx,ury,'(x1,y1)')
    #if (urx - llx != ury - lly) :
    #    return 0
    #print (lly,ury,llx,urx)
    object = hdu[0].data[lly:ury,llx:urx] * conversion_factor
    #object_scaled = np.log10(hdu[0].data[lly:ury,llx:urx]*(1.0e15))  #scaling
    template = hdu
    if blur > 0 :
        #object = gaussian_filter(object,sigma=blur)
        object = ndimage.median_filter(object, size=blur)
    template[0].header['NAXIS1'] = urx - llx
    template[0].header['NAXIS2'] = ury - lly
    #template[0].header['EXPTIME'] = 1.0
    #template[0].header['GAIN'] = 1.0
    #print (urx - llx,ury - lly)
    template[0].header['CRPIX1'] = hdu[0].header['CRPIX1']-llx
    template[0].header['CRPIX2'] = hdu[0].header['CRPIX2']-lly
    object = object-back
    where_are_NaNs = np.isnan(object)
    object[where_are_NaNs] = 0
    template[0].data = object
    template.writeto(output, overwrite=True, output_verify='fix')
    del template,hdu

############################################################

def copy_header(fitsfile1,fitsfile2):
    hdu1 = fits.open(fitsfile1)
    hdu2 = fits.open(fitsfile2)
    hdu2[0].header = hdu1[0].header
    hdu2.writeto(fitsfile2, overwrite=True, output_verify='fix')

############################################################

def make_fancy_png(fitsfile,pngfile,text='',zoom=1) :
    main = fits.open(fitsfile)
    header = get_fits_header(fitsfile)
    X = header['NAXIS1']
    Y = header['NAXIS2']
    image = main[0].data
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    x1 = int(X/2 - X/2/zoom)
    x2 = int(X/2 + X/2/zoom)
    y1 = int(Y/2 - Y/2/zoom)
    y2 = int(Y/2 + Y/2/zoom)
    image0 = image[x1:x2,y1:y2]

    #image = sigma_clip(image,sigma=3,maxiters=1)
    scale = ZScaleInterval() #LogStretch()
    #min_ = np.nanmedian(image)-0.5*np.nanstd(image)
    #max_ = np.nanmedian(image)+0.1*np.nanstd(image)
    ax.imshow(scale(image0),cmap='gist_gray') #LogNorm #,vmin=min_, vmax=max_
    #ax.axis('off')
    ax.invert_yaxis()
    #fig.tight_layout()
    #[0., 0., 1., 1.]
    ax.text(int(X*0.05),int(Y*0.95),text,color='red',fontsize=30)
    fig.savefig(pngfile,dpi=150)
    plt.close()

############################################################

def make_galaxy_frames(gal_id, resampled=False):
    print ('- Making cropped frames and weight maps')
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]

    if resampled == True: resampled='_resampled'
    elif resampled == False: resampled=''

    for fn in filters:
        #initial crop for maiing life easier!
        crop_frame(data_dir+gal_name+'_'+fn+resampled+'.fits',gal_name,FRAME_SIZE[fn]/2,fn,ra,dec,format='_cropped.fits')
        crop_frame(data_dir+gal_name+'_'+fn+resampled+'.weight.fits',gal_name,FRAME_SIZE[fn]/2,fn,ra,dec,format='_cropped.weight.fits')
        crop_frame(data_dir+gal_name+'_'+fn+resampled+'.fits',gal_name,GAL_FRAME_SIZE[fn]/2,fn,ra,dec,format='_gal_cropped.fits')
        crop_frame(data_dir+gal_name+'_'+fn+resampled+'.weight.fits',gal_name,GAL_FRAME_SIZE[fn]/2,fn,ra,dec,format='_gal_cropped.weight.fits')

        make_fancy_png(data_dir+gal_name+'_'+fn+'_gal_cropped.fits',img_dir+gal_name+'_'+fn+'_gal_cropped.jpg',text=gal_name+' '+fn)

############################################################

def make_rgb(gal_id,rgb_filters=None):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    if rgb_filters == None:
        print ('RGB filters are not specified. Skipping making an RGB image ...')
        return

    fn_r = rgb_filters[2]
    fn_g = rgb_filters[1]
    fn_b = rgb_filters[0]
    fitsfile_r = data_dir+gal_name+'_'+fn_r+'_gal_cropped.fits'
    fitsfile_g = data_dir+gal_name+'_'+fn_g+'_gal_cropped.fits'
    fitsfile_b = data_dir+gal_name+'_'+fn_b+'_gal_cropped.fits'

    image_r = get_fits_data(fitsfile_r)
    image_g = get_fits_data(fitsfile_g)
    image_b = get_fits_data(fitsfile_b)

    fwhm_r = (FWHMS_ARCSEC[fn_r])[0]
    fwhm_g = (FWHMS_ARCSEC[fn_g])[0]
    fwhm_b = (FWHMS_ARCSEC[fn_b])[0]
    #print (fwhm_r,fwhm_g,fwhm_b)
    fwhm_max = np.max([fwhm_r,fwhm_g,fwhm_b])
    #print (fwhm_max)

    #image_r = gaussian_filter(image_r,sigmea=fwhm_max/fwhm_r)
    #image_g = gaussian_filter(image_g,sigma=fwhm_max/fwhm_g)
    #image_b = gaussian_filter(image_b,sigma=fwhm_max/fwhm_b)

    image = make_lupton_rgb(image_r, image_g, image_b, Q=5, stretch=0.5)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, origin='lower') #LogNorm
    ax.invert_yaxis()
    pngfile = img_dir+gal_name+'_RGB_gal_cropped.jpg'
    fig.savefig(pngfile,dpi=150)
    plt.close()


############################################################

def clean_data(gal_id): #tbw
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    if COSMIC_CLEAN == False :
        clean_data_dir = data_dir
    elif COSMIC_CLEAN == True :
        #running LACOSMIC (add gain as input, tbw)
        clean_cosmic(data_dir+gal_name+'_'+fn+'_cropped.fits',
            clean_data_dir+gal_name+'_'+fn+'_cropped.fits', gain=1)
        data_dir = clean_data_dir

############################################################

def clean_cosmic(dirty_frame,clean_frame,gain=1):
    fits_dirty = get_fits_data(dirty_frame)
    fits_dirty_data = fits_dirty[0].data
    cleaned_image, cosmic_mask = lacosmic(fits_dirty_data,3.0,0.5,0.2,effective_gain=gain,readnoise=100,maxiter=5)
    fits_dirty_data = fits_dirty[0].data
    fits_dirty[0].data = cleaned_image
    fits_dirty.writeto(clean_frame,overwrite=True)

############################################################

def expand_fits_table(table,new_param,new_param_values) :
    fits = FITS(table,'rw')
    fits[-1].insert_column(name = new_param, data = new_param_values)
    fits.close()

############################################################

def update_header(fits_file_name, header_keyword, value):
    fits_file = fits.open(fits_file_name)
    fits_file[0].header[header_keyword] = value
    fits_file.writeto(fits_file_name, overwrite=True)


############################################################

def estimate_aper_corr(gal_id):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    for fn in filters:
        try:
            psf_file = psf_dir+'psf_'+fn+'.fits'
            aper_size_arcsec = (PSF_REF_RAD_ARCSEC[fn])[0]
            #aper_size_pixel = aper_size_arcsec/PIXEL_SCALES[fn]
            print ('- estimating aperture correction value for filter and aperture radius (arcsec): '\
                , fn, ', ', aper_size_arcsec)

            psf_fits_file = fits.open(psf_file)
            #print (psf_file)
            psf_data = psf_fits_file[0].data
            #print (psf_data)
            psf_pixel_scale = psf_fits_file[0].header['PIXELSCL']
            X = float(psf_fits_file[0].header['NAXIS1'])
            Y = float(psf_fits_file[0].header['NAXIS2'])
            aper_size_pixel = aper_size_arcsec/psf_pixel_scale

            total_flux = np.nansum(psf_data)

            aper = CircularAperture((X/2., Y/2.), aper_size_pixel)
            aper_area = aper.area_overlap(data=psf_data,method='exact')
            flux, flux_err = aper.do_photometry(data=psf_data,method='exact')

            flux_ratio = float(flux[0]) / total_flux
            #print (aper_size_pixel, flux_ratio)
            PSF_REF_RAD_FRAC = flux_ratio
        except:
            donothing = 1

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
            print ("* PSF file is not found. using the default value of FWHM (arcsec)", (FWHMS_ARCSEC[fn])[0])

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
###
welcome()
initialize_params()
get_data_info(gal_id)
estimate_fwhm(gal_id)
estimate_aper_corr(gal_id)
