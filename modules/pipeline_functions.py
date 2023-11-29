import os, sys
import math
import numpy as np
import matplotlib.pyplot as plt
#from astroquery.mast import Observations
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from astropy.stats import sigma_clip
from astropy.visualization import *
from astropy.visualization import make_lupton_rgb
from astropy.table import Table, join_skycoord, Column
from astropy import table
import shutil
import photutils
from photutils.aperture import CircularAperture
from photutils.aperture import CircularAnnulus
import scipy.optimize as opt
from fitsio import FITS
from modules.initialize import *
#from modules.psf import *
#from lacosmic import lacosmic

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

def intro(gal_id):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    data_name = gal_data_name[gal_id]
    global main_filter, other_filters, n_other_filters
    print (f"{bcolors.OKCYAN}- Analysis Started ... "+ bcolors.ENDC)
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
    data_name = gal_data_name[gal_id]

    for fn in filters:

        main_data_path = main_data_dir+data_name+'_'+fn+'.fits'

        pixel_scale = get_pixel_scale(main_data_path)
        PIXEL_SCALES[fn] = pixel_scale
        PRIMARY_FRAME_SIZE[fn] = int(PRIMARY_FRAME_SIZE_ARCSEC/pixel_scale+0.5)
        FRAME_SIZE[fn] = int(FRAME_SIZE_ARCSEC/pixel_scale+0.5)
        GAL_FRAME_SIZE[fn] = int(GAL_FRAME_SIZE_ARCSEC/pixel_scale+0.5)

        data_path = data_dir+gal_name+'_'+fn+'.fits'
        main_weight_path = main_data_dir+data_name+'_'+fn+'.weight.fits'
        weight_path = main_data_dir+data_name+'_'+fn+'.weight.fits'

        if os.path.exists(main_data_path):
            crop_frame(main_data_dir+data_name+'_'+fn+'.fits',gal_name,PRIMARY_FRAME_SIZE[fn]/2,fn,ra,dec,format='.fits')
        else:
            print ('*** File does no exists: '+str(data_path))

        if os.path.exists(main_weight_path):
            crop_frame(main_data_dir+data_name+'_'+fn+'.weight.fits',gal_name,PRIMARY_FRAME_SIZE[fn]/2,fn,ra,dec,format='.weight.fits')
        else:
            print ('*** Weight does no exists: '+str(data_path)+', crating a weight-map with values of 1')
            make_weight_map(main_data_path,main_weight_path)
            crop_frame(main_data_dir+data_name+'_'+fn+'.weight.fits',gal_name,PRIMARY_FRAME_SIZE[fn]/2,fn,ra,dec,format='.weight.fits')

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
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    data_name = gal_data_name[gal_id]
    methods = gal_methods[gal_id]

    print ('- Check the input data for compatibility with pipeline')
    for fn in filters:
        try:
            gain = get_gain(data_dir+gal_name+'_'+fn+'.fits')
        except:
            update_header(data_dir+gal_name+'_'+fn+'.fits', 'GAIN', INPUT_GAIN[fn])
            print (f"{bcolors.WARNING}- No GAIN is found in the header of the fits file. The value is set to the value in the configuration file.."+ bcolors.ENDC)

        try:
            exptime = get_exptime(data_dir+gal_name+'_'+fn+'.fits')
            if exptime <= 1e-9:
                update_header(data_dir+gal_name+'_'+fn+'.fits', 'EXPTIME', INPUT_EXPTIME[fn])
                print (f"{bcolors.WARNING}- The EXPTIME value found in the header of the fits file is not valid. The value is set to the value in the configuration file."+ bcolors.ENDC)
        except:
            update_header(data_dir+gal_name+'_'+fn+'.fits', 'EXPTIME', INPUT_EXPTIME[fn])
            print (f"{bcolors.WARNING}- No EXPTIME is found in the header of the fits file. The value is set to the value in the configuration file."+ bcolors.ENDC)

        try:
            zp = get_zp_AB(data_dir+gal_name+'_'+fn+'.fits')
        except:
            update_header(data_dir+gal_name+'_'+fn+'.fits', 'MAGZERO', INPUT_ZP[fn])
            print (f"{bcolors.WARNING}- No ZP is found in the header of the fits file. The value is set to the value in the configuration file."+ bcolors.ENDC)

        try:
            get_header(data_dir+gal_name+'_'+fn+'.fits','FILTER')
        except:
            update_header(data_dir+gal_name+'_'+fn+'.fits', 'FILTER', fn)
            print (f"{bcolors.WARNING}- No FILTER is found in the header of the fits file. The value is set on the value in the given filter-name."+ bcolors.ENDC)

    ################

    print ('- Extracting pixel-size and zero-point values from the data')
    for fn in filters:
        pixel_scale = get_pixel_scale(data_dir+gal_name+'_'+fn+'.fits')
        PIXEL_SCALES[fn] = pixel_scale
        gain = get_gain(data_dir+gal_name+'_'+fn+'.fits')
        GAIN[fn] = gain
        zp = get_zp_AB(data_dir+gal_name+'_'+fn+'.fits')
        ZPS[fn] = zp
        exptime = get_exptime(data_dir+gal_name+'_'+fn+'.fits')
        EXPTIME[fn]=exptime
        PRIMARY_FRAME_SIZE[fn] = int(PRIMARY_FRAME_SIZE_ARCSEC/pixel_scale+0.5)
        FRAME_SIZE[fn] = int(FRAME_SIZE_ARCSEC/pixel_scale+0.5)
        GAL_FRAME_SIZE[fn] = int(GAL_FRAME_SIZE_ARCSEC/pixel_scale+0.5)
    print ('- Frame exposure times are:'+str(EXPTIME))
    print ('- Pixel-sizes are: '+str(PIXEL_SCALES))
    print ('- Zero-points are: '+str(ZPS))
    print ('- GAINs are : '+str(GAIN))

    print ('- Cut-out sizes are: '+str(FRAME_SIZE))

    #PIXEL_SCALE = PIXEL_SCALES[0]
    #print ('- The adopted pixel-size for this analysis is: '+str(PIXEL_SCALE))


############################################################

def resample_data(gal_id):
    print ('- Resampling data')
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    data_name = gal_data_name[gal_id]
    for fn in filters:
        resample_swarp(data_dir+gal_name+'_'+fn+'.fits',data_dir+gal_name+'_'+fn+'.weight.fits',\
                    gal_name,FRAME_SIZE[fn],fn,ra,dec,pixel_size=PIXEL_SCALES[fn],format='_resampled.fits',weight_format='_resampled.weight.fits')

############################################################

def get_fits_data(fitsfile):
    #Function is developed by Aku Venhola
    hdu = fits.open(fitsfile)
    fits_data = hdu[0].data
    return fits_data

############################################################

def get_fits_header(fitsfile):
    #Function is developed by Aku Venhola
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
    #Function is developed by Aku Venhola

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

def get_gain(fitsfile):
    header = get_fits_header(fitsfile)
    try:
        try:
            gain = header['CCDGAIN']
        except:
            gain = header['GAIN']
    except:
        print (f"{bcolors.WARNING}*** no GAIN has been found in the header of the input data. Gain is set on 1.0."+ bcolors.ENDC)
        gain = 1
    return gain

############################################################

def get_zp_AB(fitsfile):
    header = get_fits_header(fitsfile)
    pixel_scale = get_pixel_scale(fitsfile)
    try:
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
    except:
        if header['FILTER'] == 'VIS' :
            zp = float(header['MAGZERO'])
        else:
            zp = float(header['MAGZERO'])



    zp = (int(zp*100+0.49999))/100
    return zp

############################################################

def get_exptime(fitsfile):
    header = get_fits_header(fitsfile)
    exptime = header['EXPTIME']
    return exptime

############################################################

def add_fits_files(fits1, fits2, out_fits):
    img1 = fits.open(fits1)
    img2 = fits.open(fits2)
    data1 = img1[0].data
    data2 = img2[0].data
    data1 = data1+data2
    img1[0].data = data1
    img1.writeto(out_fits,overwrite=True)

############################################################

def attach_sex_tables(tables,output_table) :
    os.system('rm '+output_table)
    out_table = fits.open(tables[0])
    out_table.writeto(output_table,overwrite=True)

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
        out_table.writeto(output_table,overwrite=True)

############################################################

def find_data(gal_id):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    data_name = gal_data_name[gal_id]
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
        command = swarp_executable+' '+temp_dir+'temp.fits -c '+external_dir+'default.swarp -IMAGEOUT_NAME '+output+' -WEIGHTOUT_NAME '+output_weight+\
            ' -WEIGHT_TYPE MAP_WEIGHT '+'-WEIGHT_IMAGE '+fitsfile_weight+\
            ' -IMAGE_SIZE '+str(radius_pix)+','+str(radius_pix)+' -PIXEL_SCALE '+str(pixel_size)+\
            ' -CENTER_TYPE MANUAL -CENTER '+str(ra)+','+str(dec)+' -SUBTRACT_BACK N'# -VERBOSE_TYPE QUIET'

    else :
        command = swarp_executable+' '+temp_dir+'temp.fits -c '+external_dir+'default.swarp -IMAGEOUT_NAME '+output+' -WEIGHTOUT_NAME '+output_weight+\
            ' -WEIGHT_TYPE MAP_WEIGHT '+'-WEIGHT_IMAGE '+fitsfile_weight+\
            ' -IMAGE_SIZE '+str(radius_pix)+','+str(radius_pix)+' -PIXELSCALE_TYPE MANUAL -PIXEL_SCALE '+str(pixel_size)+\
            ' -RESAMPLE Y -CENTER_TYPE MANUAL -CENTER '+str(ra)+','+str(dec)+' -SUBTRACT_BACK N'# -VERBOSE_TYPE QUIET'

    #print (command)
    os.system(command)
    return output,output_weight

############################################################

def crop_frame(fitsfile, obj_name, radius_pix, filter_name, ra, dec, pixel_size=0, blur=0, back=0, format='_xxx.fits', output=None, save_fits=True) :
    # original function by Aku Venhola, modified by Teymoor Saifollahi
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
    if save_fits == True:
        template.writeto(output, overwrite=True, output_verify='fix')
    del template,hdu


############################################################

def copy_header(fitsfile1,fitsfile2):
    hdu1 = fits.open(fitsfile1)
    hdu2 = fits.open(fitsfile2)
    hdu2[0].header = hdu1[0].header
    hdu2.writeto(fitsfile2, overwrite=True, output_verify='fix')

############################################################

def make_fancy_png(fitsfile,pngfile,text='',zoom=1, mode='lsb', cmap='gist_gray') :
    main = fits.open(fitsfile)
    header = get_fits_header(fitsfile)
    X = header['NAXIS1']
    Y = header['NAXIS2']
    #print (X,Y)
    image = main[0].data
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    x1 = int(X/2 - X/2/zoom)
    x2 = int(X/2 + X/2/zoom)
    y1 = int(Y/2 - Y/2/zoom)
    y2 = int(Y/2 + Y/2/zoom)
    dx = x2-x1
    dy = y2-y1
    image = image[y1:y2,x1:x2]

    if dx > 5000 and dy > 5000:
        print ('* frame is too large. Rebining the frame 10 times.')
        x1, x2, y1, y2 = 0, int(dx-(dx%10)), 0, int(dy-(dy%10))
        dx = int(x2-x1)
        dy = int(y2-y1)
        #print (dx,dy)
        image = image[y1:y2,x1:x2]
        #print (np.shape)
        #print (int(dx/5),int(dy/5))
        image = rebin(image,(int(dy/10),int(dx/10)))


    scale = ZScaleInterval() #LogStretch()
    ax.imshow(scale(image),cmap=cmap) #LogNorm #,vmin=min_, vmax=max_
    #ax.axis('off')
    ax.invert_yaxis()
    #fig.tight_layout()
    #[0., 0., 1., 1.]
    #ax.text(int(X*0.05),int(Y*0.90),text,color='red',fontsize=50)
    fig.savefig(pngfile+'.zscale.png',dpi=150)

    image0 = sigma_clip(image,3, masked=False)
    min_g = np.nanmedian(image0)-0.05*np.nanstd(image0)
    max_g = np.nanmedian(image0)+5*np.nanstd(image)
    image0 = image - min_g
    image0 = image0 / (max_g - min_g)
    image0[image0>1]=1
    image0[image0<0]=0
    image0 = image0*255
    image0 = np.log(image0**0.5+10)

    #print (min_g,max_g)
    ax.imshow((image0),cmap=cmap)
    ax.invert_yaxis()
    #ax.text(int(X*0.05),int(Y*0.90),text,color='red',fontsize=50)
    fig.savefig(pngfile+'.log.png',dpi=100)

    zoom = 6
    x1 = int(X/2 - X/2/zoom)
    x2 = int(X/2 + X/2/zoom)
    y1 = int(Y/2 - Y/2/zoom)
    y2 = int(Y/2 + Y/2/zoom)
    dx = x2-x1
    dy = y2-y1
    image0z = image0[y1:y2,x1:x2]

    image0 = sigma_clip(image,3, masked=False)
    min_g = np.nanmedian(image0)-0.1*np.nanstd(image0)
    max_g = np.nanmedian(image0z)+5*np.nanstd(image0)
    image0 = image - min_g
    image0 = image0 / (max_g - min_g)
    image0[image0>1]=1
    image0[image0<0]=0
    image0 = image0*255
    image0 = np.log(image0**0.5+10)

    ax.imshow((image0z),cmap=cmap)
    ax.invert_yaxis()
    #ax.text(int(X*0.05),int(Y*0.90),text,color='red',fontsize=50)
    fig.savefig(pngfile+'.log.zoom.png',dpi=150)

    ###

    image0 = sigma_clip(image,3, masked=False)
    image0 = gaussian_filter(image0,sigma=0.25)

    min_g = np.nanmedian(image0)-0.1*np.nanstd(image0)
    max_g = np.nanmedian(image0)+5*np.nanstd(image)

    min_b = np.nanmedian(image0)-0.1*np.nanstd(image0)
    max_b = np.nanmedian(image0)+0.1*np.nanstd(image0)

    image0 = image
    image0 = gaussian_filter(image0,sigma=0.25)

    image = image - min_g
    image = image / (max_g - min_g)
    image[image>1]=1
    image[image<0]=0
    image = image*255
    image = np.log(image**0.5+100)
    image = (image-np.nanmin(image)) / (np.nanmax(image)-np.nanmin(image))
    image = image*255

    image0 = image0 - min_b
    image0 = image0 / (max_b - min_b)
    image0[image0>1]=1
    image0[image0<0]=0
    image0 = (1-image0)*255
    image0[image0==0] = image[image0==0]
    ax.imshow((image0),cmap=cmap,vmin=0, vmax=255)
    ax.invert_yaxis()
    #ax.text(int(X*0.05),int(Y*0.95),text,color='red',fontsize=30)
    fig.savefig(pngfile+'.lsb.png',dpi=150)

    plt.close()

############################################################

def make_galaxy_frames(gal_id, resampled=False):
    print ('- Making cropped frames and weight maps')
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    data_name = gal_data_name[gal_id]
    methods = gal_methods[gal_id]

    if 'RESAMPLE'in methods:
        resampled='_resampled'
    else:
        resampled=''

    for fn in filters:
        #initial crop for maiing life easier!
        crop_frame(data_dir+gal_name+'_'+fn+resampled+'.fits',gal_name,FRAME_SIZE[fn]/2,fn,ra,dec,format='_cropped.fits')
        crop_frame(data_dir+gal_name+'_'+fn+resampled+'.weight.fits',gal_name,FRAME_SIZE[fn]/2,fn,ra,dec,format='_cropped.weight.fits')
        #try:
        make_fancy_png(data_dir+gal_name+'_'+fn+'_cropped.fits',img_dir+gal_name+'_'+fn+'_cropped.jpg',text=gal_name+' '+fn)
        #except:
        #print ('*** something happened (not enough memory?) and the fancy JPG frame could not be made.')

        if 'FIT_GAL' in methods:
            crop_frame(data_dir+gal_name+'_'+fn+resampled+'.fits',gal_name,GAL_FRAME_SIZE[fn]/2,fn,ra,dec,format='_gal_cropped.fits')
            crop_frame(data_dir+gal_name+'_'+fn+resampled+'.weight.fits',gal_name,GAL_FRAME_SIZE[fn]/2,fn,ra,dec,format='_gal_cropped.weight.fits')

            make_fancy_png(data_dir+gal_name+'_'+fn+'_gal_cropped.fits',img_dir+gal_name+'_'+fn+'_gal_cropped.jpg',text=gal_name+' '+fn)

############################################################

def make_rgb(gal_id,rgb_filters=None):
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    data_name = gal_data_name[gal_id]

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

    fwhm_r = (FWHMS_ARCSEC[fn_r])
    fwhm_g = (FWHMS_ARCSEC[fn_g])
    fwhm_b = (FWHMS_ARCSEC[fn_b])
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
    data_name = gal_data_name[gal_id]
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

def color(fn1,fn2):
    c = GC_REF_MAG[fn1] - GC_REF_MAG[fn2]
    return c

############################################################

def expand_fits_table(table,new_param,new_param_values) :
    fits = FITS(table,'rw')
    fits[-1].insert_column(name = new_param, data = new_param_values)
    fits.close()
    #f = fits.open(table)
    #values_to_add = new_param_values
    #column = fits.Column(name=new_param, format="L", array=values_to_add)
    #f[1].columns.add_col(column)
    #f.writeto(table,overwrite=True)

############################################################

#def expand_fits_table(input_table,new_param,new_param_values) :
#    #fits_file = fits.open(input_table)
#    #data = fits_file[1].data
#    table = Table.read(input_table, format='fits')
#    #print (table)
#    c = Column(data=new_param_values, name=new_param)
#    table.add_column(c)
#    table.write(input_table, format='fits', overwrite='True')
#    #table.close()


############################################################

def update_header(fits_file_name, header_keyword, value):
    fits_file = fits.open(fits_file_name)
    fits_file[0].header[header_keyword] = value
    fits_file.writeto(fits_file_name, overwrite=True)

############################################################

def csv_to_fits(input_csv,output_fits):
    """
    csv-to-fits.py  INPUT.csv  OUTPUT.fits

    Convert a CSV file to a FITS bintable

    - In the first line of the CSV are described names of columns.
        - A column named "id" is treated as 64bit integer.
        - A column named "starnotgal" is treated as boolean (0/1).
        - Columns named "ra" and "dec" are treated as double precision float.
        - Other columns are treated as single precision float.

    """
    # Function is developed by some nice people on the internet
    recarray = loadCsvAsNumpy(input_csv)
    bintable = convertNumpyToFitsBinTable(recarray)
    saveFitsBinTable(bintable, output_fits)

############################################################

def loadCsvAsNumpy(filename):
    # Function is developed by some nice people on the internet
    dtypes = {
        "id" : np.int64,
        "starnotgal" : np.bool_,
        "RA" : np.float64,
        "DEC" : np.float64,
    }

    def nameFilter(name):
        uname = name.upper()
        if uname == "RA" or uname == "DEC":
            return uname
        elif uname == "RA_ERR":
            return "RA_err"
        elif uname == "DEC_ERR":
            return "DEC_err"
        elif uname == "ID" or uname == "STARNOTGAL":
            return name.lower()
        else:
            return name

    with open(filename) as f:
        headers = map(nameFilter, f.readline().strip().split(','))

    dtype = [ (name, dtypes[name] if name in dtypes else np.float32)
        for name in headers ]

    return np.loadtxt(filename,
        dtype = dtype, delimiter = ',', skiprows = 1)

############################################################

def convertNumpyToFitsBinTable(recarray):
    #Function is developed by some nice people on the internet
    return fits.FITS_rec.from_columns(fits.ColDefs(recarray))

############################################################

def saveFitsBinTable(bintable, filename):
    #Function is developed by some nice people on the internet
    primaryHDU = fits.PrimaryHDU()
    binTableHDU = fits.BinTableHDU(bintable)
    fits.HDUList([primaryHDU, binTableHDU]).writeto(filename)

############################################################

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).sum(-1).sum(1)

############################################################

def merge_cats(): #gcs, sim_gcs

    print ('- Merging the catalogs produced in this run...')

    # 1
    source_cats = []
    for gal_id in gal_params.keys():
        gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
        cat_name = cats_dir+gal_name+'_master_cat_forced.fits'
        source_cats.append(cat_name)
        id = gal_id

    data_name = gal_data_name[id]
    output_cat = cats_dir+data_name+'_forced_merged.fits'
    topcat_friendly_output_cat = cats_dir+data_name+'_forced_merged+.fits'
    attach_sex_tables(source_cats,output_cat)
    clean_cat = clean_dublicates_in_cat(output_cat)
    make_cat_topcat_friendly(clean_cat,topcat_friendly_output_cat)

    # 2
    #print ('test')
    if EXTRACT_DWARFS == True:
        source_cats = []
        for gal_id in gal_params.keys():
            gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
            cat_name = cats_dir+gal_name+'_lsb_master_cat_forced.fits'
            source_cats.append(cat_name)
            id = gal_id

        data_name = gal_data_name[id]
        output_cat = cats_dir+data_name+'_lsb_forced_merged.fits'
        topcat_friendly_output_cat = cats_dir+data_name+'_lsb_forced_merged+.fits'
        attach_sex_tables(source_cats,output_cat)
        clean_cat = clean_dublicates_in_cat(output_cat)
        make_cat_topcat_friendly(clean_cat,topcat_friendly_output_cat)


############################################################

def merge_gc_cats(): #gcs, sim_gcs

    print ('- Merging the catalogs produced in this run...')

    # gc-cat
    source_cats = []
    for gal_id in gal_params.keys():
        gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
        cat_name = final_cats_dir+gal_name+'_selected_GCs.fits'
        source_cats.append(cat_name)
        id = gal_id

    data_name = gal_data_name[id]
    output_cat = final_cats_dir+data_name+'_selected_GCs_merged.fits'
    topcat_friendly_output_cat = final_cats_dir+data_name+'_selected_GCs_merged+.fits'
    attach_sex_tables(source_cats,output_cat)
    clean_cat = clean_dublicates_in_cat(output_cat)
    make_cat_topcat_friendly(clean_cat,topcat_friendly_output_cat)


    ### selected-source 
    source_cats = []
    for gal_id in gal_params.keys():
        gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
        cat_name = final_cats_dir+gal_name+'_selected_GCs+.size-selected.fits'
        source_cats.append(cat_name)
        id = gal_id

    data_name = gal_data_name[id]
    output_cat = final_cats_dir+data_name+'_selected_GCs+.size-selected_merged+.fits'
    topcat_friendly_output_cat = final_cats_dir+data_name+'_selected_GCs+.size-selected_merged+.fits'
    attach_sex_tables(source_cats,output_cat)
    clean_cat = clean_dublicates_in_cat(output_cat)
    make_cat_topcat_friendly(clean_cat,topcat_friendly_output_cat)

############################################################

def merge_sims(): #gcs, sim_gcs

    print ('- Merging the simulated catalogues produced in this run...')

    # 1
    source_cats = []
    for gal_id in gal_params.keys():
        gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
        fn_det = filters[0]
        cat_name = art_dir+gal_name+'_'+fn_det+'_ALL_DET_ART_GCs.fits'
        source_cats.append(cat_name)
        id = gal_id

    data_name = gal_data_name[id]
    output_cat = cats_dir+data_name+'_ALL_DET_ART_GCs_merged.fits'
    topcat_friendly_output_cat = cats_dir+data_name+'_ALL_DET_ART_GCs_merged+.fits'
    attach_sex_tables(source_cats,output_cat)
    make_cat_topcat_friendly(output_cat,topcat_friendly_output_cat)

    # 2
    source_cats = []
    for gal_id in gal_params.keys():
        gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
        fn_det = filters[0]
        cat_name = art_dir+gal_name+'_'+fn_det+'_ALL_ART_GCs.fits'
        source_cats.append(cat_name)
        id = gal_id

    data_name = gal_data_name[id]
    output_cat = cats_dir+data_name+'_ALL_ART_GCs_merged.fits'
    topcat_friendly_output_cat = cats_dir+data_name+'_ALL_ART_GCs_merged+.fits'
    attach_sex_tables(source_cats,output_cat)
    make_cat_topcat_friendly(output_cat,topcat_friendly_output_cat)

############################################################

def make_cat_topcat_friendly(input_cat,output_cat):
    ttypes = list()
    cat = fits.open(input_cat)
    n_cols = (len((cat[1].data)[0]))
    for i in range (0,n_cols) :
        #i = i+1
        #ttypes.append('TTYPE'+str(i))
        #for ttype in ttypes :
        #print (cat[1].header[ttype])
        col_name = cat[1].columns[i].name
        #print (col_name)
        if '-' in col_name :
            cat[1].columns[i].name = col_name.replace('-','_')

    cat.writeto(output_cat,overwrite=True)

############################################################

def clean_dublicates_in_cat(cat_name):

    # Thanks to Petra Awad for helping to make the function faster

    main = fits.open(cat_name)
    data = main[1].data
    #data = data[:10000]
    RA = data['RA']
    DEC = data['DEC']
    N = len(RA)
    #print (N)
    R = np.vstack((RA, DEC)).T
    l = 0.1/3600
    repeated_idxs = []
    unique_idxs = np.full(N, True, dtype=bool)

    M = rangesearch(R, R, l)
    C = np.array([len(m) for m in M])
    ind = np.where(C > 1)[0]
    #print(len(ind))
    possible_repeated_indices = ind
    #print (possible_repeated_indices)

    for i in possible_repeated_indices:

        if i in repeated_idxs:
            #unique_idxs.append(False)
            unique_idxs[i] = False
            #print (i)
            continue

        #unique_flag = 1
        ra0 = RA[i]
        dec0 = DEC[i]

        t = np.argwhere( (abs(RA-ra0)<l) & (abs(DEC-dec0)<l) )
        tf = t.flatten(order='c')
        tfd = np.setdiff1d(tf,i)

        for idx in tfd:
            repeated_idxs.append(idx)
            #print (idx)

        #if unique_flag == 1 :
            #unique_idxs.append(True)

    data = data[unique_idxs]
    #print (len(data))
    main[1].data = data
    main.writeto(cat_name+'.unique.fits',overwrite=True)

    return (cat_name+'.unique.fits')

############################################################

def rangesearch(X, Y, Radius, return_distances = False):

    from sklearn.neighbors import KDTree

    '''
    - Short-hand for applying MATLAB's rangesearch function
    for finding indices and distances to nearest neighbors (NN).  
    - Default for returning distances is False.
    '''
    
    tree = KDTree(X)
    NN = tree.query_radius(Y, Radius, return_distances)
    
    return NN

############################################################

def crossmatch(cat1,cat2,ra_param1,dec_param1,ra_param2,dec_param2,max_sep_arcsec,filter_name2,output_cat):

    cat = fits.open(cat1, ignore_missing_end=True)
    cat1_data = cat[1].data
    ra1 = cat1_data[ra_param1]
    dec1 = cat1_data[dec_param1]

    cat = fits.open(cat2, ignore_missing_end=True)
    cat2_data = cat[1].data

    if (ra_param2 == ra_param1) and (dec_param2 == dec_param1):
        cat[1].columns[ra_param2].name = ra_param2+'_'
        cat[1].columns[dec_param2].name = dec_param2+'_'
        ra_param2 = ra_param2+'_'
        dec_param2 = dec_param2+'_'
        cat2 = temp_dir+'temp0.fits'
        cat.writeto(cat2 ,overwrite=True)
        cat = fits.open(cat2, ignore_missing_end=True)
        cat2_data = cat[1].data
        
    ra2 = cat2_data[ra_param2]
    dec2 = cat2_data[dec_param2]
    #print (ra2,dec2)
    #print (ra_param2,dec_param2)

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

def crossmatch_left_wing(cat1,cat2,ra_param1,dec_param1,ra_param2,dec_param2,max_sep_arcsec,filter_name2,output_cat):

    cat = fits.open(cat1, ignore_missing_end=True)
    cat1_data = cat[1].data
    ra1 = cat1_data[ra_param1]
    dec1 = cat1_data[dec_param1]

    cat = fits.open(cat2, ignore_missing_end=True)
    cat2_data = cat[1].data

    if (ra_param2 == ra_param1) and (dec_param2 == dec_param1):
        cat[1].columns[ra_param2].name = ra_param2+'_'
        cat[1].columns[dec_param2].name = dec_param2+'_'
        ra_param2 = ra_param2+'_'
        dec_param2 = dec_param2+'_'
        cat2 = temp_dir+'temp0.fits'
        cat.writeto(cat2 ,overwrite=True)
        cat = fits.open(cat2, ignore_missing_end=True)
        cat2_data = cat[1].data
        
    ra2 = cat2_data[ra_param2]
    dec2 = cat2_data[dec_param2]
    #print (ra2,dec2)
    #print (ra_param2,dec_param2)

    c1 = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)
    c2 = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)
    join_func = join_skycoord(max_sep_arcsec * u.arcsec)
    j = join_func(c1, c2)

    os.system('cp '+cat1+' '+temp_dir+'temp1.fits')
    os.system('cp '+cat2+' '+temp_dir+'temp2.fits')

    #print (len(ra1))
    #print (len(j[0]))
    #print (j[0])

    expand_fits_table(temp_dir+'temp1.fits','JOIN_ID_'+filter_name2,j[0])
    expand_fits_table(temp_dir+'temp2.fits','JOIN_ID_'+filter_name2,j[1])

    t1 = Table.read(temp_dir+'temp1.fits',format='fits')
    t2 = Table.read(temp_dir+'temp2.fits',format='fits')
    t12 = table.join(t1, t2, keys='JOIN_ID_'+filter_name2, join_type='left')
    print (t12)
    t12.write(output_cat,overwrite=True)
    