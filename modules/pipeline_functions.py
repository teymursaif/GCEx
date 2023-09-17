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
from fitsio import FITS
from modules.initialize import *
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
    print ('- Copying data to the input-data directory: '+(main_data_dir))
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]

    for fn in filters:
        main_data_path = main_data_dir+gal_name+'_'+fn+'.fits'
        data_path = data_dir+gal_name+'_'+fn+'.fits'
        main_weight_path = main_data_dir+gal_name+'_'+fn+'.weight.fits'
        weight_path = main_data_dir+gal_name+'_'+fn+'.weight.fits'

        if os.path.exists(main_data_path):
            crop_frame(main_data_dir+gal_name+'_'+fn+'.fits',gal_name,PRIMARY_FRAME_SIZE[fn]/2,fn,ra,dec,format='.fits')
        else:
            print ('*** File does no exists: '+str(data_path))

        if os.path.exists(main_weight_path):
            crop_frame(main_data_dir+gal_name+'_'+fn+'.weight.fits',gal_name,PRIMARY_FRAME_SIZE[fn]/2,fn,ra,dec,format='.weight.fits')
        else:
            print ('*** Weight does no exists: '+str(data_path)+', crating a weight-map with values of 1')
            make_weight_map(main_data_path,main_weight_path)
            crop_frame(main_data_dir+gal_name+'_'+fn+'.weight.fits',gal_name,PRIMARY_FRAME_SIZE[fn]/2,fn,ra,dec,format='.weight.fits')

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
    print ('- Extracting pixel-size and zero-point values from the data')
    gal_name, ra, dec, distance, filters, comments = gal_params[gal_id]
    for fn in filters:
        pixel_scale = get_pixel_scale(main_data_dir+gal_name+'_'+fn+'.fits')
        PIXEL_SCALES[fn] = pixel_scale
        gain = get_gain(main_data_dir+gal_name+'_'+fn+'.fits')
        GAIN[fn] = gain
        zp = get_zp_AB(main_data_dir+gal_name+'_'+fn+'.fits')
        exptime = get_exptime(main_data_dir+gal_name+'_'+fn+'.fits')
        ZPS[fn] = zp
        PRIMARY_FRAME_SIZE[fn] = int(PRIMARY_FRAME_SIZE_ARCSEC/pixel_scale+0.5)
        FRAME_SIZE[fn] = int(FRAME_SIZE_ARCSEC/pixel_scale+0.5)
        GAL_FRAME_SIZE[fn] = int(GAL_FRAME_SIZE_ARCSEC/pixel_scale+0.5)
        EXPTIME[fn]=exptime
    print ('- Frame exposure times are:'+str(EXPTIME))
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
        command = swarp_executable+' '+temp_dir+'temp.fits -c '+external_dir+'default.swarp -IMAGEOUT_NAME '+output+' -WEIGHTOUT_NAME '+output_weight+\
            ' -WEIGHT_TYPE MAP_WEIGHT '+'-WEIGHT_IMAGE '+fitsfile_weight+\
            ' -IMAGE_SIZE '+str(radius_pix)+','+str(radius_pix)+' -PIXEL_SCALE '+str(pixel_size)+\
            ' -CENTER_TYPE MANUAL -CENTER '+str(ra)+','+str(dec)+' -SUBTRACT_BACK N -VERBOSE_TYPE QUIET'

    else :
        command = swarp_executable+' '+temp_dir+'temp.fits -c '+external_dir+'default.swarp -IMAGEOUT_NAME '+output+' -WEIGHTOUT_NAME '+output_weight+\
            ' -WEIGHT_TYPE MAP_WEIGHT '+'-WEIGHT_IMAGE '+fitsfile_weight+\
            ' -IMAGE_SIZE '+str(radius_pix)+','+str(radius_pix)+' -PIXELSCALE_TYPE MANUAL -PIXEL_SCALE '+str(pixel_size)+\
            ' -RESAMPLE Y -CENTER_TYPE MANUAL -CENTER '+str(ra)+','+str(dec)+' -SUBTRACT_BACK N -VERBOSE_TYPE QUIET'

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


def convertNumpyToFitsBinTable(recarray):
    #Function is developed by some nice people on the internet
    return fits.FITS_rec.from_columns(fits.ColDefs(recarray))


def saveFitsBinTable(bintable, filename):
    #Function is developed by some nice people on the internet
    primaryHDU = fits.PrimaryHDU()
    binTableHDU = fits.BinTableHDU(bintable)
    fits.HDUList([primaryHDU, binTableHDU]).writeto(filename)


############################################################

get_data_info(gal_id)
