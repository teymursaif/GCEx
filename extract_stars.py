
import sys, os
import pyfits as fits
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import colors
import pyfits

def get_header(file,keyword=None):
    '''
    Reads the fits file and outputs the header dictionary.
    OR
    If a keyword is given, returns value of the keyword.
    '''
    fits = get_fits(file)
    if keyword:
        return fits[0].header[keyword]
    else:
        return fits[0].header


def get_fits(filefile):
    '''
    Reads the input fits-file and returns the hdu table.
    '''
    hdulist = pyfits.open(filefile)
    return hdulist


def stars (filtername, field, path) :
    
    print (field)
    print (filtername)
    if filtername in ['k','j'] :
	frame = path+filtername+field+'.fits'
    if filtername in ['u','g','r','i'] :
        #pathg = '/data/users/saifollahi/vst/g_data/'
	frame = path+filtername+field+'_cropped.fits'
    os.system('rm detection.fits')
    os.system('sex '+frame + ' -c default-detection.sex' )
   

    if filtername == 'k' :
        frame = path+filtername+field+'.fits'
        os.system('sex detection.fits,'+frame+' -c default-nir.sex -CATALOG_NAME '+filtername+field+'.sex_stars_cat.fits '+ \
        '-DETECT_MINAREA 10 -DETECT_THRESH 5 -ANALYSIS_THRESH 5 ' + \
	'-FILTER_NAME default.conv ' + \
        '-PHOT_APERTURES ' + str(sys.argv[1]) + ',15 ' + \
        '-PIXEL_SCALE 0.334 ' + \
        '-CHECKIMAGE_TYPE  NONE ' + \
        '-BACK_SIZE 30 ' + \
        '-BACK_FILTERSIZE 3 ')# + \
        #'-CHECKIMAGE_NAME '+filtername+field+'.check.fits')
        fwhm = 1.
        fwhm_upper = 20
        fwhm_lower = 1.0
    
    if filtername == 'j' :
        frame = path+filtername+field+'.fits'

        zp = get_header(frame,keyword='PHOTZP')
        print (zp)

        #print ('sex detection.fits,'+frame+' -c default-nir.sex -CATALOG_NAME '+filtername+field+'.sex_stars_cat.fits '+ \
        #'-DETECT_MINAREA 5 -DETECT_THRESH 1.5 -ANALYSIS_THRESH 1.5 ' + \
        #'-FILTER_NAME default.conv ' + \
        #'-PHOT_APERTURES ' + str(sys.argv[1]) + ',30 ' + \
        #'-PIXEL_SCALE 0.334 ' )#+ \
        #'-MAG_ZEROPOINT '+ str(zp) + ' ' + \
        #'-BACK_SIZE 30 ' + \
        #'-BACK_FILTERSIZE 3 ')# + \
        #'-CHECKIMAGE_NAME '+filtername+field+'.check.fits')


        os.system('sex detection.fits,'+frame+' -c default-nir.sex -CATALOG_NAME '+filtername+field+'.sex_stars_cat.fits '+ \
        '-DETECT_MINAREA 10 -DETECT_THRESH 5 -ANALYSIS_THRESH 5 ' + \
	'-FILTER_NAME default.conv ' + \
        '-PHOT_APERTURES ' + str(sys.argv[1]) + ',30 ' + \
        '-PIXEL_SCALE 0.334 ' + \
        '-CHECKIMAGE_TYPE  NONE ' +
        '-MAG_ZEROPOINT '+ str(zp) + ' ' + \
        '-BACK_SIZE 30 ' + \
        '-BACK_FILTERSIZE 3 ')# + \
        #'-CHECKIMAGE_NAME '+filtername+field+'.check.fits')
        fwhm = 2.
        fwhm_upper = 20
        fwhm_lower = 1.0
    
    if filtername == 'u' :
        frame = path+filtername+field+'_cropped.fits'
        os.system('sex detection.fits,'+frame+' -c default.sex -CATALOG_NAME '+filtername+field+'.sex_stars_cat.fits '+ \
        '-DETECT_MINAREA 5  -DETECT_THRESH 1.5 -ANALYSIS_THRESH 1.5 ' + \
	'-FILTER_NAME default.conv ' + \
        '-PHOT_APERTURES ' + str(sys.argv[2]) + ',60 ' + \
        '-PIXEL_SCALE 0.21 ' + \
        '-CHECKIMAGE_TYPE  NONE ' + \
        '-BACK_SIZE 50 ' + \
        '-BACK_FILTERSIZE 3 ')# + \
        #'-CHECKIMAGE_NAME '+filtername+field+'.check.fits')
        fwhm = 6
        fwhm_upper = 25
        fwhm_lower = 1.0

    if filtername == 'g' or filtername == 'r' or filtername == 'i' :
        frame = path+filtername+field+'_cropped.fits'
        os.system('sex detection.fits,'+frame+' -c default.sex -CATALOG_NAME '+filtername+field+'.sex_stars_cat.fits '+ \
        '-DETECT_MINAREA 8  -DETECT_THRESH 3 -ANALYSIS_THRESH 3 ' + \
        '-PHOT_APERTURES ' + str(sys.argv[3]) + ',50 ' + \
        '-PIXEL_SCALE 0.21 ' + \
        '-CHECKIMAGE_TYPE  NONE ' +
        '-BACK_SIZE  50 ' +
        '-BACK_FILTERSIZE 3 ')# +
        #'-CHECKIMAGE_NAME '+filtername+field+'.check.fits')
        fwhm = 6
        fwhm_upper = 20
        fwhm_lower = 1.0

    os.system('rm '+filtername+field+'.sex_stars_cat.filtered.fits')
    with fits.open(filtername+field+'.sex_stars_cat.fits') as hdul:
        data = hdul[2].data
        mask = ( (data['FLAGS'] < 4) & \
        (data['FWHM_IMAGE'] > fwhm_lower) & \
        #(data['MAG_PETRO'] > 5) & \
        #(data['MAG_PETRO'] < 25) & \
        #(data['MAG_APER'] > 5) & \
        #(data['MAG_APER'] < 25) & \
        (data['MAG_AUTO'] > 10) & \
        (data['MAG_AUTO'] < 30) & \
        (data['FWHM_IMAGE'] < fwhm_upper) & \
        (data['FWHM_IMAGE'] > fwhm_lower) )
        newdata = data[mask]
        mg = newdata['MAG_APER']
        mg = np.array(mg)
        mg1 = mg[:,0]
        mg2 = mg[:,1]
        emg = newdata['MAGERR_APER']
        emg = np.array(emg)
        emg1 = emg[:,0]
        emg2 = emg[:,1]
	newdata['FLUX_BEST'] = mg1
        newdata['FLUXERR_BEST'] = mg2
        newdata['MAG_BEST'] = emg1
        newdata['MAGERR_BEST'] = emg2
        newdata['NUMBER'] = field   
        hdu = fits.BinTableHDU(data=newdata)
        #hdu.writeto(filtername+field+'.sex_stars_cat.filtered.fits')
        hdu.writeto(filtername+field+'.sex_stars_cat.filtered.fits')



def append(filtername, field) :
    fits_table_filename1 = filtername+'.stars.fits'
    fits_table_filename2 = filtername+field+'.sex_stars_cat.filtered.fits'
    print fits_table_filename1
    print fits_table_filename2
    with fits.open(fits_table_filename1) as hdul1:
        with fits.open(fits_table_filename2) as hdul2:
            nrows1 = hdul1[1].data.shape[0]
            nrows2 = hdul2[1].data.shape[0]
            nrows = nrows1 + nrows2
            hdu = fits.BinTableHDU.from_columns(hdul1[1].columns, nrows=nrows)
            for colname in hdul1[1].columns.names:
                hdu.data[colname][nrows1:] = hdul2[1].data[colname]
    os.system('rm '+filtername+'.stars.fits')
    hdu.writeto(filtername+'.stars.fits')


################################################
#[5,6,7,10,11,12,15,16,17]
f = [5,6,11]
#field_test = [5,6,7,10,11,12,15,16,17]
#for i in [5,6,7,10,11,12,15,16,17] :
field_u = [1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,31]
field_g = [1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,25,26,27,28,31,33]
field_r = [1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,25,26,27,28,31,33]
field_i = [1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,25,26,27,28,31]
#field_j = ['ADP.2016-10-06T13:12:25.824','ADP.2016-10-06T13:12:26.762','ADP.2016-10-06T13:12:28.466','ADP.2016-10-06T13:12:30.221', \
#'ADP.2016-10-06T13:12:30.550','ADP.2016-10-06T13:39:51.102','ADP.2016-10-06T13:12:26.368','ADP.2016-10-06T13:12:26.884', \
#'ADP.2016-10-06T13:12:28.675','ADP.2016-10-06T13:12:30.283','ADP.2016-10-06T13:39:49.409','ADP.2016-10-06T13:39:51.429', \
#'ADP.2016-10-06T13:12:26.638','ADP.2016-10-06T13:12:27.381','ADP.2016-10-06T13:12:29.992','ADP.2016-10-06T13:12:30.377', \
#'ADP.2016-10-06T13:39:50.894','ADP.2016-10-06T13:39:51.770']
field_j = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
field_k = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,25,26]
#field_j = [11]

for i in field_j :
    filtername = 'j'
    path='/data/users/saifollahi/VHS/J/science/'
    j = stars(filtername, str(i), path)

for i in field_k :
    filtername = 'k'
    path='/data/users/saifollahi/vircam/k_data/'
    k = stars(filtername, str(i), path)

for i in field_u :
    filtername = 'u'
    path='/data/users/saifollahi/vst/u_data/'
    u = stars(filtername, str(i), path)

for i in field_i :
    filtername = 'i'
    path='/data/users/saifollahi/vst/i_data/'
    i = stars(filtername, str(i), path)

for i in field_g :
    filtername = 'g'
    path='/data/users/saifollahi/vst/g_data/'
    g = stars(filtername, str(i), path)

for i in field_r :
    filtername = 'r'
    path='/data/users/saifollahi/vst/r_data/'
    r = stars(filtername, str(i), path)

for filtername in ['u','i','g','r','k','j'] :
#for filtername in ['k','j'] :
    if filtername == 'u' :
        field_f = field_u
    if filtername == 'g' :
        field_f = field_g
    if filtername == 'r' :
        field_f = field_r
    if filtername == 'i' :
        field_f = field_i
    if filtername == 'k' :
        field_f = field_k
    if filtername == 'j' :
        field_f = field_j

    os.system(('rm '+filtername+'.stars.fits'))
    hdu = fits.open(filtername+str(field_f[0])+'.sex_stars_cat.filtered.fits')
    hdu.writeto(filtername+'.stars.fits')

    for i in field_f :
        if i != int(field_f[0]) :
            print i
            append(filtername, str(i))

def get_header(file,keyword=None):
    '''
    Reads the fits file and outputs the header dictionary.
    OR
    If a keyword is given, returns value of the keyword.
    '''
    fits = get_fits(file)
    if keyword:
        return fits[0].header[keyword]
    else:
        return fits[0].header


def get_fits(filefile):
    '''
    Reads the input fits-file and returns the hdu table.
    '''
    hdulist = pyfits.open(filefile)
    return hdulist


ttypes = list()
for i in range (0,53) :
      i = i+1
      ttypes.append('TTYPE'+str(i))

for filtername in ['u','g','r','i'] :
#for filtername in ['u'] :
    cat_name = filtername+'.stars.fits'
    out_name = filtername+'+'+str(sys.argv[2])+'.stars.fits'
    cat = get_fits(cat_name)
    for ttype in ttypes :
        if cat[1].header[ttype] == 'ALPHA_SKY' :
            cat[1].header[ttype] = 'RA'
        if cat[1].header[ttype] == 'DELTA_SKY' :
            cat[1].header[ttype] = 'DEC'
        if cat[1].header[ttype] == 'FLUX_BEST' :
            cat[1].header[ttype] = 'MAG_APER_1'
        if cat[1].header[ttype] == 'FLUXERR_BEST' :
            cat[1].header[ttype] = 'MAG_APER_2'
        if cat[1].header[ttype] == 'MAG_BEST' :
            cat[1].header[ttype] = 'MAGERR_APER_1'
        if cat[1].header[ttype] == 'MAGERR_BEST' :
            cat[1].header[ttype] = 'MAGERR_APER_2'
        if cat[1].header[ttype] == 'NUMBER' :
            cat[1].header[ttype] = 'FIELD'

 
        cat[1].header[ttype] = cat[1].header[ttype]+'_'+filtername+'_'+str(sys.argv[2])
        labels = cat[1].header
        #print labels
    cat.writeto(out_name)
"""
for filtername in ['g'] :
    cat_name = filtername+'.stars.fits'
    out_name = filtername+'+'+str(sys.argv[3])+'.stars.fits'
    cat = get_fits(cat_name)
    for ttype in ttypes :
        if cat[1].header[ttype] == 'ALPHA_SKY' :
            cat[1].header[ttype] = 'RA'
        if cat[1].header[ttype] == 'DELTA_SKY' :
            cat[1].header[ttype] = 'DEC'
        cat[1].header[ttype] = cat[1].header[ttype]+'_'+filtername+'_'+str(sys.argv[2])
        labels = cat[1].header
        #print labels
    cat.writeto(out_name)

for filtername in ['r'] :
    cat_name = filtername+'.stars.fits'
    out_name = filtername+'+'+str(sys.argv[3])+'.stars.fits'
    cat = get_fits(cat_name)
    for ttype in ttypes :
        if cat[1].header[ttype] == 'ALPHA_SKY' :
            cat[1].header[ttype] = 'RA'
        if cat[1].header[ttype] == 'DELTA_SKY' :
            cat[1].header[ttype] = 'DEC'
        cat[1].header[ttype] = cat[1].header[ttype]+'_'+filtername+'_'+str(sys.argv[2])
        labels = cat[1].header
        #print labels
    cat.writeto(out_name)

for filtername in ['i'] :
    cat_name = filtername+'.stars.fits'
    out_name = filtername+'+'+str(sys.argv[3])+'.stars.fits'
    cat = get_fits(cat_name)
    for ttype in ttypes :
        if cat[1].header[ttype] == 'ALPHA_SKY' :
            cat[1].header[ttype] = 'RA'
        if cat[1].header[ttype] == 'DELTA_SKY' :
          cat[1].header[ttype] = 'DEC'
        cat[1].header[ttype] = cat[1].header[ttype]+'_'+filtername+'_'+str(sys.argv[3])
        labels = cat[1].header
        #print labels
    cat.writeto(out_name)
"""

for filtername in ['k'] :
#for filtername in ['k'] :
    cat_name = filtername+'.stars.fits'
    out_name = filtername+'+'+str(sys.argv[1])+'.stars.fits'
    cat = get_fits(cat_name)
    for ttype in ttypes :
        if cat[1].header[ttype] == 'ALPHA_SKY' :
            cat[1].header[ttype] = 'RA'
        if cat[1].header[ttype] == 'DELTA_SKY' :
            cat[1].header[ttype] = 'DEC'
        
        if cat[1].header[ttype] == 'FLUX_BEST' :
            cat[1].header[ttype] = 'MAG_APER_1'
        if cat[1].header[ttype] == 'FLUXERR_BEST' :
            cat[1].header[ttype] = 'MAG_APER_2'
        if cat[1].header[ttype] == 'MAG_BEST' :
            cat[1].header[ttype] = 'MAGERR_APER_1'
        if cat[1].header[ttype] == 'MAGERR_BEST' :
            cat[1].header[ttype] = 'MAGERR_APER_2'
        if cat[1].header[ttype] == 'NUMBER' :
            cat[1].header[ttype] = 'FIELD'

        cat[1].header[ttype] = cat[1].header[ttype]+'_'+filtername+'_'+str(sys.argv[1])
        #print labels
    cat.writeto(out_name)


for filtername in ['j'] :
#for filtername in ['k'] :
    cat_name = filtername+'.stars.fits'
    out_name = filtername+'+'+str(sys.argv[1])+'.stars.fits'
    cat = get_fits(cat_name)
    for ttype in ttypes :
        if cat[1].header[ttype] == 'ALPHA_SKY' :
            cat[1].header[ttype] = 'RA'
        if cat[1].header[ttype] == 'DELTA_SKY' :
            cat[1].header[ttype] = 'DEC'
        
        if cat[1].header[ttype] == 'FLUX_BEST' :
            cat[1].header[ttype] = 'MAG_APER_1'
        if cat[1].header[ttype] == 'FLUXERR_BEST' :
            cat[1].header[ttype] = 'MAG_APER_2'
        if cat[1].header[ttype] == 'MAG_BEST' :
            cat[1].header[ttype] = 'MAGERR_APER_1'
        if cat[1].header[ttype] == 'MAGERR_BEST' :
            cat[1].header[ttype] = 'MAGERR_APER_2'
        if cat[1].header[ttype] == 'NUMBER' :
            cat[1].header[ttype] = 'FIELD'

        cat[1].header[ttype] = cat[1].header[ttype]+'_'+filtername+'_'+str(sys.argv[1])
        #print labels
    cat.writeto(out_name)


















