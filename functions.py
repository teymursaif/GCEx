import numpy as np
import pyfits
import astropy
import astropy.io as fits
import os, sys
from astropy.wcs import WCS
from astropy import wcs
import math
import random
from pylab import *
from os import path
from scipy.optimize import curve_fit
from pandas.plotting import scatter_matrix
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde
from astropy.io import ascii
import matplotlib.pyplot as plt
import fitsio
from fitsio import FITS,FITSHDR
from astropy.stats import sigma_clip
import scipy
from lacosmic import lacosmic
from matplotlib.ticker import StrMethodFormatter
import matplotlib.ticker as ticker
import csv
from matplotlib.patches import Ellipse
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import Normalize
import matplotlib.colors as colors
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter


def get_fits(filefile):
    '''
    Reads the input fits-file and returns the hdu table.
    '''
    hdulist = pyfits.open(filefile)
    return hdulist

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
    

def make_directory(directory_path,directory_name) :
    if path.exists(directory_path+directory_name) :
        do_nothing = 1
    else :
        os.system('mkdir '+directory_path+directory_name)

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

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

def pointInEllipse_pixel(test, center, a, b, theta) :
	dx = test[0] - center[0]
	dy = test[1] - center[1]
	return ((dx*math.cos(theta)+dy*math.sin(theta))**2)/(a**2) + ((dx*math.sin(theta)-dy*math.cos(theta))**2)/(b**2) <= 1

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def cut(fitsfile, ra, dec, radius_pix, objectname='none', filtername='none',  back=0, overwrite=False, \
    blur=0, label=''):
    '''
    Cuts the images from the coadds.
    '''
    hdu = get_fits(fitsfile)
    hdu2 = get_fits(fitsfile)
    w=WCS(fitsfile)

    radius_pix = int(radius_pix * 1)
    x_center,y_center = w.all_world2pix(ra, dec,0)
    #print (x_center,y_center,radius_pix)
    llx = int(x_center - radius_pix)
    lly = int(y_center - radius_pix)
    urx = int(x_center + radius_pix)
    ury = int(y_center + radius_pix)
    dimx,dimy= len(hdu[0].data[0]),len(hdu[0].data)

    if llx<0:llx=0
    if lly<0:lly=0
    if urx>=dimx:urx=dimx-1
    if ury>=dimy:ury=dimy-1
    #print ('+++ Cropping area',llx,lly,'(x0,y0)',urx,ury,'(x1,y1)')
    #if (urx - llx != ury - lly) :
    #    return 0
    #print (lly,ury,llx,urx)
    object = hdu[0].data[lly:ury,llx:urx]*(1.0)
    #object2 = np.log10(hdu[0].data[lly:ury,llx:urx]*(1.0e15))  #scaling
    template = hdu
    if blur > 0 :
        #object = gaussian_filter(object,sigma=blur)
        object = ndimage.median_filter(object, size=blur)
    #template2 = hdu2
    template[0].header['NAXIS1'] = urx - llx
    template[0].header['NAXIS2'] = ury - lly
    template[0].header['EXPTIME'] = 1.0
    #template[0].header['GAIN'] = 1.0
    #print (urx - llx,ury - lly)
    template[0].header['CRPIX1'] = hdu[0].header['CRPIX1'] -llx
    template[0].header['CRPIX2'] = hdu[0].header['CRPIX2'] -lly

    object = object-back
    where_are_NaNs = isnan(object)
    object[where_are_NaNs] = 99

    template[0].data = object
    #template2[0].data = object2


    template.writeto(objectname+'_'+filtername+'_cropped'+str(label)+'.fits', clobber=True)
    #print ('MINIMUM')
    #print (np.nanmin(object))


    #template2.writeto(objectname+'_'+filtername+'_scaled'+'_'+filtername_+'.fits', clobber = overwrite)
    #print ('Saved postage stamp to '+objectname+'_'+filtername+'_cropped'+str(label)+'.fits')
    output_frame = objectname+'_'+filtername+'_cropped.fits'
    # no jpg conversion

    #print 'DONE!'
    del template,hdu
    #del template2,hdu2

    w=WCS(objectname+'_'+filtername+'_cropped'+str(label)+'.fits')
    x_gal_center,y_gal_center = w.all_world2pix(ra, dec,0)

    return output_frame, x_gal_center, y_gal_center

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def get_total_number_of_objects(table) :
    table_main = get_fits(table)
    table_data = table_main[1].data
    total_number_of_objects = len(table_data)
    print ('total numer of objects in table '+table+' is : ' + str(total_number_of_objects))
    return total_number_of_objects

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def clean_fits_table(table, cleaning_params, output_table=None) :
    print ('cleaning table : '+table)
    table_main = get_fits(table)
    table_data = table_main[1].data
    temp = table_data
    for key in cleaning_params :
        #print (len(temp))
        value1 = (cleaning_params[key])[0]
        value2 = (cleaning_params[key])[1]
        mask = ((temp[key] >= value1) & (temp[key] <= value2))
        temp = temp[mask]
        #print (len(temp))  
    table_main[1].data = temp
    if output_table == None :
        os.system('rm '+table) 
        table_main.writeto(table) 
    else :
        os.system('rm '+output_table) 
        table_main.writeto(output_table) 

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def convert_fits_to_csv(dataset,output) :
    main = get_fits(dataset)
    print ('teymoor')
    X = main[1].data
    print ('converting fits to csv is started')
    ascii.write(X, output, format='csv', fast_writer=False, overwrite=True)
    print ('converting fits to csv is finished')

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def convert_csv_to_fits(dataset,output) :
    os.system('rm '+output)
    #text_file = ascii.read(dataset)
    #text_file.write(output)
    os.system('python csv-to-fits.py '+dataset+' '+output)

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def expand_fits_table(table,new_param,new_param_values) :
    fits = FITS(table,'rw')
    fits[-1].insert_column(name = new_param, data = new_param_values) 
    fits.close()

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def attach_fits_tables(tables,output_table) :
    os.system('rm '+output_table)
    out_table = pyfits.open(tables[0])
    out_table.writeto(output_table)

    for i in range(len(tables)-1) :
        #print (i, len(tables))
        with pyfits.open(output_table) as hdul1:
            with pyfits.open(tables[i+1]) as hdul2:
                nrows1 = hdul1[1].data.shape[0]
                nrows2 = hdul2[1].data.shape[0]
                nrows = nrows1 + nrows2
                hdu = pyfits.BinTableHDU.from_columns(hdul1[1].columns, nrows=nrows)
                for colname in hdul1[1].columns.names:
                    hdu.data[colname][nrows1:] = hdul2[1].data[colname]
        
        os.system('rm '+output_table)
        hdu.writeto(output_table,clobber=True)

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def attach_sex_tables(tables,output_table) :
    os.system('rm '+output_table)
    out_table = pyfits.open(tables[0])
    out_table.writeto(output_table)

    for i in range(len(tables)-1) :
        #print (i, len(tables))
        with pyfits.open(output_table) as hdul1:
            with pyfits.open(tables[i+1]) as hdul2:
                nrows1 = hdul1[1].data.shape[0]
                nrows2 = hdul2[1].data.shape[0]
                nrows = nrows1 + nrows2
                hdu1 = pyfits.BinTableHDU.from_columns(hdul1[1].columns, nrows=nrows)
                for colname in hdul1[1].columns.names:
                    hdu1.data[colname][nrows1:] = hdul2[1].data[colname]

                nrows1 = hdul1[2].data.shape[0]
                nrows2 = hdul2[2].data.shape[0]
                nrows = nrows1 + nrows2
                hdu2 = pyfits.BinTableHDU.from_columns(hdul1[2].columns, nrows=nrows)
                for colname in hdul1[2].columns.names:
                    hdu2.data[colname][nrows1:] = hdul2[2].data[colname]

        
        os.system('rm '+output_table)
        out_table[1] = hdu1
        out_table[2] = hdu2
        out_table.writeto(output_table)

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def get_column_number_for_param(table,params) :
    table_main = get_fits(table)
    table_cols = table_main[1].columns.names
    column_number = -1
    column_numbers = list()
    for param in params :
        for i in range(len(table_cols)) :
            key = table_cols[i]
            #print (table_head.keys()))
            if key == param :
                column_number = i
                column_numbers.append(i)

    return column_numbers

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def shorten_table(table,params_column_numbers,add_label,preselection=0) :
    table_main = get_fits(table)
    table_data = table_main[1].data
    table_cols = table_main[1].columns.names
    if preselection == 1 :
        mask = table_data['preselected'] == 1 
        table_data = table_data[mask]
    for i in range(len(params_column_numbers)) :
        if i == 0 :
            temp = table_data[table_cols[params_column_numbers[i]]]
            col = pyfits.ColDefs([pyfits.Column(name=table_cols[params_column_numbers[i]], format='D', array=np.array(temp))])
            columns = col
        elif i > 0 :
            temp = table_data[table_cols[params_column_numbers[i]]]
            col = pyfits.ColDefs([pyfits.Column(name=table_cols[params_column_numbers[i]], format='D', array=np.array(temp))])
            columns = columns + col
    hdu = pyfits.BinTableHDU.from_columns(columns)
    os.system('rm '+add_label+table)
    hdu.writeto(add_label+table)

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def resample(main_data,obj_name,filter_name, px=0.05,label='') :
    output = obj_name+'_'+filter_name+label+'_resampled.fits'
    command = 'SWarp '+main_data+' -c default.swarp -IMAGEOUT_NAME '+output+' -PIXEL_SCALE '+str(px)
    os.system(command)
    return output

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def resample_weight(main_data,weight_data,obj_name,filter_name, px=0.05,label='') :
    X = get_header(main_data,keyword='NAXIS1')
    Y = get_header(main_data,keyword='NAXIS2')
    output = obj_name+'_'+filter_name+label+'.weight_resampled.fits'
    command = 'SWarp '+weight_data+' -c default.swarp -IMAGEOUT_NAME '+output+' -IMAGE_SIZE '+\
        str(X)+','+str(Y)+' -PIXEL_SCALE '+str(px)
    os.system(command)
    return output

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def make_psf(udg,filter,data_dir,cats_dir,psfs_dir,sex_dir,ra_param,dec_param,x_param,y_param,image_size,star_cat):
    print ('+++ Making PSF for galaxy '+udg+' in '+filter)
    #os.system('rm '+psfs_dir+udg+'_'+filter+'_PSF.fits')
    table_main = get_fits(cats_dir+star_cat)
    table_data = table_main[1].data
    RA = table_data[ra_param]
    DEC = table_data[dec_param]
    X = table_data[x_param]
    Y = table_data[y_param]
    N = len(RA)
    psf_frames = list()
    #os.system('rm ' + psfs_dir +'*'+'psf*')
    print ('+ Number of stars : '+str(N))
    psfs = list()
    for i in range(len(RA)) :
        ra = RA[i]
        dec = DEC[i]
        data_cropped, x, y = cut(data_dir+udg+'_'+filter+'.fits', ra, dec, image_size, udg, filter, overwrite=True)
        os.system('mv '+data_cropped+' '+psfs_dir+udg+'_'+filter+'_psf'+str(i)+'.fits')

        psf = get_fits(psfs_dir+udg+'_'+filter+'_psf'+str(i)+'.fits')
        psf_data = psf[0].data
        psf_data_back = np.nanmedian(sigma_clip(psf_data,sigma=3))
        psf[0].data = psf_data - psf_data_back
        psf.writeto(psfs_dir+udg+'_'+filter+'_psf'+str(i)+'.fits', clobber=True)

        psf_data = scipy.ndimage.zoom(psf_data, 10, order=3)
        psf_data = np.array(psf_data)
        
        x_center = image_size*10
        y_center = image_size*10
        x_max, y_max = np.unravel_index(psf_data.argmax(), psf_data.shape)
        #print (x_max,y_max)
        dx = int(x_max-x_center+5)
        dy = int(y_max-y_center+5)
        #print (dx,dy)
        psf_data = np.roll(psf_data, -1*dx, axis=0)
        psf_data = np.roll(psf_data, -1*dy, axis=1)

        psf[0].data = psf_data
        psf.writeto(psfs_dir+udg+'_'+filter+'_psf'+str(i)+'_resampled.fits', clobber=True)
        #psf_data_min = np.sort(psf_data)[:int(len(psf_data)/2)]
        psf_data_back = np.nanmedian(sigma_clip(psf_data,sigma=3))
        #print (psf_data_back)
        psf_data = psf_data - psf_data_back
        psf_data_sum = np.nansum(psf_data)
        psf_data = psf_data / psf_data_sum
        if np.shape(psf_data) == (image_size*2*10,image_size*2*10) :
            psfs.append(psf_data)
            psf_frames.append(psfs_dir+udg+'_'+filter+'_psf'+str(i)+'.fits')
        #print (psf_data)
        #print (np.shape(psf_data))
        #print ('-------------')

    psfs = np.array(psfs)
    psf_median = np.median(psfs,axis=0)
    psf_median[psf_median<0] = 0
    psf_median_sum = np.nansum(psf_median)
    psf_median = psf_median / psf_median_sum
    PSF = get_fits(psfs_dir+udg+'_'+filter+'_psf'+str(0)+'.fits')
    #print (psf_median)
    PSF[0].data = psf_median
    PSF.writeto(psfs_dir+udg+'_'+filter+'_PSF_resampled.fits', clobber=True)
    psf_median = rebin(psf_median, (int(image_size*2), int(image_size*2)))
    PSF[0].data = psf_median
    PSF.writeto(psfs_dir+udg+'_'+filter+'_PSF.fits', clobber=True)

    return psf_frames

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).sum(-1).sum(1)

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def clean_cosmic_rays(dirty_frame,clean_frame,gain):
    fits_dirty = get_fits(dirty_frame)
    fits_dirty_data = fits_dirty[0].data 
    cleaned_image, cosmic_mask = lacosmic(fits_dirty_data,3.0,0.5,0.2,effective_gain=gain,readnoise=100,maxiter=5)
    fits_dirty_data = fits_dirty[0].data
    fits_dirty[0].data = cleaned_image
    fits_dirty.writeto(clean_frame,clobber=True)

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def add_psfs_to_frame(frame,weight,udg,filter,data_dir,psfs_dir,sex_dir,cat_dir,art_dir,n_stars,mags,zp,image_size,gain) :
    image_size = int(image_size)
    half_image_size = int(image_size/2)
    for mag in mags :
        print ('+++ Making artificial stars of magnitude '+str(mag))
        psf_file = psfs_dir+udg+'_'+filter+'_PSF_resampled.fits'
        psf_frame = pyfits.open(psf_file)
        psf_data = psf_frame[0].data

        where_are_NaNs = isnan(psf_data)
        psf_data[where_are_NaNs] = 0
        psf_data[psf_data<0] = 0
        psf_data[psf_data>10] = 0
        #print (np.sum(psf_data))

        main_frame = pyfits.open(frame)
        main_data = main_frame[0].data

        weight_frame = pyfits.open(weight)
        weight_data = main_frame[0].data

        temp = weight_data[weight_data>0]
        temp = np.sort(temp)
        weight_th = temp[int(0.01*len(temp))]

        X = get_header(frame,'NAXIS1')
        Y = get_header(frame,'NAXIS2')
        #main_data=np.zeros((Y,X))

        cat = open(art_dir+udg+'_'+filter+'_+artificial_stars_mag'+str(mag)+'.cat','w')
        cat.write('X,Y,RA,DEC\n')
        for i in range(n_stars) :
            mag_scale = zp-mag
            flux_scale = 2.5**mag_scale
            dx_rand = int(np.random.uniform(-10, 10, 1))
            dy_rand = int(np.random.uniform(-10, 10, 1))
            #print (dx_rand,dy_rand)
            psf_data_artificial = np.roll(psf_data, dx_rand, axis=0)
            psf_data_artificial = np.roll(psf_data_artificial, dy_rand, axis=1)
        
            psf_data_artificial = rebin(psf_data, (image_size, image_size))*flux_scale
            psf_data_noisy = np.random.poisson(psf_data_artificial*gain, psf_data_artificial.shape)
            psf_data_noisy = psf_data_noisy / float(gain)

            if i == 0 :
                psf_frame[0].data = psf_data_noisy
                #psf_frame.writeto(psfs_dir+udg+'_'+filter+'_PSF_noisy_example_mag'+str(mag)+'.fits',clobber=True)
                #print (psf_data_artificial[half_image_size,half_image_size], psf_data_noisy[half_image_size,half_image_size])

            weight_check = 0
            while weight_check == 0 :
                x = int(np.random.uniform(0, X, 1))
                y = int(np.random.uniform(0, Y, 1))
                weight_value = weight_data[y,x]
                if weight_value > weight_th and y > half_image_size and y < Y-half_image_size and x > half_image_size and x < X-half_image_size:
                    weight_check = 1
                else :
                    weight_check = 0
            
            #print (x,y)
            w=WCS(frame)
            ra, dec = w.all_pix2world(x, y,0)
            cat.write(str(x)+','+str(y)+','+str(ra)+','+str(dec)+'\n')
            for ii in range(image_size) :
                for jj in range(image_size) :
                    #print (Y, y-half_image_size+jj)
                    #print (X, x-half_image_size+ii)
                    main_data[y-half_image_size+jj,x-half_image_size+ii] = main_data[y-half_image_size+jj,x-half_image_size+ii] + psf_data_noisy[jj,ii]
            
        main_frame[0].data = main_data
        main_frame.writeto(art_dir+udg+'_'+filter+'_+artificial_stars_mag'+str(mag)+'.fits',clobber=True)
        cat.close()

    return art_dir+udg+'_'+filter+'_+artificial_stars_mag'+str(mag)+'.fits', art_dir+udg+'_'+filter+'_+artificial_stars_mag'+str(mag)+'.cat'


def cross_match(cat1,cat2,ra_param1,dec_param1,ra_param2,dec_param2,output_cat):

    ###

    cat1 = pyfits.open(cat1, ignore_missing_end=True)
    cat1_data = cat1[1].data
    ra1 = cat1_data[ra_param1]
    dec1 = cat1_data[dec_param1]
    cat2 = pyfits.open(cat2, ignore_missing_end=True)
    cat2_data = cat2[1].data
    ra2 = cat2_data[ra_param2]
    dec2 = cat2_data[dec_param2]

    c = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)
    catalog = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)
    matches = catalog[idx]
    #dra, ddec = c.spherical_offsets_to(matches)

    max_sep = 0.1 * u.arcsec
    sep_constraint = d2d < max_sep
    c_matches = c[sep_constraint]
    catalog_matches = catalog[idx[sep_constraint]]
    N1 = len(idx[sep_constraint])
    print (N1)
    ###
    
    c = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)
    catalog = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)
    matches = catalog[idx]
    #dra, ddec = c.spherical_offsets_to(matches)

    max_sep = 0.1 * u.arcsec
    sep_constraint = d2d < max_sep
    c_matches = c[sep_constraint]
    catalog_matches = catalog[idx[sep_constraint]]
    #print (idx[sep_constraint])
    N2 = len(idx[sep_constraint])

    idxs = idx[sep_constraint]
    #print (idxs)
    idxs = np.array(idxs)
    #idxs = np.sort(idxs)
    
    cat = list()
    cat = cat2_data[idxs][:]
    
    cat2[1].data = cat
    cat2.writeto(output_cat,clobber=True)

    
    print (N1,N2)
    return N1,N2
    

def select_point_sources(main_data,udg,filter,psf,psf_scale,source_cat,art_source_cat,out_source_cat,selection_param,mag_param,Z_mode,comp_params,ell_param,mags) :
    main = get_fits(source_cat)
    main_data = main[1].data

    art = get_fits(art_source_cat)
    art_data = art[1].data

    if Z_mode == True :
        c_medians = list()
        c_stds = list()
        bright_stars_art_data = art_data[(art_data[mag_param]<26) & (art_data[mag_param]>22)]
        
        for comp_param in comp_params :
            c_list = bright_stars_art_data[comp_param] 
            c_list = c_list[abs(c_list)<10]
            c_list = sigma_clip(c_list,sigma=5,maxiters=1)
            c_median = np.nanmedian(c_list)
            c_std = np.nanstd(c_list)
            c_medians.append(c_median)
            c_stds.append(c_std)
        
        z = 0
        z_art = 0
        #zz = 0
        #zz_art = 0
        for i in range(len(comp_params)) :
            comp_param = comp_params[i]
            c_median = c_medians[i]
            c_std = c_stds[i]
            c_list = abs(main_data[comp_param] - c_median)
            c_art_list = abs(art_data[comp_param] - c_median)
            z = z + c_list
            z_art = z_art+c_art_list
            #print (comp_param, c_median, c_std)

            #mags = np.arange(22,30.01,0.1)
            #std = find_std_of_param_for_magnitude(art_source_cat,comp_param,mag_param,mags, 0.2, c_median)

            #c_list = abs(main_data[comp_param] - c_median) / std[((main_data[mag_param]-22)*10).astype(int)]
            #c_art_list = abs(art_data[comp_param] - c_median) / std[((art_data[mag_param]-22)*10).astype(int)]
            #zz = zz + c_list
            #zz_art = zz_art+c_art_list
        ##### ellipticity 

        #std = find_std_of_param_for_magnitude(art_source_cat,ell_param,mag_param,mags, 0.1, 0)
        #ee = abs(main_data[ell_param]) / std[((main_data[mag_param]-22)*10).astype(int)]
        #ee_art = abs(art_data[ell_param]) / std[((art_data[mag_param]-22)*10).astype(int)]

        ###

        z = np.array(z)
        z_art = np.array(z_art) 

        #zz = np.array(zz)
        #zz_art = np.array(zz_art) 

        #ee = np.array(ee)
        #ee_art = np.array(ee_art) 

        os.system('rm '+' '+udg+'_'+filter+'_catalogue+Z.fits')
        os.system('rm '+' '+udg+'_'+filter+'_artificial_stars_catalogue+Z.fits')
        os.system('cp '+source_cat+' '+udg+'_'+filter+'_catalogue+Z.fits')
        os.system('cp '+art_source_cat+' '+udg+'_'+filter+'_artificial_stars_catalogue+Z.fits')
        expand_fits_table(udg+'_'+filter+'_catalogue+Z.fits','Z_'+filter,z)
        expand_fits_table(udg+'_'+filter+'_artificial_stars_catalogue+Z.fits','Z_'+filter,z_art)
        #expand_fits_table(udg+'_'+filter+'_catalogue+Z.fits','ZZ_'+filter,zz)
        #expand_fits_table(udg+'_'+filter+'_artificial_stars_catalogue+Z.fits','ZZ_'+filter,zz_art)
        #expand_fits_table(udg+'_'+filter+'_catalogue+Z.fits','EE_'+filter,ee)
        #expand_fits_table(udg+'_'+filter+'_artificial_stars_catalogue+Z.fits','EE_'+filter,ee_art)
    else :
        c = main_data[comp_params]
        c_art = art_data[comp_params]

    main = get_fits(udg+'_'+filter+'_catalogue+Z.fits')
    main_data = main[1].data
    main_data_mags = main_data[mag_param]
    main_data_Z = main_data['Z_'+filter]
    main_data_c = main_data['c48_'+filter]
    N = len(main_data_mags)

    art = get_fits(udg+'_'+filter+'_artificial_stars_catalogue+Z.fits')
    art_data = art[1].data
    art_data_mags = art_data[mag_param]
    art_data_Z = art_data['Z_'+filter]
    art_data_c = art_data['c48_'+filter]
    M = len(art_data_mags)

    MEDIAN, STD1, STD2 = find_std_of_param_for_magnitude(udg+'_'+filter+'_artificial_stars_catalogue+Z.fits','c48_'+filter,mag_param,mags,0.1)
    
    mask1 = list()
    for i in range(N) :
        m0 = main_data_mags[i]
        if m0 > np.max(mags) :
            m0 = np.max(mags)
        m = int(m0*10)/10.
        m = int((m-22)*10)
        median = MEDIAN[m]
        std1 = STD1[m]
        std2 = STD2[m]
        z = main_data_Z[i]
        c = main_data_c[i]
        if c < median + 4*std2 + 0.05 and c > median - 4*std1 - 0.05 :
            temp = True
        else :
            temp = False
        mask1.append(temp)
    mask1 = np.array(mask1)

    mask2 = list()
    for i in range(M) :
        m0 = art_data_mags[i]
        if m0 > np.max(mags) :
            m0 = np.max(mags)
        m = int(m0*10)/10.
        m = int((m-22)*10)
        median = MEDIAN[m]
        std1 = STD1[m]
        std2 = STD2[m]
        z = art_data_Z[i]
        c = art_data_c[i]
        if c < median + 4*std2 + 0.05 and c > median - 4*std1 - 0.05 :
            temp = True
        else :
            temp = False
        mask2.append(temp)
    mask2 = np.array(mask2)
        

    #mask = (art_data[mag_param]<26)
    #zz_art = art_data['ZZ_'+filter]
    #ee_art = art_data['EE_'+filter]
    #zz_art = zz_art(main_data[mag_param]<27)
    #ee_art = ee_art(main_data[mag_param]<27)
    #zz_art = sigma_clip(zz_art,sigma=3)
    #ee_art = sigma_clip(ee_art,sigma=3)
    #zz_art = np.sort(zz_art[mask])
    #ee_art = np.sort(ee_art[mask])
    #zz_th = zz_art[int(0.99*len(zz_art))]
    #ee_th = ee_art[int(0.9*len(ee_art))]

    #mask = ((main_data['ZZ'+'_'+filter]<zz_th) ) # & (main_data['EE'+'_'+filter]<ee_th))

    point_data = main_data[mask1]
    point_data = point_data[point_data['NIMAFLAGS_ISO'+'_'+filter]>-1]
    point_data = point_data[point_data['mag'+'_'+filter]>22.0]

    main[1].data = point_data
    main.writeto(udg+'_'+filter+'_catalogue_points_sources.fits',clobber=True)

    #mask = ((art_data['ZZ'+'_'+filter]<zz_th) ) #& (art_data['EE'+'_'+filter]<ee_th))
    point_data = art_data[mask2]
    point_data = point_data[point_data['NIMAFLAGS_ISO'+'_'+filter]>-1]
    point_data = point_data[point_data['mag'+'_'+filter]>22.0]
    
    art[1].data = point_data
    art.writeto(udg+'_'+filter+'_selected_artificial_stars.fits',clobber=True)

    return udg+'_'+filter+'_catalogue_points_sources.fits',udg+'_'+filter+'_selected_artificial_stars.fits', MEDIAN, MEDIAN, STD1, STD2
    

def find_std_of_param_for_magnitude(table, param, mag_param, mags, bin_size) :
    main = get_fits(table)
    main_data = main[1].data
    STD1 = list()
    STD2 = list()
    MEDIAN = list()
    i = 0
    for mag in mags :
        mask = ((main_data[mag_param]>22) & (main_data[mag_param]<26) & (main_data[param]<10))
        temp = main_data[mask]
        temp = temp[param] #- median
        temp = sigma_clip(temp,sigma=3,maxiters=1)
        median = np.nanmedian(temp)
        MEDIAN.append(median)
    
    C = MEDIAN[0]
    
    for mag in mags :

        mask = ((main_data[mag_param]>mag-bin_size) & (main_data[mag_param]<mag+bin_size) & (main_data[param]<10) & (main_data[param]<= C))
        temp = main_data[mask]
        temp = temp[param] #- median
        temp = sigma_clip(temp,sigma=5,maxiters=3)
        std = np.nanstd(temp)
        #median = np.nanmedian(temp)
        STD1.append(std)

        mask = ((main_data[mag_param]>mag-bin_size) & (main_data[mag_param]<mag+bin_size) & (main_data[param]<10) & (main_data[param]>= C))
        temp = main_data[mask]
        temp = temp[param] #- median
        temp = sigma_clip(temp,sigma=5,maxiters=3)
        std = np.nanstd(temp)
        #median = np.nanmedian(temp)
        STD2.append(std)
    
        
    MEDIAN = np.array(MEDIAN)
    STD1 = np.array(STD1)
    STD1 = gaussian_filter(STD1,sigma=3)
    STD2 = np.array(STD2)
    STD2 = gaussian_filter(STD2,sigma=3)
    #print (STD)
    return MEDIAN, STD1, STD2

def make_total_comp_cat(comp_cats,out_comp,average_shift) :
    comps_total = 1
    out_cat = open(out_comp,'w')
    out_cat.write('mag,completeness\n')
    i = -1
    for cat in comp_cats :
        i = i+1
        mags, comps = read_comp_cats(cat)
        if i == 0:
            mags0 = mags
        bin_size = mags[1]-mags[0]
        n_shift = int(average_shift[i]/bin_size+math.copysign(1, average_shift[i])*0.5)
        #print (n_shift)
        comps = np.roll(comps, n_shift, axis=0)
        comps_total = comps_total*comps
    
    for i in range(len(mags0)):
        out_cat.write(str(mags0[i])+', '+str(comps_total[i])+'\n')
    out_cat.close()
    return mags, comps_total


def read_comp_cats(comp_cat) :
    cat = open(comp_cat)
    i = 0
    mags = list()
    comps = list()
    for line in cat :
        #print (line)
        i = i+1
        if i == 1 :
            continue
        line = line.split(',')
        mag = float(line[0])
        comp = float(line[1])
        mags.append(mag)
        comps.append(comp)
    mags = np.array(mags)
    comps = np.array(comps)
    return mags, comps

##############

def color_select(selected_cat, secondary_cat, udg, main_filter, secondary_filter, final_selected_cat, filter1, filter2, colour1, colour2) :
    N1, N2 = cross_match(secondary_cat, selected_cat,'RA_'+secondary_filter,\
        'DEC_'+secondary_filter,'RA_'+main_filter,'DEC_'+main_filter,final_selected_cat)

    main0 = pyfits.open(secondary_cat)
    main_data0 = main0[1].data
    RA0 = main_data0['RA_'+secondary_filter]
    DEC0 = main_data0['DEC_'+secondary_filter]
    mag0 = main_data0['mag_'+secondary_filter]
    M = len(RA0)

    cols = main0[1].columns.names
    for col in cols :
        expand_fits_table(final_selected_cat,col,np.full(N2,-99.9999))

    main = pyfits.open(final_selected_cat)
    main_data = main[1].data
    RA = main_data['RA_'+main_filter]
    DEC = main_data['DEC_'+main_filter]
    mag = main_data['mag_'+main_filter]
    N = len(RA)

    for i in range(N) :
        check = 0 
        for j in range(M) :
            ra = RA[i]
            dec = DEC[i]
            ra0 = RA0[j]
            dec0 = DEC0[j]
            dist = np.sqrt( (ra-ra0)**2 + (dec-dec0)**2 )*3600
            if dist <= 0.1 and abs(mag[i]-mag0[j]) < 2.0 :
                for col in cols :
                    if 'APER' in col :
                        continue
                    #print (main_data[i][col])
                    #print (main_data0[j][col])
                    main_data[i][col] = main_data0[j][col]
                check = 1
                break

            if check == 1 :
                continue
                
    main[1].data = main_data
    main.writeto(final_selected_cat,clobber=True)

    cat = pyfits.open(final_selected_cat, ignore_missing_end=True)
    cat_data = cat[1].data
    mag1 = cat_data['mag_'+filter1]
    mag2 = cat_data['mag_'+filter2]
    c = mag1 - mag2 

    emag1 = cat_data['emag_'+filter1]
    emag2 = cat_data['emag_'+filter2]
    ec = np.sqrt(emag1*emag1 + emag2*emag2) 
    
    color_selects = list()
    for i in range(len(c)):
        color_select = 0.0
        if (c[i] < colour2 and c[i] > colour1) :
            color_select = 1.0
        elif (c[i]-ec[i] < colour2 and c[i]-ec[i] > colour1) :
            color_select = 1.0
        elif (c[i]+ec[i] < colour2 and c[i]+ec[i] > colour1) :
            color_select = 1.0
        color_selects.append(color_select) 
    color_selects = np.array(color_selects)

    
    expand_fits_table(final_selected_cat,filter1+'_'+filter2,c)
    expand_fits_table(final_selected_cat,'err_'+filter1+'_'+filter2,ec)
    expand_fits_table(final_selected_cat,'colour_select',color_selects)
    os.system('cp '+final_selected_cat+' cat_'+udg+'.fits')
    clean_fits_table(final_selected_cat,{'colour_select':[0.5,1.5]})
    #clean_fits_table(final_selected_cat,{filter1+'_'+filter2:[colour1,colour2]})
