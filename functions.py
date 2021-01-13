import numpy as np
import pyfits
import astropy
import astropy.io as fits
import os, sys
import math
import random
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

def get_fits(filefile):
    '''
    Reads the input fits-file and returns the hdu table.
    '''
    hdulist = pyfits.open(filefile)
    return hdulist
    

def make_directory(directory_path,directory_name) :
    if path.exists(directory_path+directory_name) :
        do_nothing = 1
    else :
        os.system('mkdir '+directory_path+directory_name)

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


def get_total_number_of_objects(table) :
    table_main = get_fits(table)
    table_data = table_main[1].data
    total_number_of_objects = len(table_data)
    print ('total numer of objects in table '+table+' is : ' + str(total_number_of_objects))
    return total_number_of_objects

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


def convert_fits_to_csv(dataset,output) :
    main = get_fits(dataset)
    print ('teymoor')
    X = main[1].data
    print ('converting fits to csv is started')
    ascii.write(X, output, format='csv', fast_writer=False, overwrite=True)
    print ('converting fits to csv is finished')


def convert_csv_to_fits(dataset,output) :
    os.system('rm '+output)
    #text_file = ascii.read(dataset)
    #text_file.write(output)
    os.system('python csv-to-fits.py '+dataset+' '+output)


def expand_fits_table(table,new_param,new_param_values) :
    fits = FITS(table,'rw')
    fits[-1].insert_column(name = new_param, data = new_param_values) 
    fits.close()


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
        hdu.writeto(output_table)

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

def resample(main_data,obj_name,filter_name) :
    output = obj_name+'_'+filter_name+'_resampled.fits'
    command = 'SWarp '+main_data+' -c default.swarp -IMAGEOUT_NAME '+output
    os.system(command)
    return output

def resample_weight(main_data,weight_data,obj_name,filter_name) :
    X = get_header(main_data,keyword='NAXIS1')
    Y = get_header(main_data,keyword='NAXIS2')
    output = obj_name+'_'+filter_name+'.weight_resampled.fits'
    command = 'SWarp '+weight_data+' -c default.swarp -IMAGEOUT_NAME '+output+' -IMAGE_SIZE '+str(X)+','+str(Y)
    os.system(command)
    return output