import sys, os
import pyfits
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from astropy.wcs import WCS
from astropy import wcs
from astropy.io import fits
import os.path
import scipy
from scipy import signal
from scipy import ndimage
from pylab import *
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.image as mpimg
from matplotlib.ticker import ScalarFormatter
import astropy.io.fits
from matplotlib.ticker import StrMethodFormatter
import matplotlib.ticker as ticker
from functions import *
from astropy.stats import sigma_clip
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from scipy import ndimage, misc
import os.path


def get_ellipse(xc,yc,a,b,pa,res = 360):
    '''
    Returns an ellipse with the given parameters.
    pa in degrees and context of astronomy.
    '''
    try:
        dum=len(xc)
    except:
        xc,yc,a,b,pa=np.array([xc]),np.array([yc]),np.array([a]),np.array([b]),np.array([pa])
    pa = (pa+90.) / 360. *2.*np.pi
    theta = np.arange(res+1)/float(res)*2.*np.pi
    theta = np.array([theta for i in range(len(xc))]).T
    xc = np.array([xc for i in range(res+1)])
    yc = np.array([yc for i in range(res+1)])
    a = np.array([a for i in range(res+1)])
    b = np.array([b for i in range(res+1)])
    pa = np.array([pa for i in range(res+1)])
    r = a*b / np.sqrt((b*np.cos(theta-pa))**2. + (a*np.sin(theta-pa))**2.)
    elx = xc + r * np.cos(theta)
    ely = yc + r * np.sin(theta)
    return elx,ely

#----------------------------------------------------------------------

#----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

def get_aperture_mag(data,ellipse,mask=None,sky_annulus=None):
    '''
    Get aperture magnitude with the given ellipse parameters.
    ellipse = [xc,yc,A,B,PA] (PA in radians and from x -axis)
    sky_annulus = [minr,maxr]

    Returns:
    sum,error (no ksy paerture)
    sum,error,skylevel (sky aperture)
    '''
    xc,yc,A,B,PA = ellipse
    nx,ny = len(data[0]),len(data)
    A = A
    B = B
    # Define elliptical coordinates
    x,y = np.ogrid[-yc:float(ny)-yc,-xc:float(nx)-xc]
    angles = np.arctan(y/x)
    angles = np.pi/2.-angles
    angles[:yc,:] = angles[:yc,:]+np.pi
    angles[yc,:xc] = np.pi
    distance = np.sqrt(x**2.+y**2.)
    d, a = distance , angles
    edist = d / ( B / np.sqrt((B*np.cos(a-PA))**2.+(A*np.sin(a-PA))**2.))
    edist[yc,xc] = 0.
    # Gather the pixels within  the aperture (exclude the masked pixels)
    pix_ind = edist < A
    #if mask != None:
    mask_ind = mask==0
    pix_ind *= mask_ind
    if sky_annulus != None:
        p1 = edist > sky_annulus[0]
        p2 = edist < sky_annulus[1]
        sky_ind = p1*p2
        #if mask != None:
        sky_ind*=mask_ind
        skylevel = np.median(data[sky_ind])
        skysigma = np.std(data[sky_ind])
    flux = data[pix_ind]
    sum = np.sum(data[pix_ind])
    error = -2.5*np.log10(np.median(flux))+2.5*np.log10(np.median(flux)+np.std(flux)/np.sqrt(len(flux)))

    ##############
    min_step   = 5
    n_sky_bins = 18
    dist_sky   = 5. #REFFS
    nx,ny = len(data[0]),len(data)
    xc,yc,A,B,PA = ellipse
    fluxbins = []
    errors   = []
    rs = []
    #Define elliptical coordinates
    x,y = np.ogrid[-yc:float(ny)-yc,-xc:float(nx)-xc]
    angles = np.arctan(y/x)
    angles = np.pi/2.-angles
    angles[:yc,:] = angles[:yc,:]+np.pi
    angles[yc,:xc] = np.pi
    distance = np.sqrt(x**2.+y**2.)
    d, a = distance , angles
    edist = d / ( B / np.sqrt((B*np.cos(a-PA))**2.+(A*np.sin(a-PA))**2.))
    edist[yc,xc] = 0.
    reff = A

    # Deal with the sky (if asked):
    dthet      = 2.*np.pi/n_sky_bins
    sky_bins   = []
    sky_bins_values = []
    sky_radius = int( dist_sky * reff )
    binsize = 3
    if (sky_radius+binsize >= len(data)/2):
        sky_radius = len(data)/2-binsize
    max_r = sky_radius+binsize > edist
    min_r = sky_radius-binsize < edist
    rind = min_r*max_r
    for i in range(n_sky_bins):
        min_the =    i*dthet   < angles
        max_the = (i+1.)*dthet > angles
        ind_the = min_the*max_the
        ind     = ind_the*rind
        #if mask != None:
        sky_pix = data[ind][mask[ind] == 0 ]
        #else:
        #    sky_pix = data[ind]

        sky_bins_values.append('None')
        not_nan = ~np.isnan(sky_pix)
        sky_pix = sigma_clip(sky_pix[not_nan],3,maxiters=3)
        sky_bins.append(np.median(sky_pix))

        not_nan = ~np.isnan(sky_bins)
        sky_bin = sigma_clip(np.array(sky_bins)[not_nan],3,maxiters=3)
        sky_noise = np.std(np.array(sky_bin))
        sky_level = np.median(np.array(sky_bin))
    #############
    if sky_annulus==None:
        return sum,error
    else:
        #print sky_level, sky_noise, skylevel, skysigma
        #print sky_level, sky_noise, skylevel, skysigma
        return sum,error,sky_level,sky_noise
        #return sum,error,skylevel,skysigma

#----------------------------------------------------------------------

#----------------------------------------------------------------------

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
    print ('Saved postage stamp to '+objectname+'_'+filtername+'_cropped'+str(label)+'.fits')
    output_frame = objectname+'_'+filtername+'_cropped.fits'
    # no jpg conversion

    #print 'DONE!'
    del template,hdu
    #del template2,hdu2

    w=WCS(objectname+'_'+filtername+'_cropped'+str(label)+'.fits')
    x_gal_center,y_gal_center = w.all_world2pix(ra, dec,0)

    return output_frame, x_gal_center,y_gal_center

    

#----------------------------------------------------------------------

#----------------------------------------------------------------------

def mask_stars (frame, ra, dec, objname, filtername, r_cut, r_cut_back_mask=0, q=1, pa=0, blurred=0, label='') :
    #frame = main_data
    #print ('+++ Making mask frame for '+frame)
    X = get_header(frame,keyword='NAXIS1')
    Y = get_header(frame,keyword='NAXIS2')

  
    os.system('sextractor '+frame+' -c sex/default_galfit.sex -CATALOG_NAME '+frame+'.sex_cat.fits '+ \
    '-DETECT_MINAREA 5 -DETECT_THRESH 2 -ANALYSIS_THRESH 2 \
    -BACK_SIZE 16 -CHECKIMAGE_NAME '+objname+'_'+filtername+'.check.fits')

    os.system('rm '+frame+'.sex_cat.fits')
    img1 = pyfits.open(frame)
    img2 = pyfits.open(objname+'_'+filtername+'.check.fits')
    data1 = img1[0].data
    data2 = img2[0].data

    data1 = data1-data2
    #print (data1)
    #data1 = data1.astype(int)
    img1[0].data = data1
    img1.writeto(objname+'_'+filtername+'_median_subtracted'+str(label)+'.fits',clobber=True)

    os.system('sextractor '+objname+'_'+filtername+'_median_subtracted'+str(label)+'.fits'+\
        ' -c sex/default_galfit.sex -CATALOG_NAME '+\
    objname+'_'+filtername+'.sex_cat.fits '+ \
    '-DETECT_MINAREA 5 -DETECT_THRESH 2 -ANALYSIS_THRESH 2 \
    -BACK_SIZE 64 -CHECKIMAGE_TYPE NONE')

    #merge fits catalogues
    cat = pyfits.open(objname+'_'+filtername+'.sex_cat.fits')

    img = pyfits.open(frame)
    #img = pyfits.open(objname+'_'+filtername+'_'+'.fits')
    #img2 = pyfits.open(objname+'_'+filtername+'.fits')
    table = cat[2].data
    data = img[0].data

    data2 = img[0].data
    data3 = img[0].data
    N = len(table)
    #print (image)
    for i in range (0,N) :
        params = table[i]
        ra_star = params[16]
        dec_star = params[17]
        x = params[14]
        y = params[15]
        fwhm = params[27]
        flag = params[25]
        A = params[18]
        B = params[19]
        r = math.sqrt((ra-ra_star)**2+(dec-dec_star)**2)
        if flag <= 9999 and r >= 0.2/3600.:
            #print (r*3600., ra_star, dec_star)
            if x >= X or y >= Y :
                continue
            x = int(x)
            y = int(y)
            fwhm = int(fwhm+0.5)
            #mask_size = int(2.0*(fwhm))
            if blurred==0 :
                mask_size = int(10*B)
            if blurred==1 :
                mask_size = int(10*B)
            #print (mask_size)
            #data[y][x] = 0  #?????
            #data2[y][x] = min_value#
            for i in range(0,mask_size+1) :
                for j in range(0,mask_size+1) :

                    rr = math.sqrt((i)**2+(j)**2)
                    if rr <= mask_size :
                        if x+i < X and y+j < Y :
                            data[y+j][x+i] = 1
                            #data2[y+j][x+i] = min_value
                        if x+i < X and y-j > 0 :
                            data[y-j][x+i] = 1
                            #data2[y-j][x+i] = min_value
                        if x-i > 0 and y+j < Y :
                            data[y+j][x-i] = 1
                            #data2[y+j][x-i] = min_value
                        if x-i > 0 and y-j > 0 :
                            data[y-j][x-i] = 1
                            #data2[y-j][x-i] = min_value

            

        """                   
        elif flag <= 9999 and r <= 8./3600.:
            #print (r*3600., ra_star, dec_star)
            x = int(x)
            y = int(y)
            fwhm = int(fwhm+0.5)
            #mask_size = int(2.0*(fwhm))
            mask_size = int(6*B)
            #data[y][x] = 0  #?????
            
            #data2[y][x] = min_value#
            for i in range(0,mask_size+1) :
                for j in range(0,mask_size+1) :

                    rr = math.sqrt((i)**2+(j)**2)
                    if rr <= mask_size :
                        if x+i < X and y+j < Y :
                            data[y+j][x+i] = 1
                            #data2[y+j][x+i] = min_value
                        if x+i < X and y-j > 0 :
                            data[y-j][x+i] = 1
                            #data2[y-j][x+i] = min_value
                        if x-i > 0 and y+j < Y :
                            data[y+j][x-i] = 1
                            #data2[y+j][x-i] = min_value
                        if x-i > 0 and y-j > 0 :
                            data[y-j][x-i] = 1
                            #data2[y-j][x-i] = min_value
        """

    if r_cut_back_mask > 0 :
        #ellipse = [X/2,Y/2,r_cut_back_mask,q*r_cut_back_mask,(90+pa)/360.*2.*np.pi]
        for i in range(X) :
            for j in range(Y) :
                check = pointInEllipse_pixel([i,j], [X/2,Y/2], r_cut_back_mask,q*r_cut_back_mask,\
                    (90+pa)/360.*2.*np.pi)
                if check == True :
                    data[i,j] = data[i,j]
                elif check == False :
                    data[i,j] = 1



        
    where_are_NaNs = isnan(data)
    data[where_are_NaNs] = 1
    data[abs(data)<0.0000000001] = 1
    data = data.astype(int)
    img[0].data = (data)
    #print (data)
    img.writeto(objname+'_'+filtername+'_masked'+str(label)+'.fits',clobber=True)

    w = 1 - data
    data2 = data2*w
    #data2 = np.log10(data2)
    #data2 = 10**(data2)
    img[0].data = (data2)
    img.writeto(objname+'_'+filtername+'_masked+'+str(label)+'.fits',clobber=True)

    w = 1 - data
    #data3 = data3*w
    img[0].data = (w)
    img.writeto(objname+'_'+filtername+'_weight'+str(label)+'.fits',clobber=True)

    os.system('rm '+objname+'_'+filtername+'.sex_cat.fits')

    return objname+'_'+filtername+'_masked'+str(label)+'.fits',\
           objname+'_'+filtername+'_masked+'+str(label)+'.fits',\
           objname+'_'+filtername+'_median_subtracted'+str(label)+'.fits',\
           objname+'_'+filtername+'_weight'+str(label)+'.fits'



#----------------------------------------------------------------------

#----------------------------------------------------------------------

def sersic(r, ne, re, n, bkg) :
    return ne * np.exp(-1*(1.9992*n-0.3271)*((r/re)**(1./n)-1)) + bkg

def make_imfit_input(main_data, obj_name, filter_name, x, y, back, e=0.5, pa=0, constrain=0) :
    """
    X0   724 710,740
    Y0   703  610,720
    FUNCTION Sersic
    PA    -65.8    -65.8001,-65.80
    ell    0.344   0.344,0.344001
    n      0.72   0.72,0.720001
    I_e    0.002     0,0.1
    r_e    197     197,197.0001
    FUNCTION FlatSky
    I_sky   0.01933 0.01933,0.01933001
    """
    x = int(x)
    y = int(y)
    if constrain == 0 :
        output_file = obj_name+'_'+filter_name+'_imfit_input.dat'
        text = open(output_file,'w')
        text.write('X0 '+str(x)+' '+str(x-25)+','+str(x+25)+'\n')
        text.write('Y0 '+str(y)+' '+str(y-25)+','+str(y+25)+'\n')
        text.write('FUNCTION Sersic'+'\n')
        text.write('PA '+str(pa)+' '+str(0)+','+str(360)+'\n')
        text.write('ell '+str(e)+' '+str(0)+','+str(1)+'\n')
        text.write('n '+str(1)+' '+str(0.5)+','+str(2.0001)+'\n')
        text.write('I_e '+str(3*back)+' '+str(1*back)+','+str(1)+'\n')
        text.write('r_e '+str(160)+' '+str(80)+','+str(320)+'\n')
        text.write('FUNCTION FlatSky'+'\n')
        text.write('I_sky '+str(back)+' fixed'+'\n')

    if constrain == 1 :
        output_file = obj_name+'_'+filter_name+'_imfit_input.dat'
        text = open(output_file,'w')
        text.write('X0 '+str(x)+' fixed'+'\n')
        text.write('Y0 '+str(y)+' fixed'+'\n')
        text.write('FUNCTION Sersic'+'\n')
        text.write('PA '+str(pa)+' fixed'+'\n')
        text.write('ell '+str(e)+' fixed'+'\n')
        text.write('n '+str(1)+' '+str(0.5)+','+str(2)+'\n')
        text.write('I_e '+str(3*back)+' '+str(1*back)+','+str(1)+'\n')
        text.write('r_e '+str(150)+' '+str(80)+','+str(320)+'\n')
        text.write('FUNCTION FlatSky'+'\n')
        text.write('I_sky '+str(back)+' fixed'+'\n')


    return obj_name+'_'+filter_name+'_imfit_input.dat'


def make_weight_map(frame,weight_file,back,back_std) :
    main = get_fits(frame)
    data = main[0].data
    weight = data
    data = data-back 
    data[data < 0] = 0
    weight[weight > 1] = 0
    gain = get_header(frame,'CCDGAIN')
    sigma = (back_std**2+data)
    weight = sigma #1./(sigma)
    #weight = (weight*1000)
    #weight = weight**2
    main[0].data = weight
    main.writeto(weight_file,clobber=True)

def change_weight_map(weight_in,weight_out,make_weight) :
    main = get_fits(weight_in)
    weight = main[0].data
    #weight = weight**0.1
    weight[weight>0] = weight[weight>0]**make_weight#**0.05#**0.5
    #median = np.median(weight[weight>0])
    #weights = weight-1
    #weight[weight<0] = 0
    #weight[weight<median] = 0
    #weight = weight**(1./3.)
    main[0].data = weight
    main.writeto(weight_out,clobber=True)


def run_imfit(frame,mask,input_file,obj_name,filter_name,back=0,back_std=0,make_weight=0) :

    #main = get_fits(frame)
    #data = main[0].data
    #data[data<-90] = 99
    #main[0].data = data
    #os.system('rm '+frame)
    #main.writeto(frame)
    res = obj_name+'_'+filter_name+'_res.fits'
    model = obj_name+'_'+filter_name+'_model.fits'
    weight = obj_name+'_'+filter_name+'_weight.fits'
    weight_out = obj_name+'_'+filter_name+'_weight_out.fits'
    #make_weight_map(frame,weight,back,back_std)
    #command = './fit/imfit '+frame+' -c '+input_file+' -save-model '+model+ ' -save-residual '+res+\
    #    ' --mask '+mask+' --sky='+str(back)
    #print (command)

    #gain = get_header(frame,'CCDGAIN')
    #readnoise = get_header(frame,'READNSEA')
    #exptime = get_header(frame,'TEXPTIME')

    common = ''#--model-errors'#--gain '+str(gain)+' --readnoise '+str(readnoise)+' --exptime '+str(exptime)

    #os.system('./fit/imfit '+frame+' -c '+input_file+' -save-model '+model+ ' -save-residual '+res+\
    #        ' --mask '+mask+' --save-weights '+weight)#t+common+ ' --poisson-mlr')#+' --sky='+str(back))


    if make_weight == -1 : #use exisitng weight-map
        os.system('./fit/imfit '+frame+' -c '+input_file+' -save-model '+model+ ' -save-residual '+res+\
            ' --mask '+mask+' --save-weights '+weight_out+' --noise '+weight+' --errors-are-weights'+common)

    elif make_weight == 1 : #internal weightmap (**1)
        os.system('./fit/imfit '+frame+' -c '+input_file+' -save-model '+model+ ' -save-residual '+res+\
            ' --mask '+mask)
    
    else: 
        os.system('./fit/imfit '+frame+' -c '+input_file+' -save-model '+model+ ' -save-residual '+res+\
            ' --mask '+mask+' --save-weights '+weight+common)#+ ' --poisson-mlr')#+' --sky='+str(back))
        change_weight_map(weight,weight,make_weight)
        os.system('./fit/imfit '+frame+' -c '+input_file+' -save-model '+model+ ' -save-residual '+res+\
            ' --mask '+mask+' --save-weights '+weight_out+' --noise '+weight+' --errors-are-weights'+common)
            #+ ' --poisson-mlr')#+' --sky='+str(back))
        

    
    text = open('bestfit_parameters_imfit.dat')
    for lines in text :
        line = lines.split()
        #print (line)
        # check convergence

        if len(line) > 0 :
            if line[0] == 'PA' :
                pa = float(line[1])
                epa = float(line[4])
            if line[0] == 'ell' :
                ell = float(line[1])
                eell = float(line[4])
            if line[0] == 'n' :
                n = float(line[1])
                en = float(line[4])
            if line[0] == 'r_e' :
                r = float(line[1])
                er = float(line[4])
            if line[0] == 'I_e' :
                i = float(line[1])
                ei = float(line[4])

            if line[0] == 'X0' :
                x0 = int(float(line[1]))
            if line[0] == 'Y0' :
                y0 = int(float(line[1]))


    return pa,ell,n,i,r,epa,eell,en,ei,er,x0,y0


def estimate_frame_back(frame,r_cut_back_mask) :

    X = get_header(frame,keyword='NAXIS1')
    Y = get_header(frame,keyword='NAXIS2')
    #r_cut_back_mask = r_cut_back_mask
    median_list = list()
    i = 0
    while i < 10000 :
        x = random.randint(51,X-51)
        y = random.randint(51,Y-51)

        if x < (X/2.+r_cut_back_mask) and x > (X/2.-1*r_cut_back_mask) :
            continue
        if y < (Y/2.+r_cut_back_mask) and y > (Y/2.-1*r_cut_back_mask) :
            continue

        main = get_fits(frame)
        fits_data = main[0].data
        fits_data = fits_data[x-50:x+50,y-50:y+50]
        fits_data = fits_data[fits_data>0]
        #fits_data = sigma_clip(fits_data,3,maxiters=3)
        median = np.median(fits_data)
        #if abs(median) > 0 :
        median_list.append(median)
        i = i+1

        #print (x,y,median)
    median_list = np.array(median_list)
    median_list = sigma_clip(median_list,3,maxiters=1)
    plt.hist(median_list,bins=20,color='blue')
    #median_list = sigma_clip(median_list,2,maxiters=5)
    #plt.hist(median_list,bins=20,color='green')
    median = np.median(median_list)
    std = np.std(median_list)
    plt.plot([median,median],[0,100],'k-')
    plt.plot([median+1*std,median+1*std],[0,100],'k--')
    plt.plot([median-1*std,median-1*std],[0,100],'k--')

    plt.savefig(frame+'_hist_background_values.png')
    plt.close()

    print ('+++ inital background values : '+ str(median) + ', '+str(std))

    return median, std, median_list

#----------------------------------------------------------------------

#----------------------------------------------------------------------

def make_galfit_feedme_file_sersic (main_file,mask_file, sigma_file, psf_file, objname,\
    ra, dec, reff, filtername, fit_x1, fit_x2, fit_y1, fit_y2, r_mag, axis_ratio, pos_angel, nuc, sersic_index, sky,\
        zp, pix_size, x_gal_center, y_gal_center) :
    print ('--- Making galfit configuration file for galaxy ' + str(objname))
    galfit_conf = open(objname+'_galfit.conf','w')
    galfit_conf.write('# IMAGE and GALFIT CONTROL PARAMETERS\n')
    galfit_conf.write('A) '+main_file+'\n')
    galfit_conf.write('B) '+objname+'_'+filtername+'_galfit_imgblock'+'.fits\n')
    galfit_conf.write('C) '+sigma_file+'\n')
    galfit_conf.write('D) '+psf_file+'\n')
    galfit_conf.write('E) 1\n')
    galfit_conf.write('F) '+mask_file+'\n')
    galfit_conf.write('G) none\n')
    x = get_header(main_file,keyword='NAXIS1')
    y = get_header(main_file,keyword='NAXIS2')
    galfit_conf.write('H) '+str(fit_x1)+' '+str(fit_x2)+' '+str(fit_y1)+' '+str(fit_y2)+'\n')
    galfit_conf.write('I) '+str(100)+' '+str(100)+'\n')
    galfit_conf.write('J) ' + str(zp) +'\n')
    galfit_conf.write('K) '+str(pix_size)+' '+str(pix_size)+'\n')
    galfit_conf.write('O) regular \n')
    galfit_conf.write('P) 0\n\n')
    galfit_conf.write('# Sersic\n')
    galfit_conf.write('0) sersic\n')
    galfit_conf.write('1) '+str(x_gal_center)+' '+str(y_gal_center)+' 1 1\n')
    galfit_conf.write('3) '+str(r_mag)+' 1\n')
    galfit_conf.write('4) '+str(reff)+' 1\n')
    galfit_conf.write('5) '+str(sersic_index)+' 1\n')
    galfit_conf.write('6) '+str(0.0000)+' 0\n')
    galfit_conf.write('7) '+str(0.0000)+' 0\n')
    galfit_conf.write('8) '+str(0.0000)+' 0\n')
    galfit_conf.write('9) '+str(axis_ratio)+' 1\n')
    galfit_conf.write('10) '+str(pos_angel)+' 1\n')
    galfit_conf.write('Z) 0\n\n')

    if nuc != -1 :
        print ('--- *sersic + psf + sky')
        galfit_conf.write('# psf\n')
        galfit_conf.write('0) psf\n')
        galfit_conf.write('1) '+str(x/2)+' '+str(y/2)+' 0 0\n')
        galfit_conf.write('3) '+str(float(nuc)+2.)+' 1\n')
        galfit_conf.write('Z) 0\n\n')
    else :
        print ('--- *sersic + sky')

    galfit_conf.write('# sky\n')
    galfit_conf.write('0) sky\n')
    galfit_conf.write('1) '+str(sky)+' 1\n')
    galfit_conf.write('2) '+str(0.000)+' 1\n')
    galfit_conf.write('3) '+str(0.000)+' 1\n')
    galfit_conf.write('Z) 0\n\n')

    galfit_conf.close()

#----------------------------------------------------------------------

#----------------------------------------------------------------------

def run_galfit_sersic (main_file,objname, filter_name) :

    print ('- Running galfit for ' + str(objname))
    os.system('rm fit.log')
    os.system('rm galfit.0*')
    os.system('rm '+objname+'_'+filter_name+'_galfit_imgblock*fits')
    os.system('./galfit/galfit '+objname+'_galfit.conf')

    if os.path.isfile(objname+'_'+str(filter_name)+'galfit_imgblock_model.fits') :
        os.system('rm '+objname+'_'+str(filter_name)+'galfit_imgblock_model.fits')
    if os.path.isfile(objname+'_'+str(filter_name)+'galfit_imgblock_res.fits') :
        os.system('rm '+objname+'_'+str(filter_name)+'galfit_imgblock_res.fits')
    if os.path.isfile(objname+'_'+str(filter_name)+'galfit_imgblock_res0.fits') :
        os.system('rm '+objname+'_'+str(filter_name)+'galfit_imgblock_res0.fits')

    model_frame = objname+'_'+filter_name+'_'+'galfit_imgblock_model.fits'
    res_frame = objname+'_'+filter_name+'_'+'galfit_imgblock_res.fits'
    #res0_frame = objname+'_'+filter_name+'_'+'galfit_imgblock_res0.fits'

    if os.path.isfile(objname+'_'+filter_name+'_galfit_imgblock'+'.fits') :
        imgblock = pyfits.open(objname+'_'+filter_name+'_galfit_imgblock'+'.fits')
        model = get_fits(main_file)
        res = get_fits(main_file)
        #res0 = get_fits(main_file)
        model[0].data = imgblock[2].data
        res[0].data = imgblock[3].data
        model.writeto(model_frame)
        res.writeto(res_frame)
        #res0.writeto(res0_frame)
    else :
        print (objname+'_'+filter_name+'_galfit_imgblock'+'.fits does not exist.')

    return model_frame, res_frame


def read_galfit_sersic (main_file, objname, filtername) :
    #print ('--- Reading galfit output values from fit.log')
    log = open('fit.log','r')
    frame = main_file
    i = 0
    n = 0
    X = get_header(main_file,keyword='NAXIS1')
    Y = get_header(main_file,keyword='NAXIS2')
    #x = X/2
    #y = Y/2

    mag_1, reff_1, d_mag_1, d_reff_1, pa, d_pa, axis_ratio, d_axis_ratio, \
    sky, sky_dx, sky_dy, d_sky, d_sky_dx, d_sky_dy = \
    (99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999)
    sersic_index_1, d_sersic_index_1 = 99999,99999
    x, y = 99999,99999
 

    for lines in log :
        i = i + 1
        line = lines.split()
        if frame in lines :
            n = 1
            continue
        if n > 0 :
            n = n + 1

        if n == 6 :
            if ('*' in str(lines)) :
                conv = 0
                continue
                print ('--- *warning: galfit solution has not been converged.')
            elif n == 6 :
                conv = 1

        if n == 6 and conv == 1 :
            mag_1 = float(line[5])
            reff_1 = float(line[6])
            sersic_index_1 = float(line[7])
            axis_ratio = float(line[8])
            pa = float(line[9])

            x = (line[3])
            y = (line[4])
            nx = len(x)
            ny = len(y)
            x = float(x[:nx-1])
            y = float(y[:ny-1])
            #print (x,y,nx,ny)

        if n == 7 and conv == 1 :
            d_mag_1 = float(line[3])
            d_reff_1 = float(line[4])
            d_sersic_index_1 = float(line[5])
            d_axis_ratio = float(line[6])
            d_pa = float(line[7])

        if n == 8 and line[2] == '[' :
            sky = float(line[5])
            sky_dx = float(line[6])
            sky_dy = float(line[7])
        if n == 8 and line[2] != '[' :
            sky = float(line[4])
            sky_dx = float(line[5])
            sky_dy = float(line[6])
        if n == 9 :
            d_sky = float(line[0])
            d_sky_dx = float(line[1])
            d_sky_dy = float(line[2])

    #print (objname, mag_1, reff_1, d_mag_1, d_reff_1, sersic_index_1, d_sersic_index_1, X, Y, \
    # x, y, sky, sky_dx, sky_dy, d_sky, d_sky_dx, d_sky_dy)
    return mag_1, reff_1, d_mag_1, d_reff_1, sersic_index_1, d_sersic_index_1, X, Y, \
        x, y, pa, d_pa, axis_ratio, d_axis_ratio, sky, sky_dx, sky_dy, d_sky, d_sky_dx, d_sky_dy


#----------------------------------------------------------------------

#----------------------------------------------------------------------

def make_radial_profile(data,ellipse,mask=None,sn_stop=1./3.,rad_stop=None,binsize=4,sky=False) :
    '''
    Makes a radial profile using the input paramateres and outputs
    the (masked) clipped values for each bin.

    input:

    ellipse     :    [xc,yc,a,b,pa] (pa should be in rads fron x axis)
    sn_stop     :    at whic signal to noise the profile should stop
    sky         :    it True, measures the sky level and variations at three reffs in bins

    returns:
    rad, flux, errors, (skyval, skynoise)
    '''
    #print (ellipse)
    min_step   = 5
    n_sky_bins = 12
    dist_sky   = 4. #REFFS
    nx,ny = len(data[0]),len(data)
    xc,yc,A,B,PA = ellipse
    fluxbins = []
    errors   = []
    rs = []
    #Define elliptical coordinates
    x,y = np.ogrid[-yc:float(ny)-yc,-xc:float(nx)-xc]
    angles = np.arctan(y/x)
    angles = np.pi/2.-angles
    angles[:yc,:] = angles[:yc,:]+np.pi
    angles[yc,:xc] = np.pi
    distance = np.sqrt(x**2.+y**2.)
    d, a = distance , angles
    edist = d / ( B / np.sqrt((B*np.cos(a-PA))**2.+(A*np.sin(a-PA))**2.))
    edist[yc,xc] = 0.
    #Iterate through the bins
    #r1,r2 = 0., 0.+binsize
    r1, r2 = 0, binsize
    step = 1
    while (step >= 0):
        uplim  = edist < r2
        lowlim = edist >= r1
        ind = uplim*lowlim
        bin_pix = data[ind]
        #if mask != None:
        bin_pix = bin_pix[ mask[ind] == 0  ]
        error = np.std(bin_pix)/np.sqrt(float(len(bin_pix)))
        if len(bin_pix)>5:
            bin_pix = sigma_clip(bin_pix,3,maxiters=3)
        level = np.mean(bin_pix)
        #store the bins values
        errors.append(error)
        fluxbins.append(level)
        rs.append((r1+r2)/2.)
        #Check if the time to stop has come
        #r1 += binsize
        #r2 += binsize
        r1 = r2
        r2 += binsize
        area = np.pi * ((r2*r2*(B/A))-(r1*r1*(B/A)))
        if rad_stop==None:
            if ( (level / error < sn_stop ) & (step>min_step) ) | (step*binsize>0.9*len(data)/2.):
                step=-10
        else:
            if r2 > rad_stop:
                step=-10
        step+=1
    # Calculate reff:
    fluxbins,errors,rs = np.array(fluxbins),np.array(errors),np.array(rs)
    reff = A

    # Deal with the sky (if asked):
    if sky:
        dthet      = 2.*np.pi/n_sky_bins
        sky_bins   = []
        sky_bins_values = []
        sky_radius = int( dist_sky * reff )
        if (sky_radius+binsize >= len(data)/2):
            sky_radius = len(data)/2-binsize
        max_r = sky_radius+binsize > edist
        min_r = sky_radius-binsize < edist
        rind = min_r*max_r
        for i in range(n_sky_bins):
            min_the =    i*dthet   < angles
            max_the = (i+1.)*dthet > angles
            ind_the = min_the*max_the
            ind     = ind_the*rind
            #if mask != None:
            sky_pix = data[ind][mask[ind] == 0 ]
            #else:
            #    sky_pix = data[ind]

            sky_bins_values.append('None')
            not_nan = ~np.isnan(sky_pix)
            sky_pix = sigma_clip(sky_pix[not_nan],3,maxiters=5)
            sky_bins.append(np.median(sky_pix))

        not_nan = ~np.isnan(sky_bins)
        sky_bin = sigma_clip(np.array(sky_bins)[not_nan],3,maxiters=5)
        sky_noise = np.std(np.array(sky_bin))
        sky_level = np.median(np.array(sky_bin))
        return rs,fluxbins,errors,sky_level,sky_noise,sky_bin, sky_bins_values, area
    else:
        return rs,fluxbins,errors, area



#----------------------------------------------------------------------

#----------------------------------------------------------------------



def fit_galaxy_sersic(main_data,ra,dec,obj_name,filter_name,pix_size,fit_dir,zp,\
    r_cut,r_cut_back,r_cut_back_mask,r_cut_fit,plotting=False) :

    print ('\n+ Sersic fitting for the galaxy : '+str(obj_name))
    print (r_cut,r_cut_back,r_cut_back_mask)
    #r_cut = 700 #arcsec
    Re = list()
    n = list()
    PA = list()
    q = list()
    Ie = list()

    e_Re = list()
    e_n = list()
    e_PA = list()
    e_q = list()
    e_Ie = list()

    #r_cut0 = r_cut
    #estimate background 
    """
    r_cut = r_cut_back
    cropped_frame, x_gal_center, y_gal_center = cut(main_data, ra, dec, r_cut , obj_name, \
        filter_name, overwrite=True, blur=8)
    while True :
        back_average, back_std, backs_all = estimate_frame_back(cropped_frame, r_cut_back_mask)
        if back_average > 0 and back_std > 0 :
            break
        
    #backs = np.linspace(back_average-back_std,back_average+back_std,num=5)

    #back_average, back_std = estimate_frame_back(cropped_frame)
    #backs = np.random.normal(back_average, back_std, 20)
    """
    """
    # fit using the backbground value - find initial pa and e
    r_cut = r_cut0
    print ('+ finding intial Position angle and ellipticity')
    print ('+++ cropping ')
    cropped_frame, x_gal_center, y_gal_center = cut(main_data, ra, dec, r_cut , obj_name, \
        filter_name, overwrite=True, blur=8, label='_blurred')
    print (x_gal_center, y_gal_center)
    print ('+++ masking ')
    mask_frame, masked_frame, check_frame = mask_stars(cropped_frame, ra, dec, obj_name, filter_name, \
        r_cut,blurred=1,label='_blurred')
    print ('+++ fitting ')

    imfit_input = make_imfit_input(cropped_frame, obj_name, filter_name, x_gal_center,y_gal_center,back_average)
    sersic_params = run_imfit(cropped_frame,mask_frame,imfit_input, obj_name, filter_name, back_average, \
        back_std, make_weight=1)

    pa_initial = sersic_params[0]
    e_initial = sersic_params[1]

    x_initial = sersic_params[10]
    y_initial = sersic_params[11]
    """
    ########## fit using the backbground value and constrain using initial pa and e
    """
    r_cut = r_cut_back
    cropped_frame, x_gal_center, y_gal_center = cut(main_data, ra, dec, r_cut , obj_name, \
        filter_name, overwrite=True, blur=0)
    while True :
        back_average, back_std, backs_all = estimate_frame_back(cropped_frame, r_cut_back_mask)
        if back_average > 0 and back_std > 0 :
            break

    backs = np.linspace(back_average-back_std,back_average+back_std,num=5)
    """
    #r_cut = r_cut0
    print ('+ MAIN fitting of the galaxy')
    print ('+++ cropping ')
    cropped_frame, x_gal_center, y_gal_center = cut(main_data, ra, dec, r_cut , obj_name, \
        filter_name, overwrite=True, blur=0)
    print (x_gal_center, y_gal_center)
    print ('+++ masking ')
    #mask_frame, masked_frame, check_frame = mask_stars(cropped_frame, ra, dec, obj_name, filter_name, \
    #   r_cut, r_cut_back_mask, q=1-e_initial, pa=pa_initial)
    mask_frame, masked_frame, check_frame, weight_frame = mask_stars(cropped_frame, ra, dec, obj_name, filter_name,r_cut)
    print ('+++ fitting ')

    #imfit_input = make_imfit_input(cropped_frame, obj_name, filter_name, x_initial,y_initial,back_average,\
    #pa=pa_initial,e=e_initial,constrain=1)
    #sersic_params = run_imfit(cropped_frame,mask_frame,imfit_input, obj_name, filter_name, back_average, \
    #    back_std, make_weight=0)

    os.system('rm fit.log')
    os.system('rm galfit.0*')
    fit_x1 = r_cut-r_cut_fit
    fit_y1 = r_cut-r_cut_fit
    fit_x2 = r_cut+r_cut_fit
    fit_y2 = r_cut+r_cut_fit

    make_galfit_feedme_file_sersic(cropped_frame,mask_frame,weight_frame,'None',\
        obj_name, ra, dec, 150, filter_name, fit_x1, fit_x2, fit_y1, fit_y2, \
        20, 0.5, 0, -1, 1, 0.001, zp, pix_size, x_gal_center, y_gal_center) 

    model_frame, res_frame = run_galfit_sersic (cropped_frame, obj_name, filter_name)

    mag_best, re_best, d_mag_best, d_re_best, n_best, d_n_best, X, Y, \
    x_best, y_best, pa_best, d_pa_best, axis_ratio_best, d_axis_ratio_best, sky_best, \
    sky_dx_best, sky_dy_best, d_sky_best, d_sky_dx_best, d_sky_dy_best = \
    read_galfit_sersic(cropped_frame,obj_name,filter_name)

    print (x_best, y_best)
    print (mag_best, re_best, n_best, pa_best, axis_ratio_best, sky_best)
    print (d_mag_best, d_re_best, d_n_best, d_pa_best, d_axis_ratio_best, d_sky_best)

    #re_best = sersic_params[4]
    #n_best = sersic_params[2]
    #PA_best = sersic_params[0]
    #q_best = 1.-sersic_params[1]
    #Ie_best = sersic_params[3]

    #x0 = sersic_params[10]
    #y0 = sersic_params[11]

    

    re_best_arcsec = re_best * pix_size
    re_best_kpc = re_best * pix_size / 2.0

    d_re_best_arcsec = d_re_best * pix_size
    d_re_best_kpc = d_re_best * pix_size / 2.0


    #if obj_name in ['DFX1','DF17'] :
    #    ORI = 0
    #else :
    #    ORI = get_header(main_data,keyword='ORIENTAT')
    pa_best_corr = pa_best

    #print (re_best,n_best,PA_best_corr,q_best,Ie_best)
    #print (PA_best,PA_best_corr)

    Ie_best = 99
    d_Ie_best = 99

    #c = ((pix_size)**2)
    #m_half_flux = mag_best+0.756
    #flux_within_re = c*10**((m_half_flux-zp)*-0.4)
    #Ie_best  = flux_within_re / (3.141592*re_best_kpc*re_best_kpc*axis_ratio_best)

    #os.system('cp *weight*.fits '+fit_dir)
    #os.system('mv bestfit_parameters_imfit.dat '+fit_dir)

    #os.system('mv *imfit_input.dat '+fit_dir)
    #os.system('mv *res.fits '+fit_dir)
    #os.system('mv *model.fits '+fit_dir)

    """
    print ('\n+ Using existing Sersic parameters for the galaxy : '+str(obj_name))
    sersic_cat = open(fit_dir+obj_name+'_'+filter_name+'_sersic_params.csv')
    for line in sersic_cat :
        line = line.split(',')
        re_best = float(line[0])
        n_best = float(line[1])
        PA_best = float(line[2])
        q_best = float(line[3])
        Ie_best = float(line[4])
        re_std = float(line[5])
        n_std = float(line[6])
        PA_std = float(line[7])
        q_std = float(line[8])
        Ie_std = float(line[9])
    """
    """
    for back in backs :
        continue
        imfit_input = make_imfit_input(cropped_frame, obj_name, filter_name, x_gal_center,y_gal_center,back)
        sersic_params = run_imfit(cropped_frame,mask_frame,imfit_input, obj_name, filter_name, back, back_std)
        #print (sersic_params)
        #if sersic_params[5] > 0 and sersic_params[6] > 0 and sersic_params[7] > 0 and \
        #sersic_params[8] > 0 and sersic_params[9] > 0  :
        Re.append(sersic_params[4])
        n.append(sersic_params[2])
        PA.append(sersic_params[0])
        q.append(1-sersic_params[1])
        Ie.append(sersic_params[3])

        e_Re.append(sersic_params[9])
        e_n.append(sersic_params[7])
        e_PA.append(sersic_params[5])
        e_q.append(1-sersic_params[6])
        e_Ie.append(sersic_params[8])
    """

    #os.system('rm *cropped*.fits')
    #os.system('rm *masked*.fits')
    #os.system('rm *imfit_input.dat')
    #os.system('rm *res.fits')
    #os.system('rm *model.fits')
    #os.system('rm *_imfit.dat')
    #os.system('rm *weight*.fits')
    """
    Re = np.array(Re)
    n = np.array(n)
    PA = np.array(PA)
    q = np.array(q)
    Ie = np.array(Ie)     

    e_Re = np.array(e_Re)
    e_n = np.array(e_n)  
    e_Ie = np.array(e_Ie)  

    re_std = np.std(Re-re_best)
    n_std = np.std(n-n_best)
    PA_std = np.std(PA-PA_best)
    q_std = np.std(q-q_best)
    Ie_std = np.std(Ie-Ie_best)

    re_std_arcsec = re_std * pix_size
    re_std_kpc = re_std* pix_size / 2.0
    """
    #print (re_best,n_best,PA_best_corr,q_best,Ie_best)
    #print (re_std,n_std,PA_std,q_std,Ie_std)

    #re_best_2 = np.average(Re,weights=1./(e_Re))
    #n_best_2 = np.average(n,weights=1./(e_n))
    #Ie_best_2 = np.average(Ie,weights=1./(e_Ie))

    #re_best_kpc_2 = re_best_2 * pix_size * 0.5


    sersic_cat = open(obj_name+'_'+filter_name+'_sersic_params.csv','w')
    #sersic_cat.write('Re,n,PA,q,Ie,err_Re,err_n,err_PA,err_q,err_Ie\n')
    sersic_cat.write(str(re_best_kpc)+','+str(n_best)+','+str(pa_best_corr)+','+str(axis_ratio_best)+','+str(mag_best)+','+\
        str(d_re_best_kpc)+','+str(d_n_best)+','+str(d_pa_best)+','+str(d_axis_ratio_best)+','+str(d_mag_best)+','+\
        str(x_best)+','+str(y_best))
    sersic_cat.close()

    if plotting == True :
        x_best = int(x_best+0.5)
        y_best = int(y_best+0.5)
        
        print (re_best,n_best,pa_best,axis_ratio_best,mag_best)
        
        ### plotting galaxy light profile + sersci fit

        plt.rc('axes', labelsize=36)
        plt.rc('xtick', labelsize=36) 
        plt.rc('ytick', labelsize=36) 

        plt.rc('axes', linewidth=1.7)

        plt.rcParams['xtick.major.width']=1.7
        plt.rcParams['xtick.major.size']=10
        plt.rcParams['xtick.minor.size']=6

        plt.rcParams['ytick.major.width']=1.7
        plt.rcParams['ytick.major.size']=10
        plt.rcParams['ytick.minor.size']=6
        plt.rc('text', usetex=True)

        frame = obj_name+'_'+filter_name+'_cropped.fits'
        #model_frame = fit_dir+obj_name+'_'+filter_name+'_model.fits'
        #res_frame = fit_dir+obj_name+'_'+filter_name+'_res.fits'
        mask_frame = obj_name+'_'+filter_name+'_masked.fits'
        mask_frame2 = obj_name+'_'+filter_name+'_masked+.fits'

        bin_size=5
        c = ((pix_size)**2)

        img = pyfits.open(frame)
        data = img[0].data

        fig, ax = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [2.0, 1]})
        fig1, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4,figsize=(12, 4), \
            gridspec_kw={'wspace':0.05, 'hspace':0}, squeeze=True,frameon=False) #figsize=(11, 4),
        fig.subplots_adjust(hspace=0)
        #fig1.subplots_adjust(left=-1, right=1, top=1, bottom=-1)
        #fig1.subplots_adjust(wspace=0.1)
        fig2, ax5 = plt.subplots(1,1,figsize=(8,8),frameon=False)

        img2 = pyfits.open(model_frame)
        model = img2[0].data
        img3 = get_fits(mask_frame) 
        mask0 = img3[0].data
        img4 = get_fits(res_frame) 
        res = img4[0].data
        img5 = get_fits(mask_frame2) 
        mask = img5[0].data

        ellipse = [x_best,y_best,re_best,axis_ratio_best*re_best,(90+pa_best)/360.*2.*np.pi]
        rs,fluxbins,errors,sky_level,sky_noise,sky_bin, sky_bins_values, area\
            = make_radial_profile(data,ellipse,mask=mask0,rad_stop=1000,binsize=bin_size,sky=True)
        fluxbins0 = fluxbins
        fluxbins = (fluxbins) - sky_best
        e = np.sqrt(errors**2 + sky_noise**2)

        rs_m,fluxbins_m,errors_m,sky_level_m,sky_noise_m,sky_bin_m, sky_bins_values_m, area_m\
            = make_radial_profile(model,ellipse,mask=mask0,rad_stop=1000,binsize=bin_size,sky=True)
        fluxbins0_m = fluxbins
        fluxbins_m = (fluxbins_m) - sky_best
        e_m = np.sqrt(errors_m**2 + sky_noise_m**2)

        rs_res,fluxbins_res,errors_res,sky_level_res,sky_noise_res,sky_bin_res, sky_bins_values_res, area_res\
            = make_radial_profile(res,ellipse,mask=mask0,rad_stop=1000,binsize=bin_size,sky=True)
        fluxbins0_res = fluxbins
        fluxbins_res = (fluxbins_res)
        e_res = np.sqrt(errors_res**2 + sky_noise_res**2)


        #####
 
        magbins = -2.5*np.log10((fluxbins)/c)+zp
        magbins_up = -2.5*np.log10((fluxbins+e)/c)+zp
        magbins_down = -2.5*np.log10((fluxbins-e)/c)+zp

        magbins_m = -2.5*np.log10((fluxbins_m)/c)+zp
        magbins_res = -2.5*np.log10((fluxbins_res)/c)+zp


        ###
        
        l = 'R$_e$ = ' + str(re_best_kpc)[:4] +' kpc, n = '+str(n_best)[:4]
        #l2 = 'R$_e$ = ' + str(re_best_kpc_2)[:4] +' kpc, n = '+str(n_best_2)[:4]
        #ax[1].text(2.0,2.4,l, fontsize=36, color='red',alpha=0.8)


        #f = sersic(rs,Ie_best_2,re_best_2,n_best_2,0)
        #m = -2.5*np.log10(f/c)+zp
        #ax[0].plot(rs*pix_size*0.5,m,color='green',markersize=3, lw=6, label=l,zorder=3)
        #ax[1].plot(rs*pix_size*0.5,magbins-m,color='green',markersize=1, lw=5)

        #f = sersic(rs,Ie_best,re_best,n_best,0)
        #m = -2.5*np.log10(f/c)+zp

        ax[0].plot(rs*pix_size*0.5,magbins,color='black',markersize=1, lw=5, label=obj_name+' (F'+str(filter_name)+'W)')
        ax[0].plot(rs*pix_size*0.5,magbins_up,'k--',markersize=1, lw=3)
        ax[0].plot(rs*pix_size*0.5,magbins_down,'k--',markersize=1, lw=3)

        ax[0].plot(rs_m*pix_size*0.5,magbins_m,color='red',markersize=3, lw=5, label=l,zorder=3)
        ax[1].plot(rs*pix_size*0.5,magbins-magbins_m,color='black',markersize=1, lw=5)
    
        """
        backs = np.random.normal(sky_best, d_sky_best, 2000)

        for i in range(len(backs)) :
            #continue
            fluxbins2 = (fluxbins0) - backs[i]
            #e = np.sqrt(errors**2 + sky_noise**2)
    
            magbins2 = -2.5*np.log10((fluxbins2)/c)+zp
            ax[0].plot(rs*pix_size*0.5,magbins2,'k-',markersize=1, lw=2, alpha=0.01, zorder=0)
            ax[1].plot(rs*pix_size*0.5,magbins2-m,color='k',markersize=1, alpha=0.01, lw=2, zorder=0)
        """

        ax[0].legend(loc='upper right', fontsize=30)
        ax[0].yaxis.labelpad = 26
        ax[1].plot([-2,20],[0,0],color='red',lw=5)

        ax[0].set_xlim([-0.001,0.5*r_cut_fit*0.05+0.05])
        #plt.xlim([0,10])
        ax[0].set_ylim(29.45,23.0)
        #ax[0].set_ylim(22.70,22.55)
        ax[1].set_ylim(-0.55,0.55)
        ax[1].set_xlabel('R [kpc]')
        ax[0].set_ylabel('$\mu$ (mag/arcsec$^2$)')
        ax[1].set_ylabel('$\Delta \mu$ (mag/arcsec$^2$)')

        ax[1].xaxis.set_minor_locator(ticker.FixedLocator(np.arange(0,20,0.5)))
        ax[1].xaxis.set_major_locator(ticker.FixedLocator(np.arange(0,20,2)))
        ax[1].xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        
        ax[0].yaxis.set_minor_locator(ticker.FixedLocator(np.arange(-10.0,30.,0.2)))
        ax[0].yaxis.set_major_locator(ticker.FixedLocator(np.arange(-10.0,30.,2.0)))
        ax[0].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

        ax[1].yaxis.set_major_locator(ticker.FixedLocator(np.arange(-10.0,30.,0.5)))
        ax[1].yaxis.set_minor_locator(ticker.FixedLocator(np.arange(-10.0,30.,0.05)))
        ax[1].yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
        
        #plt.show()
        ax[0].tick_params(which='both',direction='in')
        ax[1].tick_params(which='both',direction='in')
        #ax[2].tick_params(which='both',direction='in')
        plt.tight_layout()
        fig.subplots_adjust(hspace=0)
        ax[0].legend(loc='upper right', fontsize=28)
        fig.savefig(obj_name+'_'+filter_name+'_sersic_profile.png',dpi=150)

        res[res<0]=0

        #data[y0-100:y0+100,x0-100:x0+100] = 0
        min_ = np.nanmedian(model)-2*np.nanstd(model)
        max_ = np.nanmedian(model)+10*np.nanstd(model)

        res = sigma_clip(res,sigma=3,maxiters=3)
        min_2 = np.nanmedian(res)-3*np.nanstd(res)
        max_2 = np.nanmedian(res)+6*np.nanstd(res)



        mask[mask==0]=-99

        ax1.imshow(data,cmap='gist_gray',vmin=min_, vmax=max_)
        ax2.imshow(mask,cmap='gist_gray',vmin=min_, vmax=max_)
        ax3.imshow(model,cmap='gist_gray',vmin=min_, vmax=max_-(max_-min_)/2)
        ax4.imshow(res,cmap='gist_gray',vmin=min_2, vmax=max_2)

        ax5.imshow(data,cmap='gist_gray',vmin=min_, vmax=max_)

        ax1.set_title('Main Frame',color='black',fontsize=20)
        ax2.set_title('Mask',color='black',fontsize=20)
        ax3.set_title('Model',color='black',fontsize=20)
        ax4.set_title('Residuals',color='black',fontsize=20)

        ax1.plot(x_best,y_best,'r+',markersize=20)
        ax2.plot(x_best,y_best,'r+',markersize=20)
        ax3.plot(x_best,y_best,'r+',markersize=20)
        ax4.plot(x_best,y_best,'r+',markersize=20)



        ax1.text(60,r_cut*2-100,obj_name,color='red',fontsize=20)
        ax1.arrow(60,50,2.0*2/0.05,0,head_width=0, head_length=20, color='gold',lw=2)
        ax1.text(180,40,'2 kpc',fontsize=14, color='gold')
        ax1.arrow(2*r_cut-50,50,0,100,color='gold',head_width=10, head_length=10,lw=2)
        ax1.text(2*r_cut-200+90,150,'N',fontsize=14, color='gold')
        ax1.arrow(2*r_cut-50,50,-100,0,color='gold',head_width=10, head_length=10,lw=2)
        ax1.text(2*r_cut-100-75,75,'E',fontsize=14, color='gold')

        ax5.text(60,r_cut*2-100,obj_name,color='red',fontsize=40)
        ax5.arrow(60,50,2.0*2/0.05,0,head_width=0, head_length=32, color='gold',lw=4)
        ax5.text(180,40,'2 kpc',fontsize=32, color='gold')
        ax5.arrow(2*r_cut-50,50,0,100,color='gold',head_width=10, head_length=20,lw=4)
        ax5.text(2*r_cut-200+110,150,'N',fontsize=32, color='gold')
        ax5.arrow(2*r_cut-50,50,-100,0,color='gold',head_width=10, head_length=20,lw=4)
        ax5.text(2*r_cut-100-75,75,'E',fontsize=32, color='gold')



        ax1.axis('off')
        ax1.invert_yaxis()
        ax2.axis('off')
        ax2.invert_yaxis()
        ax3.axis('off')
        ax3.invert_yaxis()
        ax4.axis('off')
        ax4.invert_yaxis()

        ax5.axis('off')
        ax5.invert_yaxis()

        #plt.axis("off")
        #plt.subplots_adjust(hspace=0, wspace=0)
        #plt.tight_layout()
        #fig1.tight_layout()

        fig1.savefig(obj_name+'_'+filter_name+'_sersic_model.png',bbox_inches='tight', pad_inches = 0, dpi=150)
        fig2.savefig(obj_name+'_'+filter_name+'_fancy_frame.png',bbox_inches='tight', pad_inches = 0, dpi=200)
        plt.close()       

        os.system('mv *cropped*.fits '+fit_dir)
        os.system('mv *masked*.fits '+fit_dir)
        os.system('mv *masked+*.fits '+fit_dir)
        os.system('mv *median*.fits '+fit_dir)
        os.system('mv *check*.fits '+fit_dir)
        os.system('mv *galfit*.fits '+fit_dir)
        os.system('mv *galfit.conf '+fit_dir)
        os.system('mv fit*log '+fit_dir)  
        os.system('mv *_sersic_params.csv '+fit_dir)  

    return re_best,n_best,pa_best_corr,axis_ratio_best,Ie_best,d_re_best,d_n_best,d_pa_best,d_axis_ratio_best,d_Ie_best 


