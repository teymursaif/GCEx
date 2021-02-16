
import os, sys
import math
import numpy as np
from pandas.plotting import scatter_matrix
import pandas as pd
import pyfits
import matplotlib
from sklearn import neighbors, datasets
import random
from astropy.io import ascii
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy.optimize import curve_fit
from scipy.integrate import quad
from astropy.stats import sigma_clip
from matplotlib.ticker import StrMethodFormatter
import matplotlib.ticker as ticker
import csv
from matplotlib.patches import Ellipse
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import Normalize
import matplotlib.colors as colors
from astropy.visualization import make_lupton_rgb
from astropy.visualization import simple_norm
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
from matplotlib import patches
from astropy.stats import sigma_clip
from functions import *
from astropy.wcs import WCS
from astropy import wcs


plt.rc('axes', labelsize=32)
plt.rc('xtick', labelsize=24) 
plt.rc('ytick', labelsize=24) 
plt.rc('axes', linewidth=1.7)

plt.rcParams['xtick.major.width']=1.7
#plt.rcParams['xtick.minor.width']=1.2
plt.rcParams['xtick.major.size']=10
plt.rcParams['xtick.minor.size']=6

plt.rcParams['ytick.major.width']=1.7
#plt.rcParams['ytick.minor.width']=1.2
plt.rcParams['ytick.major.size']=10
plt.rcParams['ytick.minor.size']=6

plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 0.8
plt.rcParams["legend.markerscale"] = 2.0

plt.rcParams['hatch.linewidth'] = 2.0


def make_fancy_frame(filename,ra,dec,objectname,filtername,img_dir,frame_size=300) :
    output_frame, x_gal_center,y_gal_center = cut(filename,ra,dec,frame_size,objectname=objectname,filtername=filtername,overwrite=True)
    main = pyfits.open(output_frame)
    image = main[0].data
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    image0 = image[abs(image)>0]
    image0 = sigma_clip(image0,3,3)
    min_ = np.nanmedian(image0)-0.5*np.nanstd(image0)
    max_ = np.nanmedian(image0)+5*np.nanstd(image0)
    ax.imshow(image,cmap='gist_gray', vmin=min_, vmax=max_) #LogNorm
    #ax.axis('off')
    ax.invert_yaxis()
    #fig.tight_layout()
    #[0., 0., 1., 1.]
    fig.savefig(objectname+'_'+filtername+'.png',dpi=150)
    plt.close() 
    os.system('mv '+objectname+'_'+filtername+'.png '+img_dir)
    os.system('mv '+objectname+'_'+filtername+'_cropped.fits '+img_dir)


# ---------------------------------------------------------------------

# ---------------------------------------------------------------------


def plot_comp_mag_diagram(cats,labels,x,y,xl,yl,xrange,yrange,plot_title,out_plot,colors, ax=None, save_fig=True) :
    if save_fig == True :
        fig0, ax0 = plt.subplots(figsize=(8,8))
        ax = ax0
    elif save_fig == False :
        donothing = 1

    for i in range(len(cats)) :
        main = pyfits.open(cats[i])
        data = main[1].data
        n_objects = data.shape[0]
        X = data[x]
        Y = data[y]
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)

        if 'c48_' in x :
            ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(xrange[0],xrange[1]+0.01,0.2)))
            ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(yrange[1],yrange[0]+0.01,1)))
            ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(xrange[0],xrange[1]+0.01,0.05)))
            ax.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(yrange[1],yrange[0]+0.01,0.2)))
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

        if 'Z_' in x :
            ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(xrange[0],xrange[1]+0.01,0.2)))
            ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(yrange[1],yrange[0]+0.01,1)))
            ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(xrange[0],xrange[1]+0.01,0.05)))
            ax.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(yrange[1],yrange[0]+0.01,0.2)))
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))


        bp = ax.scatter(X,Y,20,color=colors[i],marker='o',label=labels[i],zorder=0, alpha=0.5)
        
    if save_fig == True :
        ax.set_title(plot_title,fontsize=32)
        ax.legend(loc='upper left',fontsize=24)
        plt.tight_layout()
        plt.tick_params(which='both',direction='in')
        plt.savefig(out_plot,dpi=150)
    
    ax.tick_params(which='both',direction='in')
    return bp


# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def plot_radial_profile(main_psf_file,psf_frames,radii,sex_dir,plot_title,out_plot):
    fig, ax = plt.subplots(figsize=(9,6))
    radii_str = ",".join(str(x) for x in radii)
    #print (radii_str)
    for psf_frame in psf_frames :
        command = 'sextractor '+psf_frame+' -c '+str(sex_dir)+'default.sex -CATALOG_NAME '+'temp.sex_cat.fits '+ \
        '-PARAMETERS_NAME '+str(sex_dir)+'default.param -DETECT_MINAREA 10 -DETECT_THRESH 3.0 -ANALYSIS_THRESH 3.0 ' + \
        '-DEBLEND_NTHRESH 1 -DEBLEND_MINCONT 1 -PHOT_APERTURES ' + str(radii_str) + ' ' + \
        '-MAG_ZEROPOINT ' +str(0) + ' -BACK_TYPE MANUAL -BACK_VALUE 0.0 ' + \
        '-FILTER_NAME '+str(sex_dir)+ 'tophat_2.0_3x3.conv -STARNNW_NAME '+str(sex_dir)+'default.nnw -CHECKIMAGE_TYPE '+'NONE -VERBOSE_TYPE QUIET'
        #print (command)
        os.system(command)
        main = pyfits.open('temp.sex_cat.fits') 
        data = main[2].data
        mags = data['MAG_APER']
        if (len(mags) == 0) :
            continue
        c = mags[0,3] - mags[0,7] 
        mags = mags[0,:] - mags[0,7]
        radii = np.array(radii)
        #print (mags, len(radii), len(mags))
        if c < 0.45 and c > 0.25 and len(radii) == len(mags):
            ax.plot(radii[2:]/2,mags[2:],color='grey',alpha=0.4,lw=2)
    if c < 0.45 and c > 0.25 and len(radii) == len(mags):
        #print (radii)
        #print (mags)
        ax.plot(radii[2:]/2,mags[2:],color='grey', label='selected sources',alpha=0.4,lw=2)

    for psf_frame in [main_psf_file] :
        command = 'sextractor '+psf_frame+' -c '+str(sex_dir)+'default.sex -CATALOG_NAME '+'temp.sex_cat.fits '+ \
        '-PARAMETERS_NAME '+str(sex_dir)+'default.param -DETECT_MINAREA 10 -DETECT_THRESH 3.0 -ANALYSIS_THRESH 3.0 ' + \
        '-DEBLEND_NTHRESH 1 -DEBLEND_MINCONT 1 -PHOT_APERTURES ' + str(radii_str) + ' ' + \
        '-MAG_ZEROPOINT ' +str(0) + ' -BACK_TYPE MANUAL -BACK_VALUE 0.0 ' + \
        '-FILTER_NAME '+str(sex_dir)+ 'tophat_2.0_3x3.conv -STARNNW_NAME '+str(sex_dir)+'default.nnw -CHECKIMAGE_TYPE '+'NONE -VERBOSE_TYPE QUIET'
        #print (command)
        os.system(command)
        main = pyfits.open('temp.sex_cat.fits') 
        data = main[2].data
        mags = data['MAG_APER']
        if (len(mags) == 0) :
            continue
        mags = mags[0,:] - mags[0,7]
        radii = np.array(radii)
        #print (mags, len(radii), len(mags))
        ax.plot(radii[2:]/2,mags[2:],color='red',label='reconstructed PSF',lw=4)

    ax.set_xlim([1.8,15.2])
    ax.set_ylim([-0.22,0.82])
    ax.set_xlabel('radius [pixel]')
    ax.set_ylabel('$\Delta$m$_8$ [mag]')

    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0,20,2)))
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(-0.4,1,0.2)))

    ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(0,20,1)))
    ax.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(-0.4,1.0,0.05)))

    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

    ax.legend(loc='upper right',fontsize=24)
    ax.set_title(plot_title,fontsize=32)

    plt.tight_layout()
    plt.tick_params(which='both',direction='in')
    plt.savefig(out_plot,dpi=150)
    os.system('rm temp.sex_cat.fits')

def make_plot_comp_Z_mag_diagrams(main_cat,art_cat,selected_cat,udg,filter,out_plot, mags, medians1, medians2, stds, selection_mode):

    fig, ((ax1, ax2),(ax3, ax4))  = plt.subplots(2, 2, sharex='col', sharey='row', \
        gridspec_kw={'hspace': 0.05, 'wspace': 0.05}, figsize=(16, 16))

    plot_comp_mag_diagram([main_cat,art_cat],['all sources','artificial stars'],\
        'c48_'+filter,'MAG_AUTO_'+filter,None,'m$_{'+filter+'}$',[0,1.03],[28.5,22],\
        None,None,['grey','gold'], ax=ax1, save_fig=False)

    plot_comp_mag_diagram([main_cat,selected_cat],['all sources','selected sources'],\
        'c48_'+filter,'MAG_AUTO_'+filter,'c$_{4-8}$','m$_{'+filter+'}$',[0,1.03],[28.5,22],\
        None,None,['grey','red'], ax=ax3, save_fig=False)

    ###

    plot_comp_mag_diagram([main_cat,art_cat],['all sources','artificial stars'],\
        'Z_'+filter,'MAG_AUTO_'+filter,None,None,[0,1.05],[28.5,22],\
        None,None,['grey','gold'], ax=ax2, save_fig=False)

    plot_comp_mag_diagram([main_cat,selected_cat],['all sources','selected sources'],\
        'Z_'+filter,'MAG_AUTO_'+filter,'Z',None,[0,1.05],[28.5,22],\
        None,None,['grey','red'], ax=ax4, save_fig=False)
    ###

    if selection_mode == 'UNRESOLVED' :
        ax2.plot(medians1+3*stds, mags, 'k--', lw=3)
        ax4.plot(medians1+3*stds, mags, 'k--', lw=3)
        ax2.text(0.1, 27, 'selection\nregion', color='black', fontsize=20)

    elif selection_mode == 'RESOLVED' :
        ax2.plot(medians1+3*stds, mags, 'k--', lw=3)
        ax4.plot(medians1+3*stds, mags, 'k--', lw=3)
        #ax2.text(0.1, 27, 'selection\nregion', color='black', fontsize=20)

        ax2.plot(medians2+3*stds, mags, 'k--', lw=3)
        ax4.plot(medians2+3*stds, mags, 'k--', lw=3)
        ax2.plot(medians1-3*stds, mags, 'k--', lw=3)
        ax4.plot(medians1-3*stds, mags, 'k--', lw=3)
        ax2.text(0.4, 27, 'selection\nregion', color='black', fontsize=20)

    #ax1.legend(loc='upper left',fontsize=16,ncol=1,markerscale=2)
    #ax4.legend(loc='upper left',fontsize=16,ncol=1,markerscale=2)
    #plt.title('UDG '+udg+' (F'+filter+'W)')
    fig.savefig(out_plot,dpi=150, bbox_inches='tight')
    plt.close()


def draw_elipse(a,b,pa,color,style,label,frame_size,px) :
    
    print (label,a,b)
    t = np.linspace(0,2*np.pi,1000)
    ell = np.array([a/px*np.cos(t),b/px*np.sin(t)])
    r_rot = np.array([[np.cos(pa),-np.sin(pa)],[np.sin(pa),np.cos(pa)]])
    ell_rot = np.zeros((2,ell.shape[1]))
    for i in range(ell.shape[1]) :
        ell_rot[:,i] = np.dot(r_rot,ell[:,i])
    plt.plot(frame_size+ell_rot[0,:], frame_size+0+ell_rot[1,:],color=color,linestyle=style,label=label,lw=5)



def plot_gal_gc(main_frame,ra,dec,udg,filter,crop_size,pix,scale,gc_cat,gc_ra_param,gc_dec_param,gc_mag_param,gc_mag_limit,\
    re_best_kpc,pa_best,axis_ratio_best,out_plot) :

    output_frame, x_gal_center,y_gal_center = cut(main_frame,ra,dec,crop_size,objectname=udg,filtername=filter,overwrite=True)
    main = pyfits.open(output_frame)
    image = main[0].data
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    image0 = image[abs(image)>0]
    image0 = sigma_clip(image0,3,3)
    min_ = np.nanmedian(image0)-1*np.nanstd(image0)
    max_ = np.nanmedian(image0)+5*np.nanstd(image0)
    ax.imshow(image,cmap='gist_gray', vmin=min_, vmax=max_) #LogNorm
    #ax.axis('off')
    ax.invert_yaxis()
    pa_best = math.pi/2. + pa_best*math.pi/180
    draw_elipse(re_best_kpc/scale,re_best_kpc/scale*axis_ratio_best,pa_best,'r','--','1 R$_{e}$',crop_size,pix)

    #if crop_size < 500 :
    ax.text(crop_size/5,crop_size*2-crop_size/4,udg,color='red',fontsize=48)
    ax.arrow(crop_size/5,crop_size/8,4/scale/pix,0,head_width=0, head_length=20, color='gold',lw=2)
    ax.text(crop_size/5+5.1/scale/pix,crop_size/9.1,'4 kpc',fontsize=32, color='gold')

    ax.arrow(2*crop_size-crop_size/5,crop_size/8,0,crop_size/4,color='gold',head_width=crop_size/40, head_length=crop_size/40,lw=2)
    ax.text(2*crop_size-crop_size/3.4,crop_size/2.6,'N',fontsize=32, color='gold')

    ax.arrow(2*crop_size-crop_size/5,crop_size/8,-1*crop_size/4,0,color='gold',head_width=crop_size/40, head_length=crop_size/40,lw=2)
    ax.text(2*crop_size-crop_size/2,crop_size/6,'E',fontsize=32, color='gold')


    points_cat = get_fits(gc_cat)
    points = points_cat[1].data
    ra = points[gc_ra_param]
    dec = points[gc_dec_param]

    w=WCS(output_frame)
    x,y = w.all_world2pix(ra, dec,0)
    mask = ((abs(x-crop_size)<0.98*crop_size) & (abs(y-crop_size)<0.98*crop_size) & (points[gc_mag_param]<gc_mag_limit))
    x = x[mask]
    y = y[mask]
    ax.scatter(x,y,250,edgecolor='orange',marker='o',facecolor='none', lw=2)

    fig.savefig(out_plot,dpi=150, bbox_inches='tight')
    plt.close()


def plot_completeness(cats,filter,out_plot,colors,labels) :
    i = -1
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for cat in cats:
        i = i+1
        comp_cat = open(cat,'r')
        j = 0
        mags = list()
        comps = list()
        for line in comp_cat :
            j = j+1
            if j == 1 :
                #ax.scatter(-99,-99,color=colors[i],label=labels[i])
                continue
            line = line.split(',')
            #print (line)
            mag = float(line[0])
            comp = float(line[1])
            mags.append(mag)
            comps.append(comp)

        if colors[i] == 'auto' and '475' in labels[i] :
            colors[i] = 'blue'
        if colors[i] == 'auto' and '606' in labels[i] :
            colors[i] = 'green'
        if colors[i] == 'auto' and '814' in labels[i] :
            colors[i] = 'red'

        comps = gaussian_filter(np.array(comps),sigma=4)
        ax.plot(mags,comps,color=colors[i],label=labels[i],lw=5)

        ax.set_xlim([21.8,29.2])
        ax.set_ylim([0,1])
        ax.set_xlabel('m$_{'+filter+'}$ [mag]')
        ax.set_ylabel('completeness')

        ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0,30,2)))
        ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(0,2,0.2)))

        ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(0,30,1)))
        ax.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(0,2,0.05)))

        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

        ax.legend(loc='lower left',fontsize=24)
        plt.tight_layout()
        plt.tick_params(which='both',direction='in')

    fig.savefig(out_plot,dpi=150, bbox_inches='tight')
    plt.close()


def make_plot_comp_mag_diagrams(main_cat,art_cat,selected_cat,udg,filter,out_plot, mags, medians1, medians2, stds1, stds2, selection_mode):

    fig, (ax1, ax2)  = plt.subplots(1, 2, sharex='col', sharey='row', \
        gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize=(16, 8))

    plot_comp_mag_diagram([main_cat,art_cat],['all sources','artificial\nstars'],\
        'c48_'+filter,'mag_'+filter,None,'m$_{'+filter+'}$',[-0.2,1.05],[27.5,22],\
        None,None,['grey','gold'], ax=ax1, save_fig=False)

    plot_comp_mag_diagram([main_cat,selected_cat],['all sources','selected\nsources'],\
        'c48_'+filter,'mag_'+filter,'c$_{4-8}$',None,[-0.2,1.05],[27.5,22],\
        None,None,['grey','red'], ax=ax2, save_fig=False)

    ###

    if selection_mode == 'UNRESOLVED' :
        #ax1.plot(medians1,mags,'k-',lw=3)
        ax1.plot(medians1-4*stds1-0.05, mags, 'k--', lw=3)
        ax2.plot(medians1-4*stds1-0.05, mags, 'k--', lw=3)
        ax1.plot(medians2+4*stds2+0.05, mags, 'k--', lw=3)
        ax2.plot(medians2+4*stds2+0.05, mags, 'k--', lw=3)
        #ax2.text(0.1, 27, 'selection\nregion', color='black', fontsize=20)
    """
    elif selection_mode == 'RESOLVED' :
        ax2.plot(medians1+3*stds, mags, 'k--', lw=3)
        ax4.plot(medians1+3*stds, mags, 'k--', lw=3)
        #ax2.text(0.1, 27, 'selection\nregion', color='black', fontsize=20)

        ax2.plot(medians2+3*stds, mags, 'k--', lw=3)
        ax4.plot(medians2+3*stds, mags, 'k--', lw=3)
        ax2.plot(medians1-3*stds, mags, 'k--', lw=3)
        ax4.plot(medians1-3*stds, mags, 'k--', lw=3)
        ax2.text(0.4, 27, 'selection\nregion', color='black', fontsize=20)
    """
    ax1.legend(loc='upper left',fontsize=18,ncol=1,markerscale=2)
    ax2.legend(loc='upper left',fontsize=18,ncol=1,markerscale=2)
    #plt.title('UDG '+udg+' (F'+filter+'W)')
    fig.savefig(out_plot,dpi=150, bbox_inches='tight')
    plt.close()