
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
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
from matplotlib import patches

from fit_galaxy import cut
from astropy.stats import sigma_clip

def make_fancy_frame(filename,ra,dec,objectname,filtername,img_dir) :
    output_frame, x_gal_center,y_gal_center = \
        cut(filename,ra,dec,300,objectname=objectname,filtername=filtername,overwrite=True)
    main = pyfits.open(output_frame)
    image = main[0].data
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), frameon=False)
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
    #fig.tight_layout()
    #[0., 0., 1., 1.]
    fig.savefig(objectname+'_'+filtername+'.png',dpi=150)
    plt.close() 
    os.system('mv '+objectname+'_'+filtername+'.png '+img_dir)
    os.system('mv '+objectname+'_'+filtername+'_cropped.fits '+img_dir)



