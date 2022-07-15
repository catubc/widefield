import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec
import matplotlib.patches as mpatches
from sklearn import datasets, linear_model

import parmap
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
import pycorrelate
from tqdm import tqdm, trange


from scipy.spatial import ConvexHull
from tqdm import trange
from scipy.spatial import cKDTree
import pandas as pd
import scipy
from statsmodels.stats.multitest import multipletests

import glob2
import scipy
from tqdm import trange
from statsmodels.stats.multitest import multipletests
from scipy.optimize import curve_fit

from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
         
                
#
def plot_number_04_all_mice_all_sessions_single_plot(main_dir,
                                                    tag,
                                                    root_dirs,
                                                    fig=None):

    clrs = ['grey','blue','red','green','magenta','pink']
    #
    if fig is None:
        fig=plt.figure(figsize=(10,10))

    Yfit = []
    Ydata = []
    X_array = []
    ctr=0
    for root_dir in root_dirs:
        ax=plt.subplot(3,2,ctr+1)
        # run this function to compute the # of trials
        compute_single_mouse_all_sessions_number_of_04_02_trials(main_dir+'/'+root_dir)

		#
        data = np.load(main_dir+'/'+root_dir+'/tif_files/no_trials.npz')
        abs_dates = data['abs_dates']
        y=data['n_04']

        # plot individual mice
        x = np.arange(y.shape[0])
        x = x/x.shape[0]

        #if tag=='04':
        plt.scatter(x, y,
                 #linewidth=3,
                    s=200,
                    #c=clrs[ctr],
                    c='grey',
                    alpha=.85)
        #
        #corr = np.corrcoef(x, y)[0,1]
        corr = scipy.stats.pearsonr(x,y)

        print (ctr, " cor: ", corr)

        coef = np.polyfit(x,y,1)
        poly1d_fn = np.poly1d(coef)

        # poly1d_fn is now a function which takes in x and returns an estimate for y
        Y = poly1d_fn(x)
        plt.plot(x, Y, linewidth=5,
                 c='black',
                 label= "pcorr: "+str(round(corr[0],4))+"\n"+
                        "pval: "+str(round(corr[1],4)))

        plt.xlim(x[0],x[-1])
        plt.xticks([])
        plt.yticks(fontsize=20)
        plt.ylim(bottom=0)
        plt.legend(fontsize=14)
        plt.title(root_dir)


        ctr+=1

def compute_single_mouse_all_sessions_number_of_04_02_trials(root_dir):
    #root_dirs = os.listdir(main_dir)
    #ctr_animal=0
    #fig=plt.figure()

   # for root_dir in root_dirs:

    tif_root = root_dir + '/tif_files/'
    fname_out = tif_root + "/no_trials.npz"
    
    #print (fname_out)
    if os.path.exists(fname_out)==False:
        tif_files= os.listdir(tif_root)

        # make sure this order makes sense to not loop over wrong way...
        month_names = ['July','Aug','Sep','Oct','Nov','Dec','Jan','Feb', 'Mar','Apr','May','June']

        abs_dates=[]
        n_04=[]
        n_02=[]
        reward_starts = []
        nonreward_starts = []
        for tif_file in tif_files:
            text = tif_root+ '/'+tif_file+'/*abstimes.npy'

            ctr=0
            month = None
            while True:
                # for name in month_names
                if month_names[ctr] in text:
                    month = ctr
                    break
                ctr+=1
            idx = tif_file.find(month_names[ctr])
            day = tif_file[idx+3:idx+5]
            day = day.replace("_",'')
            day = day.replace('a','')
            day = day.replace('p','')
            day = day.replace('m','')

            day = int(day)

            try:
                fname = glob2.glob(tif_root+ '/'+tif_file+'/*abstimes.npy')[0]
                abstimes = np.load(fname)
                fname = glob2.glob(tif_root+ '/'+tif_file+'/*abspositions.npy')[0]
                abspositions = np.load(fname)
                fname = glob2.glob(tif_root+ '/'+tif_file+'/*abscodes.npy')[0]
                abscodes = np.load(fname)
            except:

                print ("data missing, skipping...")

                continue
                
                # find where 04 - rewarded pulls start;
            (starts_04, starts_04_idx,starts_02, starts_02_idx) = find_code04_starts2(abscodes, abstimes, abspositions)

            reward_starts.append(starts_04)
            nonreward_starts.append(starts_02)

            abs_dates.append(month*31+day)
            n_04.append(starts_04.shape[0])
            n_02.append(starts_02.shape[0])

        abs_dates=np.array(abs_dates)
        n_04=np.array(n_04)
        n_02=np.array(n_02)
        reward_starts=np.array(reward_starts)
        nonreward_starts=np.array(nonreward_starts)

        idx = np.argsort(abs_dates)
        abs_dates = abs_dates[idx]
        abs_dates=abs_dates-abs_dates[0]
        n_04=n_04[idx]
        n_02=n_02[idx]
        reward_starts=reward_starts[idx]
        nonreward_starts=nonreward_starts[idx]

        #ax=plt.subplot(3,4,ctr_animal+1)

        # plot scatter
        #plt.scatter(abs_dates, n_04, c='black',alpha=.5)
        #plt.scatter(abs_dates, n_02, c='red',alpha=.5)

        win1 = 10
        n_04_smooth = np.convolve(n_04, np.ones(win1)/win1,mode='same')
        n_02_smooth = np.convolve(n_02, np.ones(win1)/win1,mode='same')

        
        np.savez(fname_out,
                abs_dates = abs_dates,
                n_04=n_04,
                n_02=n_02,
                n_04_smooth=n_04_smooth,
                n_02_smooth=n_02_smooth,
                reward_starts=reward_starts,
                nonreward_starts=nonreward_starts)

#        