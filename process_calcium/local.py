import sys
sys.path.append("/home/cat/code/widefield/") # Adds higher directory to python modules path.


import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
import torch
import time
import warnings
warnings.filterwarnings("ignore")


from locanmf import LocaNMF, postprocess
from locanmf import analysis_fig4 


#
import os

# pre process data module
from ProcessCalcium import ProcessCalcium
from locanmf import LocaNMF, postprocess
from locanmf import analysis_fig4 


###########################################################################
##### PROCESS [CA] DATA (EVENT TRIGGERED MAPS): ROI OR PCA TIME SERIES ####
###########################################################################
'''  Compute calcium activlearnity in ROIs selected (35) for 
     lever pull actiivty
     
     or just use PCA time courses 
'''

# Processing flags
parallel = False  # THIS DOESN"T REALLY WORK TOO MUCH MEMORY...
n_cores = 3

# select animal names
names = ['IA1']# ,'IA2','IA3','IJ1','IJ2','AQ2'] # 30HZ DATASETS
#names = ['AI3','AK4','AK5','AR4','BA2']

#names = ['AK4','AK5','AR4','BA2']


# data_dir = '/media/cat/4TBSSD/yuki/'
#data_dir = '/media/cat/4TBSATA/yuki/'
data_dir = '/home/cat/data/'
#data_dir = '/home/cat/ssd/'


# window to compute
n_sec_window = 15
lockout_window = 10   # no of seconds of previous movement / lever pull lockout
recompute = False      # overwrite previously generated data

# save PCA vs. ROI data; if selected, each dataset will be reduced to the PCs rquired to get to the explained var value
pca_etm = False
pca_explained_var_val = 0.95

################################
etm = ProcessCalcium()
etm.main_dir = data_dir
etm.export_blue_light_times = False
#
etm.random_events_lockout = 3.0  # minimum number of seconds difference between a rewarded pull and a random/control 
etm.n_sec_window = n_sec_window 
etm.recompute = recompute
etm.pca_fixed_comps = None  # fixed numer of components overrides explained_variance amount above

#
etm.low_cut = 0.1
etm.high_cut = 6.0
etm.img_rate = 30

# 
etm.remove_shift = True
etm.whole_stack = True
etm.verbose = True
#
sessions = ['all']

#features = ['left_paw','right_paw','jaw']
features = ['whole_stack']
for feature in features:
    etm.features = feature
    etm.feature_name = feature
    etm.feature_quiescence = 3    # number of seconds of no-movement prior to feature movement ;not applicable to code-04
                                    # this value is computed in generate_movements_quiescence_periods notebook;

    # 
    #etm.skip_to = 'Mar14_'  # flag to pickup reprocessing at some point; only used if overwrite flag is set to True and need to partially restart
    etm.skip_to = None  # 

    #
    for session in sessions:
        # 
        etm.sessions = session

        # 
        etm.generate_etm(names,
                         n_sec_window,
                         lockout_window,
                         recompute,
                         pca_etm,
                         pca_explained_var_val,
                         parallel,
                         n_cores,
                         )     
        
        # run locanmf on the datasets
    
