
# 
# 
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


##############################################
##############################################
##############################################

# 
root_dir = '/media/cat/4TBSSD/yuki/'
root_dir = '/home/cat/data/'

#
animal_ids = ['IA1']#,'IA2','IA3',"IJ1",'IJ2','AQ2']
session = 'IA1pm_Feb1_30Hz'   # all is an option aslo

# 
for animal_id in animal_ids:
    loca = analysis_fig4.LocaNMFClass(root_dir, animal_id, session)

    #
    # loca.run_loca() # <---- this runs locanmf for segments centred on lever pulls + random data

    #
    loca.run_loca_whole_session() # <---- this runs locanmf on whole stack of data...
