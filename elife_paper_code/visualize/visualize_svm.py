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

class Visualize():

    def __init__(self):

        self.clr_ctr = 0

        #
        self.animal_ids = ['IA1','IA2','IA3','IJ1','IJ2','AQ2']

        #
        self.labels = ["M1", "M2", "M3", "M4","M5",'M6']
        
        #
        self.manuscript_names = ['M1','M2','M3','M4','M5','M6']

        #
        self.colors = ['black','blue','red','green','magenta','pink','cyan']

        #
        self.feature_codes = [
                    'left_paw',          # 0
                    'right_paw',         # 1
                    'nose',              # 2
                    'jaw',               # 3
                    'right_ear',         # 4
                    'tongue',            # 5
                    # 'lever',           # 6
                    # 'quiescence',      # 7
                    # 'code_04',         # 8
                    # 'code_04_lockout'  # 9
                    ]

        #
        self.plot_legend = False
        
        #
        self.n_colors = [10,19,23,17,20,48]

        #
        self.linewidth = 4

        #
        self.filter=False

        #
        self.cbar_offset = 0
        
        #
        self.lockout = False
        
        #
        self.lockout_window = 10
        
        #
        self.window = 30
        
        #
        self.alpha = 1
        
        #        
        self.smooth_window = 30
        
        #
        self.significance = 0.05
		
		#
        self.cbar_thick = 0.05

		#
        self.code = 'code_04'                  # uses all ROIs/PCA to decode

		#
        self.random_flag = False  
        
        #
        self.xvalidation = 10

		#
        self.linewidth=10
        
        #
        self.edt_offset_y = .1
        
        #
        self.edt_offset_x = 3
        
        #
        self.cbar = False
        
        #
        self.title_offset = -0.05
        
        #
        self.cbar_offset=0
        
        # 
        self.pca_var = 0.95
        
        # 
        self.pca_flag = True
        
        #
        self.sliding_window = 30
        
        #
        self.imaging_rate = 30
        
        # 
        self.min_trials = 10
        
        # 
        self.fname = None
        
        # 
        self.pickle = False
        
        #
        self.compute_roi=False

        #
        self.shift = 0
        
        # 
        self.ideal_window_flag = False
        
        # 
        self.dlc_offset_flag = False







        #self.imaging_rate = 30.


    def load_data(self, fname):

        self.data = np.load(fname)



    def format_plot(self, ax):
        ''' Formats plots for decision choice with 50% and 0 lines
        '''
       # meta data
        try:
            xlims = [self.xlim[0],self.xlim[1]]
        except:
            xlims = [-self.window+1,0]

        ylims = [0.4,1.0]
        ax.plot([0,0],
                 [ylims[0],ylims[1]],
                 '--',
                linewidth=3,
                 color='black',
                alpha=.5)

        ax.plot([xlims[0],xlims[1]],
                 [0.5,0.5],
                 '--',
                linewidth=3,
                 color='black',
                alpha=.5)

        ax.set_xlim(xlims[0],xlims[1])
        ax.set_ylim(0.4,1.0)



    def format_plot2(self, ax):
        ''' Format plots for time prediction
        '''
       # meta data
        xlims = [-10,0]
        ylims = [0.0,1.0]
        ax.plot([0,0],
                 [ylims[0],ylims[1]],
                 '--',
                linewidth=3,
                 color='black',
                alpha=.5)

        ax.plot([xlims[0],xlims[1]],
                 [0.1,0.1],
                 '--',
                linewidth=3,
                 color='black',
                alpha=.5)

        ax.set_xlim(xlims[0],xlims[1])
        ax.set_ylim(ylims[0],ylims[1])
       # plt.legend(fontsize=20)


    def get_fname(self): # load ordered sessions from file

        data = np.load(os.path.join(self.main_dir, self.animal_id,'tif_files.npy'))

        #
        self.sessions = []
        for k in range(len(data)):
            self.sessions.append(os.path.split(data[k])[1].replace('.tif',''))

        #
        self.session = None
        for k in range(len(self.sessions)):
            #print (self.session_id, self.sessions[k])
            if str(self.session_id) in str(self.sessions[k]):
                self.session = self.sessions[k]
                break

        #
        if self.session is None:
            print (" COULDN't FIND SESSION...")
            self.fname = None
            return


        # select data with or without lockout
        prefix1 = ''
        if self.lockout:
            prefix1 = '_lockout_'+str(self.lockout_window)+"sec"

        # select data with pca compression
        prefix2 = ''
        if self.pca_flag:
            prefix2 = '_pca_'+str(self.pca_var)

        if self.pickle==False:
            #
            fname = os.path.join(self.main_dir, self.animal_id,
                         'SVM_Scores',
                         'SVM_Scores_'+
                         self.session+#"_"+
                         self.code+
                         prefix1+
                         '_trial_ROItimeCourses_'+
                         str(self.window)+'sec_'+
                         "Xvalid"+str(self.xvalidation)+
                         "_Slidewindow"+str(self.sliding_window)+
                         #prefix2+
                         '.npz'
                         )
        else:
            # /media/cat/4TBSSD/yuki/IA1/SVM_Scores/SVM_Scores_IA1pm_Feb1_30Hz_whole_stack_pca30Components_window15sec_Xvalid10_Slidewindow30Frames_accuracy.pk
            fname = os.path.join(self.main_dir, self.animal_id,'SVM_Scores',
                     'SVM_Scores_'+
                     self.session+
                     '_whole_stack_pca30Components_'+
                     'window'+str(self.window)+"sec"+
                     "_Xvalid"+str(self.xvalidation)+
                     "_Slidewindow"+str(self.sliding_window)+"Frames"+
                     '_accuracy.pk'
                     )
        #print ("SET FNAME: ", fname)
        self.fname = fname

        # convert wild card file name into correct filename for animal
        main_dir = os.path.join(self.main_dir,
                                self.animal_id,
                                'tif_files')
        session_corrected = os.path.split(
                            glob2.glob(main_dir+"/*"+self.session_id+"*")[0])[1]

        self.session_corrected = session_corrected
        
        
    def get_rewarded_nonrewarded_trials(self):
        
        import warnings

        warnings.simplefilter("ignore")
    
    
        #
        trials = []
        
        for session in self.sessions:
            try: 
                rewarded = np.loadtxt(os.path.join(self.main_dir,
                                                self.animal_id,
                                                'tif_files',
                                                session,
                                                'rewarded_times.txt'))
                nonrewarded = np.loadtxt(os.path.join(self.main_dir,
                                                self.animal_id,
                                                'tif_files',
                                                session,
                                                'nonrewarded_times.txt'))
                
                if rewarded.shape[0]>0 and nonrewarded.shape[0]>0:
                    trials.append([rewarded.shape[0],nonrewarded.shape[0]])
                
            except:
                pass
                #print ("couldn't find: ", session)
                
        
        return np.vstack(trials)
        
        
    #
    def get_number_of_trials(self):

        ''' There are 4 types of trials so must load them individually
        '''

        # convert wild card file name into correct filename for animal
        main_dir = os.path.join(self.main_dir,
                                self.animal_id,
                                'tif_files')
        if True:
            text = main_dir+"/*"+self.session_id+"*"
            session_corrected = os.path.split(glob2.glob(text)[0])[1]
        else:
            print ("ERROR loading")
            self.n_trials = 0
            return

        self.session_corrected = session_corrected
        #
        if self.code == 'code_04' or self.compute_roi:

            # check to see if session done
            fname_txt = os.path.join(self.main_dir,
                                     self.animal_id,
                                     'tif_files',
                                     # self.session_id,
                                     # self.session_id+
                                     session_corrected,
                                     session_corrected+"_all_locs_selected.txt")
            try:
                self.n_trials = np.loadtxt(fname_txt).shape[0]
            except:
                print (" ERror 2")
                self.n_trials = 0
        #
        elif self.code == 'code_04_lockout':
            fname_txt = os.path.join(self.main_dir,
                                     self.animal_id,
                                     'tif_files',
                                     # self.session_id,
                                     # self.session_id+
                                     session_corrected,
                                     session_corrected+"_lockout_10sec_locs_selected.txt")
            try:
                self.n_trials = np.loadtxt(fname_txt).shape[0]
            except:
                self.n_trials = 0
        #
        else:
            fname_data = os.path.join(self.main_dir,
                         self.animal_id,
                         'tif_files',
                         session_corrected,
                         session_corrected+"_3secNoMove_movements.npz")

            data = np.load(fname_data, allow_pickle=True)

            if self.code=='quiescence':
                self.n_trials = data['all_quiescent'].shape[0]
            else:

                # find the code id
                match_code = None
                for p in range(len(self.feature_codes)):
                    if self.code in self.feature_codes[p]:
                        match_code = p
                        break

                self.n_trials = len(data['feature_quiescent'][match_code])




    def get_lever_offset(self):

        fname_lever_offset = os.path.join(self.main_dir,
                                         self.animal_id,
                                         'tif_files',
                                         self.session_corrected,
                                         self.session_corrected+"_lever_offset_n_frames.txt")

        if os.path.exists(fname_lever_offset)==False:

            images_file = fname_lever_offset.replace('_lever_offset_n_frames.txt','_aligned.npy')

            aligned_images = np.load(images_file)

            # Find blue light on/off
            blue_light_threshold = 400  #Intensity threshold; when this value is reached - imaging light was turned on
            start_blue = 0; end_blue = aligned_images.shape[0]

            if np.average(aligned_images[0])> blue_light_threshold:    #Case #1: imaging starts with light on; need to remove end chunk; though likely bad recording
                for k in range(len(aligned_images)):
                    if np.average(aligned_images[k])< blue_light_threshold:
                        #self.aligned_images = self.aligned_images[k:]
                        end_blue = k
                        break
            else:                                                           #Case #2: start with light off; remove starting and end chunks;
                #Find first light on
                for k in range(len(aligned_images)):
                    if np.average(aligned_images[k])> blue_light_threshold:
                        start_blue = k
                        break

                #Find light off - count backwards from end of imaging data
                for k in range(len(aligned_images)-1,0,-1):
                    if np.average(aligned_images[k])> blue_light_threshold:
                        end_blue= k
                        break

            self.lever_offset = start_blue

            np.savetxt(fname_lever_offset, [self.lever_offset])

        else:
            self.lever_offset = int(np.loadtxt(fname_lever_offset))

    def plot_decision_choice_all(self):

        #self.get_number_of_trials()


        #
        sessions = np.load(os.path.join(self.main_dir,
                                        self.animal_id,
                                        'tif_files.npy'))

        n_row = int(sessions.shape[0]/10.)+1
        n_row = 7
        n_col = 10

        #
        fnames_pca = glob2.glob(os.path.join(self.main_dir,
                                             self.animal_id,
                                             "SVM_Scores/*.npy"))

        #
        ctr=0
        plt_flag = True
        for session in sessions:
            self.session_id = os.path.split(session)[1][:-4]

            if plt_flag:
                ax=plt.subplot(n_row,n_col,ctr+1)
                ax.set_xticks([])
                ax.set_yticks([])

            # track if session has has some plotting done
            plt_flag = False
            for fname in fnames_pca:
                if self.session_id in str(fname):

                    self.get_number_of_trials()
                    if self.n_trials < self.min_trials:
                        continue

                    if "lockout" in str(fname):
                        self.lockout = True
                        self.plot_decision_choice('blue',
                                                 str(self.pca_var),
                                                 ax)
                        plt_flag = True
                    else:
                        self.lockout = False
                        self.plot_decision_choice('black',
                                                 str(self.pca_var),
                                                 ax)
                        plt_flag = True

            if plt_flag:
                ctr+=1

        #
        plt.suptitle("ANIMAL: "+ self.animal_id +
                     ", Smoothing window: "+str(self.smooth_window)+
                     ", Min # trials: "+str(self.min_trials), fontsize=20)
        plt.show()


    def plot_first_significant(self):

        #
        sessions = np.load(os.path.join(self.main_dir,
                                        self.animal_id,
                                        'tif_files.npy'))
        #
        fnames_pca = glob2.glob(os.path.join(self.main_dir,
                                             self.animal_id,
                                             "SVM_Scores/*.npy"))

        #
        ctr=0
        for session in sessions:
            self.session_id = os.path.split(session)[1][:-4]

            for fname in fnames_pca:
                if self.session_id in str(fname):

                    self.get_number_of_trials()
                    if self.n_trials < self.min_trials:
                        continue

                    if "lockout" in str(fname):
                        self.lockout = True
                        self.process_session()

                        # compute significance
                        self.compute_significance()
                        #print (self.sig.shape)

                        # find first significant point in time



                    else:
                        self.lockout = False
                        self.process_session()


    #
    def process_session(self):

        #################################################
        ########### LOAD ACCURACY DATA ##################
        #################################################
        if self.pickle==False:
            if True:
                data = np.load(self.fname, allow_pickle=True)
            else:
                try:
                    self.fname = self.fname.replace("Scores_","Scores_ROI_")
                    data = np.load(self.fname, allow_pickle=True)
                except:
                    print( " ... data missing npy 2", self.fname)
                    self.data = np.zeros((0))
                    return
            self.data = data['accuracy']

        # else load specific code data from file
        else:
            try:
                with (open(self.fname, "rb")) as openfile:
                    data = pickle.load(openfile)
                print ("PICKLE DATA: ", len(data))
            except:
                print( " ... data missing pkl", self.fname)
                self.data = np.zeros((0))
                return

            # find the code id
            match_code = None
            print ("self.code: ", self.code)

            for p in range(len(self.feature_codes)):
                if self.code in self.feature_codes[p]:
                    match_code = p
                    break
            print ("Match code: ", match_code)
            self.data = np.array(data[match_code])

        if self.data.shape[0]==0:
            print ("COULDNT FIND DATA")
            return

        #print ("DATA; ", self.data.shape)

        #################################################
        ### PAD DATA WITH  SVM SLDING WINDOW ############
        #################################################

        if self.ideal_window_flag:
            print ("self.sig_sav: ", self.sig_save.shape)
            print ("original len data: ", self.data.shape)
            print ('self.sliding_window: ', self.sliding_window)

            pad = np.zeros((self.sliding_window, self.data.shape[1]), dtype=np.float32)
            self.data = np.vstack((pad, self.data))
            print (" padded  data: ", self.data.shape)

        #################################################
        ### LOAD SHIFTS BETWEEN LEVER AND CALCIUM #######
        #################################################
        if self.dlc_offset_flag:
            self.get_lever_offset()
            print ("loaded lever offset: ", self.lever_offset)
        else:
            self.lever_offset = 0

        #################################################
        ######## LOAD SHIFTS FROM CORRELATION ANALYSIS ##
        #################################################
        if self.shift_flag:
            fname_correlate = os.path.join(self.main_dir, self.animal_id,
                                 'tif_files', self.session,
                                'correlate.npz')

            try:
                data = np.load(fname_correlate,allow_pickle=True)
            except:
                print( " ... data missing correlate.npz", self.fname)
                self.data = np.zeros((0))
                return

            cors = data['cors'].squeeze().T

            #vis.shift = 0
            #print ("SELF SHIFT ID: ", self.shift_id_str)
            if len(self.shift_id_str)>1:
                self.shift_id = int(self.shift_id_str[0])
                self.shift_additional = float(self.shift_id_str[1:])
            else:
                self.shift_id = int(self.shift_id_str)
                self.shift_additional = 0

            #print ( " using shift: ", self.shift_id+self.shift_additional)

            corr_featur_id = self.shift_id

            temp_trace = cors[:,corr_featur_id]
            temp_trace[:2000] = 0
            temp_trace[-2000:] = 0
            self.shift = round(np.argmax(temp_trace)/1000. - 15.,2)+self.shift_additional
            #print ("SHIFT Loaded: ", self.shift)

        #
        if 'code_04' not in self.code and self.compute_roi==False:
            #print ("... DLC shift applied for body movement: ", self.shift)
            self.data = np.roll(self.data,int(self.shift*self.imaging_rate),axis=0)

        #

        if self.apply_lever_offset:
            #print ("... LEVER OFFSET applied: ", self.lever_offset)
            self.data = np.roll(self.data,-self.lever_offset,axis=0)
           # print (" rolled data: ", self.data.shape)

        #################################################
        ########### LOAD # TRIALS #######################
        #################################################
        self.get_number_of_trials()

        #################################################
        ######### COMPUTE MEAN AND STD ##################
        #################################################


        #
        mean = self.data.mean(1)

        #
        if self.smooth_window is not None:
            #mean = self.filter_trace(mean)
            data = []
            for k in range(self.data.shape[1]):
                data.append(self.filter_trace(self.data[:,k]))
            self.data = np.array(data).copy().T

            mean = self.data.mean(1)

        #
        self.mean = mean

        #
        self.std = np.std(self.data, axis=1)

        # clip the data to the required values
        self.data = self.data[(self.xlim[0]+self.window)*self.imaging_rate:
                              (self.xlim[1]+self.window)*self.imaging_rate]

        self.mean = self.mean[(self.xlim[0]+self.window)*self.imaging_rate:
                              (self.xlim[1]+self.window)*self.imaging_rate]
        #print ("self mean: ", self.mean.shape)

        self.std = self.std[(self.xlim[0]+self.window)*self.imaging_rate:
                              (self.xlim[1]+self.window)*self.imaging_rate]


    def visualize_accuracy_curves(self):
        
        #
        self.fig=plt.figure()
        self.ax = plt.subplot(111)

        # 
        self.n_trials_plotting=[]
        self.shift_SVM = True
        self.n_trials=200
        self.shift_flag = False
        self.apply_lever_offset = False  
        self.xlim = [-20,0]
        self.show_EDT = True
        self.cbar_offset = 0
        self.edt_offset = 0
        self.edt_offset_y = 0
        self.edt_offset_x = 0
        self.show_title = False
        self.show_legend = False
        self.legend_location = 1
        self.cbar=False

        # 
        self.cmap = "Blues_r"
        self.ax.set_xlabel("Time (sec)")
        self.ax.set_ylabel("Decoding accuracy")

        # 
        plt.suptitle(self.session_id+" lever pull decoding")
        
        #
        fname = os.path.join(self.root_dir,
                             self.animal_id,
                             'tif_files',
                             self.session_id,
                             'SVM_Scores_'+
                             self.session_id+ 
                             self.code+
                             '_trial_ROItimeCourses_'+
                             str(self.window)+'sec_'+
                             "Xvalid"+str(self.xvalidation)+
                             "_Slidewindow"+str(self.sliding_window)+
                             '.npz')
                
        #
        self.plot_significant_fname('blue',
                                   self.session_id,
                                  fname)
    
    #
    def process_session_concatenated(self):

        #
        try:
            data = np.load(self.fname, allow_pickle=True)
        except:
            print( " ... data missing", self.fname)
            self.data = np.zeros((0))
            return

        self.data = data['accuracy']

		#
        fname_n_trials = self.fname[:-4]+'_n_trials.npy'
        self.n_trials = np.load(fname_n_trials)

        if False:
            # gets the corect filename to be loaded below
            self.get_fname()

            #
            if os.path.exists(self.fname)==False:
                print ("missing: ", self.fname)
                self.data = np.zeros((0))
                return

        #
        mean = self.data.mean(1)

        #
        if self.smooth_window is not None:
            #mean = self.filter_trace(mean)
            data = []
            for k in range(self.data.shape[1]):
                data.append(self.filter_trace(self.data[:,k]))
            self.data = np.array(data).copy().T

            mean = self.data.mean(1)

        #
        self.mean = mean

        #
        self.std = np.std(self.data, axis=1)


    def plot_decision_choice(self, clr, label, ax=None):

        #
        self.process_session()

        # get times
        t = np.linspace(-9.5, 9.5, self.mean.shape[0])

        # plotting steps
        if ax is None:
            ax=plt.subplot(111)

        ax.set_title(self.session_id, fontsize=6.5,pad=0.9)
        ax.set_ylabel(str(self.n_trials)+" ("+str(self.n_trials_lockout)+")", fontsize=8)

        #
        ax.plot(t,
                self.mean,
                c=clr,
                label = label,
                linewidth=4)

        ax.fill_between(t, self.mean-self.std, self.mean+self.std, color=clr, alpha = 0.2)

        self.format_plot(ax)

    def compute_significance(self):

        #print ("self.data: ", self.data.shape)

        #
        sig = []
        for k in range(self.data.shape[0]):
            #res = stats.ks_2samp(self.data[k],
            #                     control)
            #res = stats.ttest_ind(first, second, axis=0, equal_var=True)

            #
            res = scipy.stats.ttest_1samp(self.data[k], 0.5)

            sig.append(res[1])


        self.sig_save = np.array(sig).copy()
        #print ("Self sig save: ", self.sig_save.shape)


        # multiple hypothesis test Benjamini-Hockberg
        temp = np.array(sig)
        #print ("data into multi-hypothesis tes:", temp.shape)
        temp2 = multipletests(temp, alpha=self.significance, method='fdr_bh')
        sig = temp2[1]

        # Expand this so we can plot it as a 1D image.
        sig=np.array(sig)[None]

        #
        thresh = self.significance
        idx = np.where(sig>thresh)
        sig[idx] = np.nan

        #
        idx = np.where(self.mean<0.5)
        sig[:,idx] = np.nan

        # save it
        self.sig = sig
        #print ("Final sig: ", self.sig.shape)

        # find earliest significant;
        earliest_continuous = 0
        for k in range(self.sig.shape[1]-1,0,-1):
            # go backwards through the significance values until hit a nan
            if np.isnan(self.sig[0][k]):
                earliest_continuous = k+1
                break

            if self.sig[0][k]<=self.significance:
                earliest_continuous = k+1
            else:
                break

        #
        #print ("earliest: ", earliest_continuous,
        #       " in sec: ", -(self.sig.shape[1]-earliest_continuous)/30.)

        self.earliest_continuous = -(self.sig.shape[1]-earliest_continuous)/30.

        self.edt = self.earliest_continuous

        #print (" signianc n-trials: ", self.n_trials)


    #
    def compute_first_decoding_time_ROI(self):
        print ("TESTING")
        #
        fname_out = os.path.join(self.main_dir, self.animal_id,
                         'first_decoding_time_'+self.code+'.npz')

        if self.ideal_window_flag==True:
            fname_out = fname_out.replace('.npz','_ideal_window.npz')

        if os.path.exists(fname_out) and self.overwrite==False:
            return

        #
        res_continuous = []
        res_earliest = []
        session_nos = []
        session_names = []
        n_trials = []
        sigs = []

        #
        self.get_sessions()
        print ("SESSIONS: ", self.session_ids)
        #
        n_good_sessions = 0
        for p in range(len(self.session_ids)):
            self.session_id = self.session_ids[p]

            #
            self.fname = self.fnames_svm[p]
            print ("self.fname: ", self.fname)

            self.process_session()
            # print ("a: ", a, self.session_id, self.data.shape)
            #
            if self.data.shape[0] == 0:
                continue


            # compute significance
            self.compute_significance()

            # save all the significant vals;
            sigs.append(self.sig_save)

            #
            self.sig = self.sig.squeeze()

            # find earliest period of significance, going back in time;
            for k in range(self.sig.shape[0]-1,0,-1):
                if np.isnan(self.sig[k])==True:
                    break

            #
            temp = -self.window+k/self.imaging_rate

            # Exclude one of the weird datapoint from the AQ2? session
            if temp>0:
                print ("n trials: ", self.n_trials, a,
                       p, temp, self.session_id, self.sig.shape)
                print ('SKIPPING')
                continue

            #
            res_continuous.append(self.edt)

            ########################################
            ########################################
            # find aboslute earliest
            k_earliest = self.sig.shape[0]
            for k in range(self.sig.shape[0]-1,0,-1):
                if np.isnan(self.sig[k])==True:
                    k_earliest = k

            earliest_temp = -self.window+k_earliest/self.imaging_rate
            res_earliest.append(earliest_temp)

            ########################################
            ########################################
            session_nos.append(p)

            #
            session_names.append(self.session_id)

            #
            n_trials.append(self.n_trials)
            n_good_sessions+=1
            print (" ... finished session...edt: ", self.edt)
            print ('')
            print ('')


        #
        print (" # OF GOOD SESSIONS: ", n_good_sessions)
        np.savez(fname_out,
                 all_res_continuous = res_continuous,
                 all_res_earliest = res_earliest,
                 all_session_nos = session_nos,
                 all_session_names = session_names,
                 all_n_trials = n_trials,
                 all_sigs = sigs
                 )
        print ('')
        print ('')
        print ('')
        print ('')
        print ('')

        

    def compute_first_decoding_time(self, lockouts):

        print ("TESTING")

        #
        #if lockouts = [False, True]
        for lockout in lockouts:
            self.lockout=lockout

            #
            if self.lockout==True:
                fname_out = os.path.join(self.main_dir, self.animal_id,
                             'first_decoding_time_lockout.npz')
            else:
                fname_out = os.path.join(self.main_dir, self.animal_id,
                             'first_decoding_time.npz')
            
            
            if self.filter_locaNMF==True:
                fname_out = fname_out[:-4]+"_0.3hzFilter.npz"

            #
            if self.ideal_window_flag==True:
                fname_out = fname_out.replace('.npz','_ideal_window.npz')

            res_continuous = []
            res_earliest = []
            session_nos = []
            session_names = []
            n_trials = []
            sigs = []

            #
            self.get_sessions()

            # 
            n_good_sessions = 0
            for p in range(len(self.session_ids)):
            #for p in range(len(self.fnames_svm)):
                self.session_id = self.session_ids[p]

                #
                self.fname = self.fnames_svm[p]
                
                if self.filter_locaNMF:
                    self.fname = self.fname[:-4]+'_.3HzFilter.npz'
                    #print ("Redirected to filtered fname: ", self.fname)
                
                if os.path.exists(self.fname)==False:
                    continue

                #    
                self.sliding_window = self.ideal_sliding_windows[p]

                #
                self.process_session()

                #
                if self.data.shape[0] == 0:
                    continue

                if self.n_trials<self.min_trials:
                    #print ("too few trials")
                    continue

                # compute significance
                self.compute_significance()

                # save all the significant vals;
                sigs.append(self.sig_save)

                #
                self.sig = self.sig.squeeze()

                # find earliest period of significance, going back in time;
                for k in range(self.sig.shape[0]-1,0,-1):
                    if np.isnan(self.sig[k])==True:
                        break

                #
                temp = -self.window+k/self.imaging_rate

                # Exclude one of the weird datapoint from the AQ2? session
                if temp>0:
                    #print ("n trials: ", self.n_trials, a,
                    #       p, temp, self.session_id, self.sig.shape)
                    continue

                #
                res_continuous.append(temp)

                # find aboslute earliest
                k_earliest = self.sig.shape[0]
                for k in range(self.sig.shape[0]-1,0,-1):
                    if np.isnan(self.sig[k])==True:
                        k_earliest = k
                res_earliest.append(-self.window+k_earliest/self.imaging_rate)

                #
                session_nos.append(p)

                #
                session_names.append(self.session_id)

                #
                n_trials.append(self.n_trials)
                n_good_sessions+=1
                #print (" ... finished session...")
                #print ('')


            #
            print (" # OF GOOD SESSIONS: ", n_good_sessions)
            np.savez(fname_out,
                     all_res_continuous = res_continuous,
                     all_res_earliest = res_earliest,
                     all_session_nos = session_nos,
                     all_session_names = session_names,
                     all_n_trials = n_trials,
                     all_sigs = sigs
                     )
            print ('')
            print ('')
            print ('')
            print ('')
            print ('')


    def compute_first_decoding_time_concatenated(self, lockouts):

        #
        #if lockouts = [False, True]
        for lockout in lockouts:
            self.lockout=lockout

            #

            if self.lockout==True:
                fname_out = os.path.join(self.main_dir, self.animal_id,
                             'first_decoding_time_concatenated_lockout.npz')
            else:
                fname_out = os.path.join(self.main_dir, self.animal_id,
                             'first_decoding_time_concatenated.npz')

            #
            res_continuous = []
            res_earliest = []
            session_nos = []
            session_names = []
            n_trials = []
            sigs = []

            #
            self.get_sessions()

            #
            for p in range(len(self.session_ids)):
                self.session_id = self.session_ids[p]
                print ("ANIMAL: ", self.animal_id, "  SESSION: ", self.session_id)

                #
                self.fname = self.fnames_svm[p]

                #
                self.process_session_concatenated()

                #
                if self.data.shape[0] == 0:
                    print ("skipping, no data")
                    print ('')
                    continue

                # compute significance
                self.compute_significance()

                # save all the significant vals;
                sigs.append(self.sig_save)

                #
                self.sig = self.sig.squeeze()

                # find earliest period of significance, going back in time;
                for k in range(self.sig.shape[0]-1,0,-1):
                    if np.isnan(self.sig[k])==True:
                        break

                #
                temp = -self.window+k/self.imaging_rate

                # Exclude one of the weird datapoint from the AQ2? session
                if temp>0:
                    print ("skipping weird datapoint...")
                    continue

                #
                res_continuous.append(temp)

                # find aboslute earliest
                k_earliest = self.sig.shape[0]
                for k in range(self.sig.shape[0]-1,0,-1):
                    if np.isnan(self.sig[k])==True:
                        k_earliest = k
                res_earliest.append(-self.window+k_earliest/self.imaging_rate)

                #
                session_nos.append(p)

                #
                session_names.append(self.session_id)

                #
                n_trials.append(self.n_trials)


            print (" # SESSIONS PROCESSED: ", len(res_continuous))
            print ('')
            print ('')
            print ('')
            print ('')
            np.savez(fname_out,
                     all_res_continuous = res_continuous,
                     all_res_earliest = res_earliest,
                     all_session_nos = session_nos,
                     all_session_names = session_names,
                     all_n_trials = n_trials,
                     all_sigs = sigs
                     )

    def compute_first_decoding_time_curves(self,
                                        data,
                                        shift):

        #
        temp = data

        print ("data ", temp)
        all_res_continuous_all = np.array(temp)+shift

        bins = np.arange(-15,2,1)
        #print (all_res_continuous_all)
        y = np.histogram(all_res_continuous_all, bins = bins)

        return all_res_continuous_all, y[0]

    def plot_first_decoding_time_curves(self,
                                        data,
                                        animal_id,
                                        shift,
                                        linestyle,
                                        clr,
                                        plotting=False):

        #
        linewidth=5
        temp = data['all_res_continuous']

        # print("temp data allrescontinuous: ", temp.shape)

        #print ("data['all_res_continuous']: ", temp)
        all_res_continuous_all = np.array(temp)+shift

        #print( "animal_id: ", animal_id, "  # sessions: ", all_res_continuous_all.shape)

        bins = np.arange(-15,2,1)
        #print (all_res_continuous_all)
        y = np.histogram(all_res_continuous_all, bins = bins)

        if plotting:
            plt.plot(y[1][:-1],y[0],
                 c=clr,
                 linestyle=linestyle,
                 linewidth=linewidth,
                 label=animal_id,
                 alpha=.8)

        return all_res_continuous_all, y[0]



    def plot_first_decoding_time(self,
                                 data,
                                 return_ids_threshold,
                                 clrs):

        #
        all_res_continuous_all = data['all_res_continuous']
        all_session_names = data['all_session_names']
        all_session_nos = data['all_session_nos']

        try:
            data = np.load(self.main_dir+'/first_decoding_time'+
                         "_minTrials"+str(self.min_trials)+
                         '_lockout_'+
                         str(self.window)+'sec.npz',
                         allow_pickle=True)
            all_res_continuous_lockout = data['all_res_continuous']
        except:
            all_res_continuous_lockout = []
            pass

        # REMOVE A COUPLE OF WEIRD SVM PREDICTION CURVES
        if return_ids_threshold is not None:
            for k in range(len(all_res_continuous_all)):
                idx = np.where(np.array(all_res_continuous_all[k])<=return_ids_threshold)[0]
                if idx.shape[0]>0:
                    print ("all: ", np.array(all_session_names[k])[idx])
                    print ("all: ", np.array(all_session_nos[k])[idx])

            for k in range(len(all_res_continuous_lockout)):
                idx = np.where(np.array(all_res_continuous_lockout[k])<=return_ids_threshold)[0]
                if idx.shape[0]>0:
                    print ("lockout: ", np.array(all_session_names[k])[idx])
                    print ("lockout: ", np.array(all_session_nos[k])[idx])

        #
        data_sets_all = []
        for k in range(len(all_res_continuous_all)):
            data_sets_all.append(all_res_continuous_all[k])
        print (data_sets_all)
        #
        data_sets_lockout = []
        for k in range(len(all_res_continuous_lockout)):
            data_sets_lockout.append(all_res_continuous_lockout[k])

        # Computed quantities to aid plotting
        hist_range = (-self.window,1)
        bins = np.arange(-self.window,1,1)

        #
        binned_data_sets_all = [
            np.histogram(d, range=hist_range, bins=bins)[0]
            for d in data_sets_all
        ]

        binned_data_sets_lockout = [
            np.histogram(d, range=hist_range, bins=bins)[0]
            for d in data_sets_lockout
        ]

        #
        binned_maximums = np.max(binned_data_sets_all, axis=1)
        spacing = 40
        x_locations = np.arange(0, spacing*6,spacing)

        # The bin_edges are the same for all of the histograms
        bin_edges = np.arange(hist_range[0], hist_range[1],1)
        centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[1:]#[:-1]
        heights = np.diff(bin_edges)

        # Cycle through and plot each histogram
        fig, ax = plt.subplots(figsize=(10,5))
        for x_loc, binned_data, binned_data_lockout in zip(x_locations, binned_data_sets_all, binned_data_sets_lockout):
            lefts = x_loc - 0.3# * binned_data
            ax.barh(centers, -binned_data, height=heights, left=lefts, color=clrs[0])

            lefts = x_loc #- 0.5 * binned_data_lockout
            ax.barh(centers, binned_data_lockout, height=heights, left=lefts, color=clrs[1])

        ax.set_xticks(x_locations)
        ax.set_xticklabels(self.labels)
        ax.set_xlim(-20,spacing*6)
        ax.set_ylim(-self.window-0.5,0)
        ax.set_ylabel("Data values")
        ax.set_xlabel("Data sets")

    def exp_func(self, x, a, b, c, d):
        return a*np.exp(-c*(x-b))+d

    def fit_exp(self, all_preds,
                all_trials,
                ax):

        # fit exponentials
        all_preds = np.array(all_preds)
        offset = 10
        popt, pcov = curve_fit(self.exp_func,
                               all_preds+offset,
                               all_trials,
                               [offset,1e-6,0.001,0])

        #print("Popt: ", popt)
        time_back = -20
        x= np.linspace(time_back,offset,1000)
        ax.plot(x+time_back+offset,self.exp_func(x,*popt),
                 linewidth=5, c='black')

    def fit_line(self, all_preds,
                all_trials,
                color,
                ax):

        # Create linear regression object
        regr = linear_model.LinearRegression()

        #
       # print (self.animal_id, self.session, all_preds)

        #
        all_preds = np.array(all_preds)[:,None]
        #print ("all preds: ", all_preds.shape)
        all_trials = np.array(all_trials)
        regr.fit(all_preds, all_trials)

        # Make predictions using the testing set
        x_test = np.arange(-self.window,0.5,1)[:,None]
        y_test = regr.predict(x_test)

        #
        ax.plot(x_test, y_test,
                 color=color,
                 linewidth=3)


    def plot_first_decoding_time_vs_n_trials(self,
                                             fname=None):
												 
        labels = ["M1", "M2", "M3", "M4","M5",'M6']

        # flag to search for any signfiicant decoding time, not just continous ones
        earliest = False

        if earliest==False:
            # data = np.load(self.main_dir + '/first_decoding_time_all_'+str(self.window)+
            #                'sec.npz',allow_pickle=True)
            if fname is None:
                fname = os.path.join(self.main_dir,
                                     '/first_decoding_time'+ "_minTrials"+str(self.min_trials)+
                                     '_all_'+
                                     str(self.window)+'sec.npz')

            data = np.load(fname,
                          allow_pickle=True)


            res_continuous_all = data['all_res_continuous']
            all_n_trials = data['all_n_trials']

            if self.lockout==True:
                try:
                    data = np.load(self.main_dir+'/first_decoding_time'+
                                 "_minTrials"+str(self.min_trials)+
                                 '_lockout_'+
                                 str(self.window)+'sec.npz',
                                 allow_pickle=True)
                    res_continuous_lockout = data['all_res_continuous']
                    lockout_n_trials = data['all_n_trials']
                except:
                    res_continuous_lockout=[]
                    lockout_n_trials = []
            else:
                res_continuous_lockout=[]
                lockout_n_trials = []
        else:
            print ("Data not found, skipping")
            return

        #
        #fig = plt.figure(figsize=(20,20))
        all_preds = []
        all_trials = []

        for k in range(len(res_continuous_all)):
			
            if labels[k]!=self.animal_id:
                continue
			
            ax=plt.subplot(1,1,1)
            plt.ylim(self.ylims[0],self.ylims[1])

            plt.xlim(-15,0)
            plt.xlabel("EDTs (sec)")
            plt.ylabel("# trials in session")
            #plt.xticks([])
            #plt.yticks([])

            trials1 = np.array(all_n_trials[k])
            predictions1 = np.array(res_continuous_all[k])
            if predictions1.shape[0]<=1:
                continue

            #
            #print ("Predictiosn1: ", perdictions1.shape)
            plt.scatter(predictions1,
                        trials1,
                        s=100,
                        c=np.arange(trials1.shape[0])+20,
                        edgecolors='black',
                        cmap=self.cmap)

            #self.fit_exp(all_preds, all_trials, ax)
            self.fit_line(predictions1,
                          trials1,
                          'black',
                          ax)

            #
            corr_pred_trials = scipy.stats.pearsonr(predictions1, trials1)
            corr_pred_time = scipy.stats.pearsonr(predictions1,
                                                  np.arange(len(predictions1)))


            # LOCKOUT TRIALS
            try:
                trials2 = np.array(lockout_n_trials[k])
                #print ("lockout trials2: ", trials2)
                predictions2 = np.array(res_continuous_lockout[k])
                corr_pred_trial_lockout = scipy.stats.pearsonr(predictions2, trials2)
                corr_pred_time_lockout = scipy.stats.pearsonr(predictions2, np.arange(len(predictions2)))
            except:
                pass

            from decimal import Decimal
            patches = []
            patches.append(mpatches.Patch(color='black',
                                       label='all vs. # trials: '+
                                       str(round(corr_pred_trials[0],2))+
                                       " ("+str("pval: {0:.1}".format(corr_pred_trials[1]))+")"
                                          ))
            patches.append(mpatches.Patch(color='grey',
                                       label='all vs. time: '+
                                       str(round(corr_pred_time[0],2))+
                                       " ("+str("pval: {0:.1}".format(corr_pred_time[1]))+")"
                                        ))

            try:
                patches.append(mpatches.Patch(color='blue',
                                           label='lockout vs. # trials: '+
                                           str(round(corr_pred_trial_lockout[0],2))+
                                           " ("+str("pval: {0:.1}".format(corr_pred_trial_lockout[1]))+")"
                                          ))

                patches.append(mpatches.Patch(color='lightblue',
                                           label='lockout vs. time: '+
                                           str(round(corr_pred_time_lockout[0],2))+
                                           " ("+str("pval: {0:.1}".format(corr_pred_time_lockout[1]))+")"
                                          ))
            except:
                pass

            if self.plot_legend:
                plt.legend(handles=patches,
                           fontsize=6)

            try:
                plt.scatter(predictions2,
                            trials2,
                            s=100,
                            c=np.arange(trials2.shape[0])+20,
                            edgecolors='black',
                            cmap=cm.Blues)

                self.fit_line(predictions2,
                              trials2,
                              'blue',
                              ax)
            except:
                pass

            # select n_trials > 100 and connect them
            idx = np.where(trials1>80)[0]
            for p in idx:
                try:
                    plt.plot([predictions1[p], predictions2[p]],
                         [trials1[p], trials2[p]],'r--')
                except:
                    pass



        plt.suptitle("All sessions all trials")


    def plot_significant_fname(self, clr, label, fname):

        # set continuos to
        self.earliest_continuous = np.nan

        # GET FILENAME IF EXISTS
        self.fname = fname

        # PROCESS SESSION
        self.process_session()

        # COUNT TRIALS
        self.n_trials_plotting.append(self.n_trials)
        if self.n_trials==0 or self.data.shape[0]==0:
            return
        #print ("self n trials: ", self.n_trials)

        # COMPUTE TIME WINDOW FOR PLOTTING
        t = np.linspace(self.xlim[0], self.xlim[1], self.mean.shape[0])
        plt.plot(t,
                 self.mean,
                 c=clr,
                 label = label + " # trials: "+str(self.n_trials),
                 linewidth=self.linewidth,
                 alpha=self.alpha)

        # FILL IN STD FOR RESULTS
        plt.fill_between(t, self.mean-self.std, self.mean+self.std, color=clr, alpha = 0.2)

        # COMPUTE SIGNIFICANCE
        self.compute_significance()

        if self.show_EDT:
            self.ax.annotate("EDT: "+str(round(self.earliest_continuous,2))+"sec",
                         xy=(self.earliest_continuous, 0.5),
                         xytext=(self.earliest_continuous-3+self.edt_offset_x,
                                 0.75+self.edt_offset_y),
                         arrowprops=dict(arrowstyle="->"),
                         fontsize=20,
                         color=clr)
            self.edt_offset+=0.02
            x = self.earliest_continuous

            #
            # if True:
                # plt.fill_between([x,0], 0,1.0 ,
                             # color='grey',alpha=.2)

        # PLOT SIGNIFICANCE IMAGE BARS
        vmin=0.0
        vmax=self.significance
        axins = self.ax.inset_axes((0,1-self.cbar_thick-self.cbar_offset,1,self.cbar_thick))
        axins.set_xticks([])
        axins.set_yticks([])

        im = axins.imshow(self.sig,
                          vmin=vmin,
                          vmax=vmax,
                          aspect='auto',
                          #cmap='viridis_r')
                          cmap=self.cmap)

        #
        ticks = np.round(np.linspace(vmin, vmax, 4),8)
        #print ("vmin, vmax; ", vmin, vmax, "ticks: ", ticks)
        #fmt = '%1.4f'
        fmt='%.0e'
        #
        if self.cbar:
            cbar = self.fig.colorbar(im,
                                ax=self.ax,
                                shrink=0.2,
                                ticks=ticks,
                                format = fmt)

            cbar.ax.tick_params(labelsize=25)

        # APPLY STANDARD FORMATS
        self.format_plot(self.ax)

        #
        if self.shift_SVM:
            try:
                fname = os.path.join(self.main_dir, self.animal_id,
                         'tif_files',
                         self.session,
                         'shift.txt'
                         )

                shift = float(np.loadtxt(fname))
                print ("SHIFT REQUIRD: ", fname, " ", shift)

            except:
                shift = 0

        else:
            shift= 0

        if self.show_title:
            plt.title(self.animal_id + "  session: "+str(self.session) +
                  "\n smoothing window: "+str(round(self.smooth_window/30.,1))+"sec"+
                  "\n [Ca] <-> DLC shift: "+str(round(shift,2))+" sec")

        #
        self.cbar_offset+=self.cbar_thick
        if self.show_legend:
            plt.legend(loc=self.legend_location,
                   fontsize=20)

        # for multi axes plots
        self.plotted = True


    def plot_significant(self, clr, label):

        # set continuos to
        self.earliest_continuous = np.nan

        # GET FILENAME IF EXISTS
        self.get_fname()
        if self.fname is None:
            print ("no file, exiting")
            return

        # PROCESS SESSION
        self.process_session()

        # COUNT TRIALS
        self.n_trials_plotting.append(self.n_trials)
        if self.n_trials==0 or self.data.shape[0]==0:
            return
        #print ("self n trials: ", self.n_trials)

        # COMPUTE TIME WINDOW FOR PLOTTING
        t = np.linspace(self.xlim[0], self.xlim[1], self.mean.shape[0])
        plt.plot(t,
                 self.mean,
                 c=clr,
                 label = label + " # trials: "+str(self.n_trials),
                 linewidth=self.linewidth,
                 alpha=self.alpha)

        # FILL IN STD FOR RESULTS
        plt.fill_between(t, self.mean-self.std, self.mean+self.std, color=clr, alpha = 0.2)

        # COMPUTE SIGNIFICANCE
        self.compute_significance()

        if self.show_EDT:
            self.ax.annotate("EDT: "+str(round(self.earliest_continuous,2))+"sec",
                         xy=(self.earliest_continuous, 0.5),
                         xytext=(self.earliest_continuous-3+self.edt_offset_x,
                                 0.75+self.edt_offset_y),
                         arrowprops=dict(arrowstyle="->"),
                         fontsize=20,
                         color=clr)
            self.edt_offset+=0.02
            x = self.earliest_continuous

            #
            if True:
                plt.fill_between([x,0], 0,1.0 ,
                             color='grey',alpha=.2)

        # PLOT SIGNIFICANCE IMAGE BARS
        vmin=0.0
        vmax=self.significance
        axins = self.ax.inset_axes((0,1-self.cbar_thick-self.cbar_offset,1,self.cbar_thick))
        axins.set_xticks([])
        axins.set_yticks([])

        im = axins.imshow(self.sig,
                          vmin=vmin,
                          vmax=vmax,
                          aspect='auto',
                          #cmap='viridis_r')
                          cmap=self.cmap)

        #
        ticks = np.round(np.linspace(vmin, vmax, 4),8)
        print ("vmin, vmax; ", vmin, vmax, "ticks: ", ticks)
        #fmt = '%1.4f'
        fmt='%.0e'
        #
        if self.cbar:
            cbar = self.fig.colorbar(im,
                                ax=self.ax,
                                shrink=0.2,
                                ticks=ticks,
                                format = fmt)

            cbar.ax.tick_params(labelsize=25)

        # APPLY STANDARD FORMATS
        self.format_plot(self.ax)

        #
        if self.shift_SVM:
            try:
                fname = os.path.join(self.main_dir, self.animal_id,
                         'tif_files',
                         self.session,
                         'shift.txt'
                         )

                shift = float(np.loadtxt(fname))
                print ("SHIFT REQUIRD: ", fname, " ", shift)

            except:
                shift = 0

        else:
            shift= 0

        if self.show_title:
            plt.title(self.animal_id + "  session: "+str(self.session) +
                  "\n smoothing window: "+str(round(self.smooth_window/30.,1))+"sec"+
                  "\n [Ca] <-> DLC shift: "+str(round(shift,2))+" sec")

        #
        self.cbar_offset+=self.cbar_thick
        if self.show_legend:
            plt.legend(loc=self.legend_location,
                   fontsize=20)

        # for multi axes plots
        self.plotted = True

    #
    def plot_significant_concatenated(self, clr, label):

        #
        self.xlim = [-15,0]

        #
        try:
            data = np.load(self.fname, allow_pickle=True)
            self.data=data['accuracy']
        except:
            print (" file missing: ", self.fname)
            return
			
       # print ("loaded data: ", self.data.shape)

        #
        if self.smooth_window is not None:
            #mean = self.filter_trace(mean)
            data = []
            for k in range(self.data.shape[1]):
                data.append(self.filter_trace(self.data[:,k]))
            self.data = np.array(data).copy().T

        #
        self.mean = self.data.mean(1)
        self.std = np.std(self.data, axis=1)

        #
        t = np.linspace(-self.window+2, 0,
                         self.mean.shape[0])

        #
        plt.plot(t,
                 self.mean,
                 c=clr,
                 label = label,
                 linewidth=self.linewidth,
                 alpha=self.alpha)

        plt.fill_between(t, self.mean-self.std, self.mean+self.std, color=clr, alpha = 0.2)

        # compute significance
        self.compute_significance()

        # set
        vmin=0.0
        vmax=self.significance

        # plot significance
        axins = self.ax.inset_axes((0,1-self.cbar_thick-self.cbar_offset,1,self.cbar_thick))
        axins.set_xticks([])
        axins.set_yticks([])
        #print ('self.sig.shape', self.sig.shape)
        self.sig = self.sig[:,
                         (self.xlim[0]+30)*30 -2*self.smooth_window: (self.xlim[1]+30)*30-2*self.smooth_window ]
        im = axins.imshow(self.sig,
                          vmin=vmin,
                          vmax=vmax,
                          aspect='auto',
                          #cmap='viridis_r')
                          cmap=self.cmap)

        # find earliest significant;
        earliest_continuous = 0
        #print ("self sig: ", self.sig.shape)
        for k in range(self.sig.shape[1]-1,0,-1):
            if self.sig[0][k]<=self.significance:
                earliest_continuous = k
            else:
                break

        #print ("earliest: ", earliest_continuous,
        #       " in sec: ", -(self.sig.shape[1]-earliest_continuous)/30.)


        self.ax.annotate("EDT: "+str(round(-(self.sig.shape[1]-earliest_continuous)/30.,1))+"sec",
                         xy=(-(self.sig.shape[1]-earliest_continuous)/30., 0.5),
                         xytext=(-(self.sig.shape[1]-earliest_continuous)/30.+self.edt_offset_y,
                                 0.75+self.edt_offset_y),
                         arrowprops=dict(arrowstyle="->",
                                         lw=5),
                         fontsize=20,
                         color=clr)
        x = -(self.sig.shape[1]-earliest_continuous)/30.

        # plot significance region
       # plt.fill_between([x-1/33.,0], 0,1.0 ,
         #                color=clr,alpha=.2)


        if self.cbar==True:

            #
            ticks = np.round(np.linspace(vmin, vmax, 4),8)
            print ("vmin, vmax; ", vmin, vmax, "ticks: ", ticks)
            #fmt = '%1.4f'
            fmt='%.0e'


            #
            cbar = self.fig.colorbar(im,
                                ax=self.ax,
                                shrink=0.2,
                                ticks=ticks,
                                format = fmt)

            cbar.ax.tick_params(labelsize=25)
            #
            self.cbar_offset+=self.cbar_thick

        self.format_plot(self.ax)
        plt.title(self.animal_id + "  session: "+str(self.session))
        
        self.ax.set_xlabel("Time (sec)")
        self.ax.set_ylabel("Decoding accuracy")
        
        plt.xlim(-15,0)  # note computation of EDT only works for -15..0 windows


        #

    def plot_trends(self, clr, label, ax):
        #
        t = np.arange(-9.5, 10.5, 1)

        colors = plt.cm.magma(np.linspace(0,1,self.n_colors))

        #
        mean = self.data.mean(1)
        std = np.std(self.data,axis=1)
        ax.plot(t, mean,
                 c=colors[clr],
                 label = label,
                 linewidth=4)

        self.format_plot(ax)


    def plot_animal_decision_longitudinal_matrix(self,
                                                 animal_name,
                                                 lockout=False,
                                                 ax=None):


        if ax is None:
            fig=plt.figure()
            ax=plt.subplot(111)
        #
        root_dir = self.main_dir+animal_name+'/'
        fnames = np.sort(glob2.glob(root_dir+'SVM_scores_'+animal_name+"*"))

        #
        #fig=plt.figure()
        img =[]
        for fname in fnames:
            if 'lockout' not in fname:
                if lockout==False:
                    self.load_data(fname)
                    temp = self.data.mean(1)
                else:
                    #
                    idx = fname.find('SVM_scores_'+animal_name)
                    fname2 = fname[:idx+12]+"_lockout"+fname[idx+12:]
                    self.load_data(fname[:idx+14]+"_lockout"+fname[idx+14:])
                    temp = self.data.mean(1)

                if self.filter:
                    temp = self.filter(temp)
                img.append(temp)

        img=np.array(img)
        ax.imshow(img)

        plt.xticks(np.arange(0,img.shape[1],2),
                           np.arange(-9.5,10,2))
        plt.ylabel("Study day")
        plt.title(animal_name)
        plt.suptitle("lockout: "+str(lockout))
        #ticks = np.linspace(vmin,vmax,4)
        #cbar = fig.colorbar(im, ax=axes[5],
        #                    ticks=ticks)


    def plot_animal_decision_longitudinal(self, animal_name):
        #animal_names = ['IA1','IA2','IA3','IJ1','IJ2','AQ2']

        idx=self.animal_ids.index(animal_name)

        #
        root_dir = self.main_dir+animal_name+'/'
        fnames = np.sort(glob2.glob(root_dir+'SVM_scores_'+animal_name+"*"))

        #
        fig=plt.figure()
        ax1=plt.subplot(1,2,1)
        ax1.set_title("All")
        ax2=plt.subplot(1,2,2)
        ax2.set_title("Lockout")

        self.n_colors = self.n_colors[idx]
        ctr=0
        for fname in fnames:
            if 'lockout' not in fname:
                print (fname)
                self.load_data(fname)
                self.plot_trends(ctr,'all',ax1)

                idx = fname.find('SVM_scores_'+animal_name)
                fname2 = fname[:idx+12]+"_lockout"+fname[idx+12:]
                print (fname2)
                self.load_data(fname[:idx+14]+"_lockout"+fname[idx+14:])
                self.plot_trends(ctr,'lockout', ax2)
                ctr+=1
        plt.suptitle(animal_name)

    def get_sessions(self):

        # select data with or without lockout
        prefix1 = ''
        if self.lockout:
            prefix1 = '_lockout_'+str(self.lockout_window)+"sec"

        prefix3 = ''
        if self.compute_roi == True:
            prefix3 = "ROI_"

        # select data with pca compression
        prefix2 = ''
        if self.pca_flag:
            prefix2 = '_pca_'+str(self.pca_var)

        # load sessions in chronological order
        self.sessions = np.load(os.path.join(self.main_dir, self.animal_id,'tif_files.npy'))

        #
        self.fnames_svm = []
        self.session_ids = []
        self.ideal_sliding_windows = []
        for k in range(len(self.sessions)):
            self.session = os.path.split(self.sessions[k])[1][:-4]
            try:
                self.session = str(self.session,'utf-8')
            except:
                pass

            if self.concatenated_flag==True:
                fname = os.path.join(self.main_dir, self.animal_id,
                     'tif_files',
                     self.session,
                     self.session+
                     '_globalPca_min_trials_concatenated200_code_04_30sec_accuracy.npz'
                     )
                self.ideal_sliding_windows.append(self.sliding_window)

            elif self.regular_flag==True:
                # animal IA1 has data in the SVM-prediction directory
                fname = os.path.join(self.main_dir, self.animal_id,
                                     'SVM_Scores/SVM_Scores_'+prefix3+
                                     self.session+
                                     self.code+prefix1+'_trial_ROItimeCourses_'+
                                     str(self.window)+'sec_Xvalid10_Slidewindow'+
                                     str(int(self.sliding_window))+'.npz')

                #
                # data is in the main tif-file session directory
                if os.path.exists(fname)==False:
                    #print ("    missing svm_scores results: ", fname)
                    fname = os.path.join(self.main_dir, self.animal_id,'tif_files',
                                         self.session,
                                         self.session + "_"+
                                         self.code+prefix1+'_trial_ROItimeCourses_'+
                                         str(self.window)+'sec_Xvalid10_Slidewindow'+
                                         str(int(self.sliding_window))+'.npz')
                self.ideal_sliding_windows.append(self.sliding_window)

            elif self.ideal_window_flag==True:
                #
                fname_ideal_len = os.path.join(self.main_dir, self.animal_id,'tif_files',self.session,
                                         self.session+"_ideal_window_len.npy")
                try:
                    self.sliding_window = np.load(fname_ideal_len)[0]
                    self.ideal_sliding_windows.append(self.sliding_window)
                except:
                    print (" No ideal sliding window computed, skipping", fname_ideal_len)
                    continue

                print ("IDEAL SLIDING WINDOW: ", self.sliding_window)

                fname = os.path.join(self.main_dir, self.animal_id,
                                     'SVM_Scores/SVM_Scores_'+prefix3+
                                     self.session+
                                     self.code+prefix1+'_trial_ROItimeCourses_30sec_Xvalid10_Slidewindow'+
                                     str(self.sliding_window)+'.npz')

            self.session_ids.append(self.session)

            self.fnames_svm.append(fname)


    def plot_animal_time_longitudinal(self, animal_name):
        #animal_names = ['IA1','IA2','IA3','IJ1','IJ2','AQ2']

        idx=self.animal_ids.index(animal_name)

        #
        root_dir = self.main_dir+animal_name+'/'
        fnames = np.sort(glob2.glob(root_dir+'SVM_scores_'+animal_name+"*"))

        #
        fig=plt.figure()
        ax1=plt.subplot(1,2,1)
        ax1.set_title("All")
        ax2=plt.subplot(1,2,2)
        ax2.set_title("Lockout")

        self.n_colors = self.ncolors[idx]
        ctr=0
        for fname in fnames:
            if 'lockout' not in fname:
                print (fname)
                self.load_data(fname)
                self.plot_trends(ctr,'all',ax1)

                idx = fname.find('SVM_scores_'+animal_name)
                fname2 = fname[:idx+12]+"_lockout"+fname[idx+12:]
                print (fname2)
                self.load_data(fname[:idx+14]+"_lockout"+fname[idx+14:])
                self.plot_trends(ctr,'lockout', ax2)
                ctr+=1

        plt.suptitle(animal_name)



    def plot_animal_decision_AUC_longitudinal(self):

            #
        ax1=plt.subplot(121)
        ax2=plt.subplot(122)
        #
        for animal_name in self.animal_ids:
            #
            root_dir = self.main_dir+animal_name+'/'
            fnames = np.sort(glob2.glob(root_dir+'SVM_scores_'+animal_name+"*"))

            #
            width = [0,20]

            #
            auc1 = []
            auc2 = []
            for fname in fnames:
                if 'lockout' not in fname:
                    self.load_data(fname)
                    auc1.append(self.data.mean(1)[width[0]:width[1]].sum(0))

                    # load lockout version
                    idx = fname.find('SVM_scores_'+animal_name)
                    fname2 = fname[:idx+12]+"_lockout"+fname[idx+12:]
                    self.load_data(fname[:idx+14]+"_lockout"+fname[idx+14:])
                    auc2.append(self.data.mean(1)[width[0]:width[1]].sum(0))

            #
            auc1 = np.array(auc1)
            auc2 = np.array(auc2)
            t = np.arange(auc1.shape[0])/(auc1.shape[0]-1)

            #
            print ("anmial: ", animal_name, t.shape, auc1.shape)
            temp2 = np.poly1d(np.polyfit(t, auc1, 1))(t)
            ax1.plot(t, temp2,
                     linewidth=4,
                     c=self.colors[self.clr_ctr],
                    label=self.animal_ids[self.clr_ctr])

            #
            ax2.plot(t, np.poly1d(np.polyfit(t, auc2, 1))(t),
                     '--', linewidth=4,
                     c=self.colors[self.clr_ctr])

            self.clr_ctr+=1

        ax1.set_xlim(0,1)
        ax2.set_xlim(0,1)
        ax1.set_ylim(9,13)
        ax2.set_ylim(9,13)
        ax1.set_title("All")
        ax2.set_title("Lockout")
        ax1.set_xlabel("Duration of study")
        ax2.set_xlabel("Duration of study")
        plt.suptitle("AUC fits to SVM decision prediction", fontsize=20)
        ax1.legend(fontsize=20)


    def plot_decision_time(self, clr, label, ax=None):

        #
        if ax is None:
            ax=plt.subplot(111)

        #
        t = np.arange(-9.5, 0.5, 1)

        #
        print (self.data.shape)

        temp = []
        for k in range(self.data.shape[1]):
            temp.append(self.data[:,k,k])
        temp=np.array(temp)

        #
        mean = temp.mean(1)
        std = np.std(temp,axis=1)
        plt.plot(t, mean,
                 c=clr,
                 label = label,
                 linewidth=4)
        plt.fill_between(t, mean-std, mean+std, color=clr, alpha = 0.1)

        plt.legend(fontsize=16)
        self.format_plot2(ax)



    def plot_decision_time_animal(self, animal_name):

        # select dataset and # of recordings
        t = np.arange(-9.5, 0.5, 1)
        idx=self.animal_ids.index(animal_name)

        colors = plt.cm.magma(np.linspace(0,1,self.n_colors[idx]))


        root_dir = self.main_dir+animal_name+'/'
        fnames = np.sort(glob2.glob(root_dir+'conf_10_'+animal_name+"*"))

        #
        traces1 = []
        traces2 = []
        for fname in fnames:
            if 'lockout' not in fname:
                self.load_data(fname)
                temp = []
                for k in range(self.data.shape[1]):
                    temp.append(self.data[:,k,k])
                traces1.append(temp)

                # load lockout version
                idx = fname.find('conf_10_'+animal_name)
                fname2 = fname[:idx+11]+"_lockout"+fname[idx+11:]
                self.load_data(fname2)
                temp = []
                for k in range(self.data.shape[1]):
                    temp.append(self.data[:,k,k])
                traces2.append(temp)

        traces1=np.array(traces1)
        traces2=np.array(traces2)
        #print (traces1.shape)
        #
        ax1=plt.subplot(1,2,1)
        ax1.set_title("all")
        for k in range(traces1.shape[0]):
            mean=traces1[k].mean(1)
            #print (mean.shape)
            ax1.plot(t, mean,
                     c=colors[k],
                     linewidth=4)

        self.format_plot2(ax1)

        #
        ax2=plt.subplot(1,2,2)
        ax2.set_title("lockout")
        for k in range(traces2.shape[0]):
            mean=traces2[k].mean(1)
            #print (mean.shape)
            ax2.plot(t, mean,
                     c=colors[k],
                     linewidth=4)
        self.format_plot2(ax2)

        plt.suptitle(animal_name)


    #
    # def save_edts_body_movements(self, animal_ids, codes):
    #
    #     #
    #     for animal_id in animal_ids:
    #         self.animal_id= animal_id
    #
    #         #
    #         fnames_good = self.main_dir+self.animal_id + '/tif_files/sessions_DLC_alignment_good.txt'
    #
    #         import csv
    #         sessions = []
    #         shift_ids = []
    #         with open(fnames_good, newline='') as csvfile:
    #             spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #             for row in spamreader:
    #                 sessions.append(str(row[0]).replace(',',''))
    #                 shift_ids.append(row[1])
    #
    #         print ("SESSIONS: ", sessions)
    #         print ("SHIFT IDS: ", shift_ids)
    #
    #         ############################
    #
    #         #
    #         edt = []
    #         n_trials = []
    #
    #         # loop over all sessions
    #         for ctr,session_id in enumerate(sessions):
    #             edt.append([])
    #             n_trials.append([])
    #             # get the shifted data by matching session ID with all sessions
    #             self.session_id = session_id
    #             for k in range(len(sessions)):
    #                 if self.session_id in sessions[k]:
    #                     ctr_plt = k
    #                     break
    #
    #             # loop over all limb movements
    #             self.xlim = [-15, 0]
    #             for code in codes:
    #
    #                 self.code = code
    #                 self.shift_id_str = shift_ids[ctr_plt]
    #
    #                 # GET FILENAME IF EXISTS
    #                 self.get_fname()
    #                 if self.fname is None:
    #                     print ("no file, exiting")
    #                     edt[ctr].append(np.nan)
    #                     n_trials[ctr].append(0)
    #                     continue
    #
    #                 # PROCESS SESSION
    #                 self.process_session()
    #                 if self.n_trials==0 or self.data.shape[0]==0:
    #                     #print ("insufficient trials ", labels[i])
    #                     edt[ctr].append(np.nan)
    #                     n_trials[ctr].append(0)
    #                     continue
    #
    #                 # GET SIGNIFICANCE AND EDT
    #                 self.compute_significance()
    #                 edt[ctr].append(self.edt)
    #                 n_trials[ctr].append(self.n_trials)
    #                 print ("*******************************")
    #                 print ("    EDT: ", code, " ", self.edt)
    #                 print ("*******************************")
    #                 print ('')
    #                 print ('')
    #
    #         n_trials = np.array(n_trials)
    #         edt = np.array(edt)
    #         print (edt.shape)
    #
    #         # data_dir = '/media/cat/4TBSSD/yuki/'
    #         np.savez(self.main_dir + animal_id+"_edt_body_parts.npz",
    #                 edt = edt,
    #                 n_trials = n_trials)


    def plot_decision_time_animal_matrix(self, animal_name):

        #
        root_dir = self.main_dir+animal_name+'/'
        fnames = np.sort(glob2.glob(root_dir+'conf_10_'+animal_name+"*"))

        #
        traces1 = []
        traces2 = []
        for fname in fnames:
            if 'lockout' not in fname:
                self.load_data(fname)
                temp = []
                for k in range(self.data.shape[1]):
                    temp.append(self.data[:,k,k].mean(0))
                traces1.append(temp)

                # load lockout version
                idx = fname.find('conf_10_'+animal_name)
                fname2 = fname[:idx+11]+"_lockout"+fname[idx+11:]
                self.load_data(fname2)
                temp = []
                for k in range(self.data.shape[1]):
                    temp.append(self.data[:,k,k].mean(0))
                traces2.append(temp)

        traces1=np.array(traces1)
        traces2=np.array(traces2)
        print (traces1.shape)
        #
        ax1=plt.subplot(1,4,1)
        ax1.set_title("all")
        ax1.imshow(traces1,vmin=0,vmax=1.0)

        #
        ax2=plt.subplot(1,4,2)
        ax2.set_title("lockout")
        ax2.imshow(traces2,vmin=0,vmax=1.0)
        plt.suptitle(animal_name)

        #
        ax2=plt.subplot(1,4,3)
        ax2.set_title("all - lockout")
        ax2.imshow(traces1-traces2,vmin=0,vmax=.25)
        plt.suptitle(animal_name)

        #
        ax2=plt.subplot(1,4,4)
        ax2.set_title("lockout - all")
        ax2.imshow(traces2-traces1,vmin=0,vmax=.25)
        plt.suptitle(animal_name)

    def plot_decision_time_all_matrix(self):


        vmin = 0
        vmax = 0.75
        axes=[]
        fig=plt.figure()
        for a in range(6):
            axes.append(plt.subplot(2,3,a+1))
            #
            root_dir = self.main_dir+self.animal_ids[a]+'/'
            fnames = np.sort(glob2.glob(root_dir+'conf_10_'+self.animal_ids[a]+"*"))

            #
            traces1 = []
            for fname in fnames:
                if 'lockout' not in fname:
                    self.load_data(fname)
                    temp = []
                    for k in range(self.data.shape[1]):
                        temp.append(self.data[:,k,k].mean(0))
                    if self.filter:
                        temp = self.filter_trace(temp)
                    traces1.append(temp)

            traces1=np.array(traces1)
            axes[a].set_title(self.animal_ids[a])
            im = axes[a].imshow(traces1,vmin=vmin,vmax=vmax)

            plt.xticks(np.arange(0,traces1.shape[1],2),
                               np.arange(-9.5,0,2))
            plt.ylabel("Study day")

        ticks = np.linspace(vmin,vmax,4)
        cbar = fig.colorbar(im, ax=axes[5],
                            ticks=ticks)
        #cbar.ax.tick_params(labelsize=16)
        #cbar.ax.set_title('Pval', rotation=0,
        #                 fontsize=16)


    def filter_trace(self,trace):

        box = np.ones(self.smooth_window)/self.smooth_window
        trace_smooth = np.convolve(trace, box, mode='valid')

        return trace_smooth


    def compare_svm_rnn(self, fnames_svm, fnames_rnn):

        #
        ctr=1
        for fname in fnames_rnn:
            idx1 = fname.find('RNN_scores/')
            idx2 = fname.find('_200')
            session = fname[idx1+11:idx2]
            for fname_svm in fnames_svm:
                if session in fname_svm and 'trial' in fname_svm:
                    data_svm = np.load(fname_svm)[:300]
                    break

            #
            ax=plt.subplot(5, 10,ctr)
            data_rnn = np.load(fname)['b_rnn']
            t = np.linspace(-10,0,data_rnn.shape[0])
            std = np.std(data_rnn,1)
            mean = data_rnn.mean(1)
            plt.plot(t, mean, c='black')
            plt.fill_between(t, mean+std, mean-std, color='black', alpha=.2)


            mean = data_svm.mean(1)
            self.smooth_window = 30
            mean = self.filter_trace(mean)
            std = np.std(data_svm,1)[15:-14]

            t = np.linspace(-10,0,mean.shape[0])
            plt.plot(t,mean,c='blue')
            plt.fill_between(t, mean+std, mean-std, color='blue', alpha=.2)

            plt.ylim(0.4,1.0)
            plt.title(os.path.split(fname_svm)[1][11:25],fontsize=8)
            plt.plot([-10,0],[0.5,0.5],'r--')
            plt.xlim(-10,0)
            if ctr!=31:
                plt.yticks([])
                plt.xticks([])

            ctr+=1

        plt.suptitle("Decision choice: RNN (black) vs. SVM (blue) ",fontsize=20)
        plt.show()

def get_lever_offset(main_dir,
                     animal_id,
                     session_corrected):

    fname_lever_offset = os.path.join(main_dir,
                                     animal_id,
                                     'tif_files',
                                     session_corrected,
                                     session_corrected+"_lever_offset_n_frames.txt")

    if os.path.exists(fname_lever_offset)==False:

        images_file = fname_lever_offset.replace('_lever_offset_n_frames.txt','_aligned.npy')

        aligned_images = np.load(images_file)

        # Find blue light on/off
        blue_light_threshold = 400  #Intensity threshold; when this value is reached - imaging light was turned on
        start_blue = 0; end_blue = aligned_images.shape[0]

        if np.average(aligned_images[0])> blue_light_threshold:    #Case #1: imaging starts with light on; need to remove end chunk; though likely bad recording
            for k in range(len(aligned_images)):
                if np.average(aligned_images[k])< blue_light_threshold:
                    #self.aligned_images = self.aligned_images[k:]
                    end_blue = k
                    break
        else:                                                           #Case #2: start with light off; remove starting and end chunks;
            #Find first light on
            for k in range(len(aligned_images)):
                if np.average(aligned_images[k])> blue_light_threshold:
                    start_blue = k
                    break

            #Find light off - count backwards from end of imaging data
            for k in range(len(aligned_images)-1,0,-1):
                if np.average(aligned_images[k])> blue_light_threshold:
                    end_blue= k
                    break

        lever_offset = start_blue

        #np.savetxt(fname_lever_offset, [self.lever_offset])

    else:
        lever_offset = int(np.loadtxt(fname_lever_offset))

    return lever_offset

#
def get_sessions(main_dir,
                 animal_id,
                 session_id):
     # load ordered sessions from file
    sessions = np.load(os.path.join(main_dir,
                                         animal_id,
                                         'tif_files.npy'))
    # grab session names from saved .npy files
    data = []
    for k in range(len(sessions)):
        data.append(os.path.split(sessions[k])[1].replace('.tif',''))
    sessions = data

    #
    if session_id != 'all':
        final_session = []
        for k in range(len(sessions)):
            if session_id in sessions[k]:
                final_session = [sessions[k]]
                break
        sessions = final_session

    # fix binary string files issues; remove 'b and ' from file names
    for k in range(len(sessions)):
        sessions[k] = str(sessions[k]).replace("'b",'').replace("'","")
        if sessions[k][0]=='b':
            sessions[k] = sessions[k][1:]

    sessions = np.array(sessions)

    return sessions


def load_trial_times_whole_stack(root_dir,
                                 animal_id,
                                 session,
                                 no_movement):

    # grab movement initiation arrays
    fname = os.path.join(root_dir, animal_id,'tif_files',
                         session,
                         session+'_'+
                         str(no_movement)+"secNoMove_movements.npz"
                         )

    # if no file return empty arrays?
    if os.path.exists(fname)==False:
        print ("FILE I SMISSING: ", fname)
        feature_quiescent = []
        #
        for k in range(7):
            feature_quiescent.append([])

        return None, None, None
    #
    data = np.load(fname, allow_pickle=True)
    feature_quiescent = data['feature_quiescent']
    all_quiescent = data['all_quiescent']

    # load rewarded lever pull trigger times also
    code_04_times, code_04_times_lockout = load_code04_times(root_dir,
                                                              animal_id,
                                                              no_movement,
                                                              session)
    code_04_times = np.array((code_04_times, code_04_times)).T
    shift_lever_to_ca = get_lever_offset_seconds(root_dir,
                                                 animal_id,
                                                 session
                                                 )
    print ("Lever to [Ca] shift: ", shift_lever_to_ca)

    #
    bins = np.arange(-10,10,1/15.)

    #
    try:
        res = pycorrelate.pcorrelate(code_04_times[:,1],
                                 np.array(feature_quiescent[1])[:,1],
                                 bins=bins)
    except Exception as e:
        print ("Exception : ", e)
		
        try:
            res = pycorrelate.pcorrelate(code_04_times[:,1],
                         np.array(feature_quiescent[0])[:,1],
                         bins=bins)
        except Exception as e:
            print ("Exception: ", e)
            return None, None, None


    argmax = np.argmax(res)
    shift_DLC_to_ca = bins[argmax]

    #
    # shift_DLC_to_ca = get_DLC_shift_seconds(root_dir,
    #                                         animal_id,
    #                                         session,
    #                                         session_number)

    print ("DLC to [Ca] shift: ", shift_DLC_to_ca)

    #code_04_times += shift_lever_to_ca

    #load_code04_times = code_04_times
    #feature_quiescent = feature_quiescent

    #
    temp_ = []
    for k in range(len(feature_quiescent)):
        temp_.append(np.array(feature_quiescent[k])-shift_lever_to_ca)
    temp_.append(all_quiescent)
    temp_.append(code_04_times - shift_DLC_to_ca - shift_lever_to_ca)

    return temp_, code_04_times, feature_quiescent



def get_DLC_shift_seconds(main_dir,
                          animal_id,
                          session,
                          session_number):

    fnames_good = os.path.join(main_dir,animal_id,'tif_files',
                  'sessions_DLC_alignment_good.txt')

    import csv
    sessions = []
    shift_ids = []
    with open(fnames_good, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            sessions.append(str(row[0]))
            shift_ids.append(row[1])

    shift_id_str = shift_ids[session_number]

    fname_correlate = os.path.join(main_dir, 
								 animal_id,
								 'tif_files', 
								 session,
								'correlate.npz')

    try:
        data = np.load(fname_correlate,allow_pickle=True)
    except:
        print( " ... data missing", fname_correlate)
        return None

    cors = data['cors'].squeeze().T

    #vis.shift = 0
    print ("sessoin ID: ", session_number, "  left/right paw/lever ID used: ", shift_id_str)
    if len(shift_id_str)>1:
        shift_id = int(shift_id_str[0])
        shift_additional = float(shift_id_str[1:])
    else:
        shift_id = int(shift_id_str)
        shift_additional = 0

    print ( " using shift: ", shift_id+shift_additional)

    corr_featur_id = shift_id

    temp_trace = cors[:,corr_featur_id]
    temp_trace[:2000] = 0
    temp_trace[-2000:] = 0
    shift = round(np.argmax(temp_trace)/1000. - 15.,2)+shift_additional
    print ("SHIFT Loaded: ", shift)

    return shift

def load_code04_times(root_dir,
                      animal_id,
                      lockout_window,
                      recording):

    #
    try:
        fname = os.path.join(root_dir,animal_id, 'tif_files',recording,
                             recording + '_locs44threshold.npy')
        locs_44threshold = np.load(fname)
    except:
        print ("locs 44 thrshold missing", recording)
        locs_code04 = np.zeros((0),'float32')
        locs_code04_lockout = np.zeros((0),'float32')
        return locs_code04, locs_code04_lockout

    #
    codes = np.load(os.path.join(root_dir,animal_id, 'tif_files',recording,
                             recording + '_code44threshold.npy'))
    code = b'04'
    idx = np.where(codes==code)[0]
    locs_selected = locs_44threshold[idx]

    if locs_selected.shape[0]==0:
        locs_code04 = np.zeros((0),'float32')
        locs_code04_lockout = np.zeros((0),'float32')
        return locs_code04, locs_code04_lockout

    diffs = locs_selected[1:]-locs_selected[:-1]
    idx = np.where(diffs>lockout_window)[0]

    #
    locs_selected_with_lockout = locs_selected[idx+1]
    if locs_selected_with_lockout.shape[0]==0:
        locs_code04 = np.zeros((0),'float32')
        locs_code04_lockout = np.zeros((0),'float32')
        return locs_code04, locs_code04_lockout

    # ADD FIRST VAL
    if locs_selected[0]>lockout_window:
        locs_selected_with_lockout = np.concatenate(([locs_selected[0]], locs_selected_with_lockout), axis=0)

    locs_code04 = locs_selected
    locs_code04_lockout = locs_selected_with_lockout

    return locs_code04, locs_code04_lockout


def get_lever_offset_seconds(main_dir,
                     animal_id,
                     session_corrected,
                     imaging_rate=30):

    fname_lever_offset = os.path.join(main_dir,
                                     animal_id,
                                     'tif_files',
                                     session_corrected,
                                     session_corrected+"_lever_offset_n_frames.txt")

    if os.path.exists(fname_lever_offset)==False:

        images_file = fname_lever_offset.replace('_lever_offset_n_frames.txt','_aligned.npy')

        aligned_images = np.load(images_file)

        # Find blue light on/off
        blue_light_threshold = 400  #Intensity threshold; when this value is reached - imaging light was turned on
        start_blue = 0; end_blue = aligned_images.shape[0]

        if np.average(aligned_images[0])> blue_light_threshold:    #Case #1: imaging starts with light on; need to remove end chunk; though likely bad recording
            for k in range(len(aligned_images)):
                if np.average(aligned_images[k])< blue_light_threshold:
                    #self.aligned_images = self.aligned_images[k:]
                    end_blue = k
                    break
        else:                                                           #Case #2: start with light off; remove starting and end chunks;
            #Find first light on
            for k in range(len(aligned_images)):
                if np.average(aligned_images[k])> blue_light_threshold:
                    start_blue = k
                    break

            #Find light off - count backwards from end of imaging data
            for k in range(len(aligned_images)-1,0,-1):
                if np.average(aligned_images[k])> blue_light_threshold:
                    end_blue= k
                    break

        lever_offset = start_blue

        #np.savetxt(fname_lever_offset, [self.lever_offset])

    else:
        lever_offset = int(np.loadtxt(fname_lever_offset))

    return lever_offset/imaging_rate


#
def plot_vertical_histograms(animal_ids,
                             code_ids,
                             clrs,
                             window,
                             return_ids_threshold):

    all_res1 = []
    all_res2 = []
    for animal_id in animal_ids:
        fname = '/media/cat/4TBSSD/yuki/'+animal_id+'_edt_body_parts.npz'

        data = np.load(fname, allow_pickle=True)
        #

        edt1 = data['edt'][:,code_ids[0]]
        all_res1.append(edt1)

        edt2 = data['edt'][:, code_ids[1]]
        all_res2.append(edt2)

        #
        if return_ids_threshold is not None:
            #for k in range(len(edt1)):
            idx = np.where(edt1<=return_ids_threshold)[0]
            if idx.shape[0]>0:
                print (animal_id, "session 1 with EDT < threshold: ", idx)

            idx = np.where(edt2<=return_ids_threshold)[0]
            if idx.shape[0]>0:
                print (animal_id, "session 2 with EDT < threshold: ", idx)
    #
    data_sets_all = []
    for k in range(len(all_res1)):
        data_sets_all.append(all_res1[k])

    #
    data_sets_lockout = []
    for k in range(len(all_res2)):
        data_sets_lockout.append(all_res2[k])

    # Computed quantities to aid plotting
    hist_range = (-window,1)
    bins = np.arange(-window,1,1)

    #
    binned_data_sets_all = [
        np.histogram(d, range=hist_range, bins=bins)[0]
        for d in data_sets_all
    ]

    binned_data_sets_lockout = [
        np.histogram(d, range=hist_range, bins=bins)[0]
        for d in data_sets_lockout
    ]

    #
    binned_maximums = np.max(binned_data_sets_all, axis=1)
    spacing = 30
    x_locations = np.arange(0, spacing*6,spacing)

    # The bin_edges are the same for all of the histograms
    bin_edges = np.arange(hist_range[0], hist_range[1],1)
    centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[1:]#[:-1]
    heights = np.diff(bin_edges)

    # Cycle through and plot each histogram
    fig, ax = plt.subplots(figsize=(6,5))
    for x_loc, binned_data, binned_data_lockout in zip(x_locations, binned_data_sets_all, binned_data_sets_lockout):
        lefts = x_loc - 0.3# * binned_data
        ax.barh(centers, -binned_data, height=heights, left=lefts, color=clrs[code_ids[0]])

        lefts = x_loc #- 0.5 * binned_data_lockout
        ax.barh(centers, binned_data_lockout, height=heights, left=lefts, color=clrs[code_ids[1]])

    ax.set_xticks(x_locations)
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_xticklabels(labels)
    ax.set_xlim(-15,spacing*6-5)
    ax.set_ylim(-window-0.5,0)
    #ax.set_ylabel("Data values")
    #ax.set_xlabel("Data sets")


def plot_pca_timecourses(animal_id,
                         session_id,
                        class_type,
                        n_pca,
                        xlim,
                         plotting):

    class_types = ['trial','random']
    clrs = ['blue','black']
    ctr=0
    if plotting:
        fig =plt.figure()
    for class_type in class_types:
        #
        fname = '/media/cat/4TBSSD/yuki/'+animal_id+'/tif_files/'+session_id+"/"+session_id+'_code_04_trial_ROItimeCourses_30sec_pca_0.95.npy'
        fname = fname.replace('trial',class_type)
        print ("fname: ", fname)

        fname_spatial = '/media/cat/4TBSSD/yuki/'+animal_id+'/tif_files/'+session_id+"/"+session_id+'_code_04_trial_ROItimeCourses_30sec_pca_0.95_spatial.npy'
        pca_space =np.load(fname_spatial).reshape(-1, 128,128)
        print ("pca_space: ", pca_space.shape)

        #
        pca_trials = np.load(fname)
        pca_trials = pca_trials[:,:,:900]
        n_pca = min(pca_trials.shape[1],n_pca)
        pca_trials = pca_trials[:,:n_pca,:900]
        print ("pca_trials: ", pca_trials.shape)

        #
        means = np.mean(pca_trials,0)
        std = np.std(pca_trials,0)
        t = np.arange(means.shape[1])/30 - 30.

        if class_type=='trial':
            ylims = []

        if plotting:
            for k in range(means.shape[0]):
                ax=plt.subplot(3,n_pca,ctr+1)

                for p in range(pca_trials.shape[0]):
                    plt.plot(t,pca_trials[p,k],
                             c=clrs[ctr//n_pca],
                             alpha=.1)
                    if p>50:
                        break

                plt.plot(t,means[k],
                        linewidth=2,
                        c='red',
                        label="PC: "+str(k+1))
                plt.legend(fontsize=20)


                if class_type=='trial':
                    vmax = max(-np.min(means[k]),
                               np.max(means[k]))*5
                    plt.ylim(-vmax,vmax)
                    ylims.append(vmax)
                else:
                    plt.ylim(-ylims[k],ylims[k])


                if ctr==0:
                    plt.ylabel("Trials")
                elif ctr==n_pca:
                    plt.ylabel("Random")

                plt.xlim(xlim[0],0)
                ctr+=1
            plt.xlim(t[0],t[-1])


    if plotting:
        for k in range(n_pca):
            ax=plt.subplot(3,n_pca,ctr+1)
            if k==0:
                plt.ylabel("PCA loadings")
            plt.imshow(pca_space[k])
            ctr+=1

        plt.suptitle(session_id+ "  # trials: "+str(pca_trials.shape[0]), fontsize=20)
        plt.show()

    return pca_trials



class PCA_Analysis():

    def __init__(self):
        self.clr_ctr = 0

#
def project_data_pca2(pa):

    # p_lever = np.zeros((pa.n_frames,
    #                     pa.triggers.shape[0],
    #                     pa.n_pca))
    print (pa.n_frames,
           pa.triggers.shape[0],
           pa.n_pca)

    #
    # total_frames = pa.n_frames
    p_lever = []

    # loop over
    for t in trange(pa.n_frames):
        arr = []
        #if True:

        #
        try:
            # loop over every lever pull/behavior instance
            for k in range(pa.triggers.shape[0]):

                # grab a snippet of raw data (PCA compressed usually)
                # this snippet ranges from two time points
                temp = pa.X[pa.triggers[k]-pa.t1-t:pa.triggers[k]-pa.t2-t]

                # make sure we are not out of bounds
                if temp.shape[0] == (pa.t1-pa.t2):
                    temp = temp.reshape(temp.shape[0],
                                   -1)
                    #
                    arr.append(temp.reshape(-1))

            #
            arr = np.array(arr)
            res = pa.pca.transform(arr)
            #p_lever[t] = res
            p_lever.append(res)

        except:
            pass


    p_lever = np.array(p_lever)
    #
    # # remove any skipped values
    # if total_frames<pa.n_frames:
    #     p_lever = p_lever[:total_frames]

    #print (" pca p_levefr resahped: ", p_lever.shape)

    return p_lever

def get_data_and_triggers(pa):

    #
    fname_triggers = os.path.join(pa.root_dir,
                                  pa.animal_id,
                                  'tif_files',
                                  pa.session,
                                  'blue_light_frame_triggers.npz')

    fname_data = os.path.join(pa.root_dir,
                                 pa.animal_id,
                                 'tif_files',
                                 pa.session,
                                 pa.session+ '_aligned_butterworth_0.1hz_6.0hz.npy')

    meta = np.load(fname_triggers)
    data = np.load(fname_data)
#    except:
 #       print (" data missing/led missing")
  #      return None, None

    end_blue = meta['end_blue']
    start_blue = meta['start_blue']
    triggers = meta['img_frame_triggers']

    data_led = data[start_blue:end_blue]

    X = data_led.reshape(data_led.shape[0],-1)

    return X, triggers


def get_pca_object_and_all_points(pa):

    fname_pickle = os.path.join(pa.root_dir, pa.animal_id,'tif_files',pa.session,
                                pa.session+"_pca_scatter_plot.pk")

    fname_all_points = os.path.join(pa.root_dir, pa.animal_id,'tif_files',pa.session,
                                pa.session+"_all_points.npy")
    X_30 = []
    for k in range(0,pa.X.shape[0]-100,pa.sliding_window):
        X_30.append(pa.X[k:k+pa.sliding_window])

    #
    X_30 = np.array(X_30)
    X_30 = X_30.reshape(X_30.shape[0],-1)
    print(" X data using : ", pa.sliding_window, " number of frames ", X_30.shape)

    if os.path.exists(fname_all_points)==False:

        # PCA ON ALL DATA
        pca = PCA(n_components=pa.n_pca)
        pca.fit(X_30)
        print (" done fit ")

        # do
        all_points = pca.transform(X_30)
        print ("all points denoised: ", all_points.shape)

        with open(fname_pickle, 'wb') as f:
            pickle.dump(pca, f)

        np.save(fname_all_points,
                all_points)
    else:

        with open(fname_pickle, "rb") as f:
            pca = pickle.load(f)
        all_points = np.load(fname_all_points)

    return pca, all_points


def get_umap_object_and_all_points(pa):

    fname_pickle = os.path.join(pa.root_dir, pa.animal_id,'tif_files',pa.session,
                                pa.session+"_umap_scatter_plot.pk")

    fname_all_points = os.path.join(pa.root_dir, pa.animal_id,'tif_files',pa.session,
                                pa.session+"_umap_all_points.npy")
    # use raw 128 x 128 data
    if False:
        X_30 = []
        for k in range(0,pa.X.shape[0]-100,pa.sliding_window):
            X_30.append(pa.X[k:k+pa.sliding_window])
        X_30 = np.array(X_30)
        X_30 = X_30.reshape(X_30.shape[0],-1)
    else:
        X_30 = pa.all_points

    print(" X data using : ", pa.sliding_window, " number of frames ", X_30.shape)

    if os.path.exists(fname_all_points)==False:

        # PCA ON ALL DATA

        import umap
        umap_3d = umap.UMAP(n_components=3,
                            init='random',
                            random_state=0)

        #fit = umap.UMAP()
        print ("Fitting UMAP...")
        umap_3d.fit(X_30)
        print ("  ... done fit ")

        # do
        print (" denoising all data using UMAP ....", X_30.shape)
        all_points = umap_3d.transform(X_30)
        #all_points = pca.transform(X_30)
        print ("all points denoised: ", all_points.shape)

        with open(fname_pickle, 'wb') as f:
            pickle.dump(umap_3d, f)

        np.save(fname_all_points,
                all_points)
    else:

        with open(fname_pickle, "rb") as f:
            umap_3d = pickle.load(f)
        all_points = np.load(fname_all_points)

    return umap_3d, all_points



def project_data_pca(pa):

    # p_lever = np.zeros((pa.n_frames,
    #                     pa.triggers.shape[0],
    #                     pa.n_pca))
    print (pa.n_frames,
           pa.triggers.shape[0],
           pa.n_pca)

    #
    # total_frames = pa.n_frames
    p_lever = []
    for t in trange(pa.n_frames):
        arr = []
        #if True:
        try:
            for k in range(pa.triggers.shape[0]):
                temp = pa.X[pa.triggers[k]-pa.sliding_window -t:pa.triggers[k]-t]
                if temp.shape[0]== pa.sliding_window:
                    temp = temp.reshape(temp.shape[0],
                                   -1)
                    #X_lever[t,k]= temp
                    arr.append(temp.reshape(-1))

            arr = np.array(arr)
            res = pa.pca.transform(arr)
            #p_lever[t] = res
            p_lever.append(res)

        except:
            pass
    p_lever = np.array(p_lever)
    #
    # # remove any skipped values
    # if total_frames<pa.n_frames:
    #     p_lever = p_lever[:total_frames]

    #print (" pca p_levefr resahped: ", p_lever.shape)

    return p_lever



def project_data_umap(pa):

    # DIMENSION OF DATA IN:
    # pa.p_lever = (1, 55, 10)

    #
    p_lever = []
    for t in trange(pa.n_frames):
        #for k in range(pa.p_lever.shape[1]):
        #    temp = pa.p_lever[t,k]

        arr = pa.p_lever[t]
        print ("Umap applid to arr: ", arr.shape)
        res = pa.umap.transform(arr)
        #p_lever[t] = res
        p_lever.append(res)

    p_lever = np.array(p_lever)

    print (" umap p_levefr resahped: ", p_lever.shape)

    return p_lever

#
def plot_pca_scatter_multi(pa,
                     n_frames = 30,
                     clr = 'red',
                     plot_all = True,
                     plot_3D=True,
                     plot_flag = True):

    # use knn triage to remove most outlier points
    triage_value = 0.0008
    knn_triage_threshold_all_points = 100*(1-triage_value)


    # apply knn to all points
    temp_points = pa.all_points[:,:2]
    #print ("temp points: ", temp_points.shape)
    idx_keep = knn_triage(knn_triage_threshold_all_points, temp_points)
    idx_keep = np.where(idx_keep==1)[0]
    all_points_knn = temp_points[idx_keep]

    # apply knn to lever/body movments
    triage_value = 0.1
    knn_triage_threshold_lever = 100*(1-triage_value)

    temp_points = pa.p_lever[0,:,:2]
    #print ("temp points: ", temp_points.shape)
    idx_keep = knn_triage(knn_triage_threshold_lever, temp_points)
    idx_keep = np.where(idx_keep==1)[0]
    p_lever_knn = temp_points[idx_keep]


    ################################################
    ###############################################
    ##############################################
    if plot_flag:
        if plot_3D:
            ax = fig.add_subplot(projection='3d')
        else:
            ax =plt.subplot(111)

    # clrs = ['red','pink','yellow']
    # cmap = matplotlib.cm.get_cmap('jet_r')

    # if plot_all:
    #     idx = np.arange(pa.all_points.shape[0])
    #     print (" all points: ", idx.shape)

    #print ("plever: ", p_lever_knn.shape)
    start = 0
    end = n_frames

   #for k in range(X_lever.shape[0]):
    if plot_flag:
        for k in range(start,end,1):
            if plot_3D:
                ax.scatter(p_lever_knn[k,:,0],
                           p_lever_knn[k,:,1],
                           p_lever_knn[k,:,2],
                        color=clr,
                        s=20,
                        edgecolor = 'black', alpha=.8)
            else:
                ax.scatter(p_lever_knn[:,0],
                           p_lever_knn[:,1],
                        color=clr,
                        s=20,
                        edgecolor = 'black', alpha=.8)

    #
    from scipy.spatial import ConvexHull

    # COMPUTE CONVEX HULL FOR ALL POINTS
    points = all_points_knn[:,:2]
    hull = ConvexHull(points)
    pa.points_simplex_all_points = points[hull.simplices]

    # COMPUTE CONVEX HULL FOR LEVER OR BODY PART
    points = p_lever_knn[:,:2]
    if points.shape[0]<3:
        pa.points_simplex = []
    else:
        hull = ConvexHull(points)
        pa.points_simplex = points[hull.simplices]

    ##############################
    if plot_flag:
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], c=clr)
        plt.plot(points[simplex[0], 0], points[simplex[-1], 1], c=clr)


        if plot_3D:
           if plot_all and pa.k==0:
                ax.scatter(all_points_knn [:,0],
                       all_points_knn [:,1],
                       all_points_knn [:,2],
                        c='black',
                        s=20,
                        edgecolor = 'black', alpha=.1)

           ax.scatter(p_lever_knn[:,0],
                       p_lever_knn[:,1],
                       p_lever_knn[:,2],
                        color=clr,
                        s=100,
                        edgecolor = 'black', alpha=.8)

        else:

           if plot_all and pa.k==0:
                ax.scatter(all_points_knn [:,0],
                       all_points_knn [:,1],
                        c='black',
                        s=100,
                        edgecolor = 'black', alpha=.1)

           ax.scatter(p_lever_knn[:,0],
                       p_lever_knn[:,1],
                        color=clr,
                        s=100,
                        edgecolor = 'black', alpha=.8)

    return pa


def plot_pca_scatter(pa):

    #
    ax =plt.subplot(111)

    cmap = pa.cmap

    # idx = np.arange(pa.all_points.shape[0])
    # print (idx.shape)

    #print ("plever: ", pa.p_lever.shape)
    start = 0
    end = pa.n_frames


    #############################################
    ########## PLOT LEVER DYNAMICS ##############
    #############################################
    from sklearn.neighbors import NearestNeighbors
    if pa.knn_triage is not None:
        triage_value = pa.knn_triage
        knn_triage_threshold = 100*(1-triage_value)

        ########### LEVER PULL #############
        p_lever_triaged = []

        # this function triages based only on t=0, i.e. not at every time point; makes better plots
        # also do triage based on first 2 dimensions
        if pa.knn_triage_based_on_last_point:
            temp_points = pa.p_lever[0,:,:2]
            print ("data-in: ", temp_points.shape)

            # print ("temp points: ", temp_points.shape)
            idx_keep = knn_triage(knn_triage_threshold, temp_points)
            idx_keep = np.where(idx_keep==1)[0]

            pa.p_lever_plot = pa.p_lever[:,idx_keep]
            print ("temp_points post: ", pa.p_lever_plot .shape)

        #
        else:
            for k in range(pa.p_lever.shape[0]):
                temp_points = pa.p_lever[k,:,:2]

                # print ("temp points: ", temp_points.shape)
                idx_keep = knn_triage(knn_triage_threshold, temp_points)
                idx_keep = np.where(idx_keep==1)[0]

                temp_points = temp_points[idx_keep]
                p_lever_triaged.append(temp_points)

            p_lever_triaged = np.array(p_lever_triaged)
            pa.p_lever_plot = p_lever_triaged

        ############ ALL POINTS #############
        if pa.plot_all:
            temp_points = pa.all_points

           # print ("all points pre triage: ", temp_points.shape)
            idx_keep = knn_triage(knn_triage_threshold, temp_points)
            idx_keep = np.where(idx_keep==1)[0]
            temp_points = temp_points[idx_keep]
            #print ("all points post triage: ", temp_points.shape)

            pa.all_points_plot = temp_points

    else:
        pa.p_lever_plot = pa.p_lever
        pa.all_points_plot = pa.all_points

    #############################################
    ########## PLOT LEVER DYNAMICS ##############
    #############################################
    # plot all dyanmics - black dots
    if pa.plot_all:
        ax.scatter(pa.all_points_plot[:,0],
                   pa.all_points_plot[:,1],
                    c='black',
                    s=100,
                    #edgecolor = 'black',
                    alpha=pa.alpha2)

    # plot t1 to t2 dynamics
    if pa.plot_dynamics:
        for k in range(start,end,1):
            ax.scatter(pa.p_lever_plot[k,:,0],
                       pa.p_lever_plot[k,:,1],
                        color=cmap(k/(end-start)),
                        s=20,
                        edgecolor = 'black',
                        alpha=pa.alpha1)

    # plot t=0 dynamcs
    if pa.t0_dynamics:
        ax.scatter(pa.p_lever_plot[0,:,0],
           pa.p_lever_plot[0,:,1],
           color='blue',
           s=100,
           edgecolor = 'black',
           alpha=pa.alpha1)

    #############################################
    ####### PLOT CONVEX HULL SIMPLEX ############
    #############################################
    # plot t0 convex hull
    linewidth=pa.linewidth
    if pa.t0_hull:
        points_in = pa.p_lever_plot[0,:,:2]
        simplex = convexhull(points_in) #-np.mean(points_in,axis=0)*2

        clr='blue'
        plot_convex_hull_function(simplex,clr,linewidth)

    ###################################################
    # PLOT T=0..30
    points_in = pa.p_lever_plot[:,:,:2].reshape(-1,2)
    simplex = convexhull(points_in) #-np.mean(points_in,axis=0)*2

    plot_convex_hull_function(simplex,
                              pa.hull_clr,
                              linewidth)

    ###################################################
    # plot all points convex hull
    if pa.plot_all==True:

        points_in = pa.all_points_plot[:,:2]

        # reverse KNN triage to keep just outliers
        if False:
            triage_value = 0.10
            knn_triage_threshold = 100*(1-triage_value)

            # print ("temp points: ", temp_points.shape)
            idx_keep = knn_triage(knn_triage_threshold,
                                  points_in)

            idx_outliers = np.where(idx_keep==0)[0]
            points_in = points_in[idx_outliers]

        #print ("computing hull of all points...", points_in.shape)

        simplex = convexhull(points_in) #-np.mean(points_in,axis=0)*2

        plot_convex_hull_function(simplex,'black',linewidth)

#
def plot_convex_hull_function(simplex, clr, linewidth):

    for k in range(simplex.shape[0]-1):
        plt.plot([simplex[k,0], simplex[k+1,0]],
                 [simplex[k,1], simplex[k+1,1]],
                 c=clr,
                 linewidth=linewidth
                 )
    #
    plt.plot([simplex[0,0], simplex[-1,0]],
             [simplex[0,1], simplex[-1,1]],
             c=clr,
             linewidth=linewidth
             )

#
# def plot_pca_scatter(pa,
#                      n_frames = 30,
#                      plot_all = True,
#                      plot_3D=True):
#
#     fig =plt.figure()
#     if plot_3D:
#         ax = fig.add_subplot(projection='3d')
#     else:
#         ax =plt.subplot(111)
#
#     clrs = ['red','pink','yellow']
#     cmap = matplotlib.cm.get_cmap('jet_r')
#
#     idx = np.arange(pa.all_points.shape[0])
#     print (idx.shape)
#
#     print ("plever: ", pa.p_lever.shape)
#     start = 0
#     end = n_frames
#
#     #
#     #for k in range(X_lever.shape[0]):
#     for k in range(start,end,1):
#         if plot_3D:
#             ax.scatter(pa.p_lever[k,:,0],
#                        pa.p_lever[k,:,1],
#                        pa.p_lever[k,:,2],
#                     color=cmap(k/(end-start)),
#                     s=20,
#                     edgecolor = 'black', alpha=.8)
#         else:
#             ax.scatter(pa.p_lever[k,:,0],
#                        pa.p_lever[k,:,1],
#                     color=cmap(k/(end-start)),
#                     s=20,
#                     edgecolor = 'black', alpha=.8)
#
#     if plot_3D:
#
#         if plot_all:
#             ax.scatter(pa.all_points[idx,0],
#                    pa.all_points[idx,1],
#                    pa.all_points[idx,2],
#                     c='black',
#                     s=20,
#                     edgecolor = 'black', alpha=.2)
#
#         ax.scatter(pa.p_lever[0,:,0],
#                    pa.p_lever[0,:,1],
#                    pa.p_lever[0,:,2],
#                     color='red',
#                     s=100,
#                     edgecolor = 'black', alpha=.8)
#
#     else:
#
#         if plot_all:
#             ax.scatter(pa.all_points[idx,0],
#                    pa.all_points[idx,1],
#                     c='black',
#                     s=100,
#                     edgecolor = 'black', alpha=.8)
#
#         ax.scatter(pa.p_lever[0,:,0],
#                    pa.p_lever[0,:,1],
#                     color='red',
#                     s=100,
#                     edgecolor = 'black', alpha=.8)
#
#     #
#     # if False:
#     #     plt.savefig('/home/cat/pca_all_plus_levers.png',dpi=300)
#     #     plt.close()
#     # else:
#     #     plt.show()
#     #                                    #

def plot_pca_scatter_lever_and_body_movements(pa, plot_3D=True):

    fig =plt.figure()
    if plot_3D:
        ax = fig.add_subplot(projection='3d')
    else:
        ax =plt.subplot(111)

    clrs = ['red','pink','yellow']
    cmap = matplotlib.cm.get_cmap('jet_r')

    idx = np.arange(pa.all_points.shape[0])

    #print ("plever: ", pa.p_lever.shape)
    start = 0
    end = 30

    #
    # # plot lever dynamics in PCA/neural space
    # for k in range(start,end,1):
    #     if plot_3D:
    #         ax.scatter(pa.p_lever[k,:,0],
    #                    pa.p_lever[k,:,1],
    #                    pa.p_lever[k,:,2],
    #                    color=cmap(k/(end-start)),
    #                    s=20,
    #                    edgecolor = 'black', alpha=.8)
    #     else:
    #         ax.scatter(pa.p_lever[k,:,0],
    #                    pa.p_lever[k,:,1],
    #                 color=cmap(k/(end-start)),
    #                 s=20,
    #                 edgecolor = 'black', alpha=.8)

    # plot t=0 in neural space for lever and all_points
    if plot_3D:
        ax.scatter(pa.p_lever[0,:,0],
                   pa.p_lever[0,:,1],
                   pa.p_lever[0,:,2],
                    color='red',
                    s=100,
                    edgecolor = 'black', alpha=.8)

        # plot all neural locationd
        ax.scatter(pa.all_points[idx,0],
                   pa.all_points[idx,1],
                   pa.all_points[idx,2],
                    c='black',
                    s=20,
                    edgecolor = 'black', alpha=.2)
    else:
        ax.scatter(pa.p_lever[0,:,0],
                   pa.p_lever[0,:,1],
                    color='red',
                    s=100,
                    edgecolor = 'black', alpha=.8)

        ax.scatter(pa.all_points[idx,0],
                   pa.all_points[idx,1],
                    c='black',
                    s=100,
                    edgecolor = 'black', alpha=.8)

    if False:
        plt.savefig('/home/cat/pca_all_plus_levers.png',dpi=300)
        plt.close()
    else:
        plt.show()

def knn_triage(th, pca_wf):

    tree = cKDTree(pca_wf)
    dist, ind = tree.query(pca_wf, k=6)
    dist = np.sum(dist, 1)

    idx_keep1 = dist <= np.percentile(dist, th)
    return idx_keep1

def get_convex_hull(pa):

    ''' COMPUTE CONVEX HULL
        - 
    
    '''
    
    # do very basic KNN triage
    n_dim = 3

    if True:

        from sklearn.neighbors import NearestNeighbors
        triage_value = 0.001
        knn_triage_threshold = 100*(1-triage_value)

        #if pca_wf.shape[0] > 1/triage_value:
        temp_points = pa.all_points[:,:n_dim]
        print ("temp points: ", temp_points.shape)
        idx_keep = knn_triage(knn_triage_threshold, temp_points)
        idx_keep = np.where(idx_keep==1)[0]

        print ("# points kept: ", idx_keep.shape,
               " of total: ", pa.all_points.shape[0])

        temp_points = temp_points[idx_keep]

    ######################################
    print ("computing convex hull # dim: ", n_dim, temp_points.shape)
    hull_all = ConvexHull(temp_points)

    #
    ratio_cumsum = []
    ratio_single = []
    ratio_random_single = []
    ratio_random_cumulative = []
    print ("p lever: ", pa.p_lever.shape)
    for k in trange(0, pa.p_lever.shape[0],1):

        # single frame convex hull; (frame???, n_times_points, n_dimensions)
        points = pa.p_lever[k:k+1,:,:n_dim].squeeze()#.reshape(-1,3)
        try:
            hull11 = ConvexHull(points)
        except:
            continue
            
        # 
        ratio_single.append(hull11.volume/hull_all.volume)

        # cumulative convex hull
        points = pa.p_lever[:k+1,:,:n_dim].reshape(-1,n_dim)
        #if k<10:
        #    print ("cumulative: ", points.shape)
        hull1 = ConvexHull(points)
        ratio_cumsum.append(hull1.volume/hull_all.volume)

        #
        #idx = np.random.randint(0,temp_points.shape[0],p_lever.shape[1])
        id_ = np.random.randint(0,temp_points.shape[0]-500,1)
        idx = np.arange(id_,id_+pa.p_lever.shape[1])
        points_random = temp_points[idx,:n_dim]
        hull3 = ConvexHull(points_random)
        ratio_random_single.append(hull3.volume/hull_all.volume)

        #
        ratio_random_cumulative.append([])
        for q in range(10):
            if True:
                idx = np.random.randint(0,temp_points.shape[0],
                                        pa.p_lever.shape[1]*(k+1))
            else:
                ns = np.random.randint(0,temp_points.shape[0]-500,k+1)
                idx = []
                for s in range(ns.shape[0]):
                    idx.append(np.arange(ns[s],ns[s]+pa.p_lever.shape[1]))
                idx=np.hstack(idx)
            points_random = temp_points[idx]
            #if q==0 and k<10:
            #    print (points_random.shape)
            hull4 = ConvexHull(points_random)
            ratio_random_cumulative[k].append(hull4.volume/hull_all.volume)

    pa.ratio_single = np.array(ratio_single)[::-1]
    pa.ratio_cumsum = np.array(ratio_cumsum)[::-1]
    pa.ratio_random_single = np.array(ratio_random_single)[::-1]
    pa.ratio_random_cumulative = np.array(ratio_random_cumulative)[::-1]

    print (pa.ratio_random_cumulative.shape)
    print (pa.ratio_random_single.shape)

    fname_out = os.path.join(pa.root_dir,
                             pa.animal_id,
                             'tif_files',
                             pa.session,
                             pa.session+"_convex_hull.npz")

    np.savez(fname_out,
             ratio_single = ratio_single,
             ratio_cumsum = ratio_cumsum,
             ratio_random_single = ratio_random_single,
             ratio_random_cumulative = ratio_random_cumulative
             )

    return pa



def plot_convex_hull22(pa):

    t = np.arange(pa.ratio_single.shape[0])[::-1]/30.-10

    ########### CUMSUM ############

    plt.plot(t, pa.ratio_cumsum,
                #s=100,
                #edgecolor='black',
                c='red',
                linewidth=3,
                label = 'Cumulative ConvexHull',
                alpha=.8)

    plt.fill_between(t, pa.ratio_cumsum,
                     pa.ratio_cumsum*0,
                     color='red',
                     alpha=.05)


    # vs random cumulative
    mean = np.mean(pa.ratio_random_cumulative,1)
    std = np.std(pa.ratio_random_cumulative,1)

    plt.plot(t,mean,
             linewidth=4,
             label = 'Cumulative ConvexHull - Random',
             c='black')

    plt.fill_between(t, mean+std, mean-std,
                     color='black',
                     alpha=.2)


    plt.ylim(0,1)
    plt.xlim(t[-1],t[0])
    plt.xlabel("Time (sec)")
    plt.ylabel("Area under curve")
    plt.legend()


def plot_convex_hull(pa):

    t = np.arange(pa.ratio_single.shape[0])/30.-10

    ########### CUMSUM ############
    plt.scatter(t, pa.ratio_cumsum,
               s=100,edgecolor='black',
               c='red', label = 'Cumulative ConvexHull',
               alpha=.8)

    # vs random cumulative
    mean = np.mean(pa.ratio_random_cumulative,1)
    std = np.std(pa.ratio_random_cumulative,1)

    plt.fill_between(t, mean+std, mean-std,
                     color='red',
                     alpha=.2)

    plt.plot(t,mean,'--',
             linewidth=4, label = 'Cumulative ConvexHull - Random',
            c='red')

    ############# SINGLE############
    plt.scatter(t, pa.ratio_single,
               s=100,edgecolor='black', label= "ConvexHull",
               c='blue',
               alpha=.8)

    # vs random single
    mean = np.mean(pa.ratio_random_single)
    std = np.std(pa.ratio_random_single)
    plt.fill_between(t, mean+std, mean-std,
                     color='blue',
                    alpha=.2)

    plt.plot([t[0],t[-1]],[mean,mean],'--',
             label= "ConvexHull - Random",
              linewidth=4,

            c='blue')

    plt.legend(*(
        [ x[i] for i in [3,1,2,0] ]
        for x in plt.gca().get_legend_handles_labels()
    ), handletextpad=0.75, loc='best',
              fontsize=20)

    plt.ylim(0,1)
    plt.xlim(t[0],t[-1])
    plt.show()



def plot_convex_hull_cumulative_only(pa, cmap, clr_ctr,
                                     alpha):


    t = np.arange(pa.ratio_single.shape[0])/30.-10

    ########### CUMSUM ############

    #plt.scatter(t, pa.ratio_cumsum,
    ax1=plt.subplot(121)
    plt.scatter(t, pa.ratio_cumsum,
               s=100,edgecolor='black',
               color=cmap(clr_ctr),
               label = 'Cumulative ConvexHull',
               alpha=alpha)
    plt.ylim(0,1)
    plt.xlim(t[0],t[-1])
    plt.xticks([])
    plt.yticks([])

    ax2=plt.subplot(122)
    plt.scatter(t, pa.ratio_cumsum/np.max(pa.ratio_cumsum),
               s=100,edgecolor='black',
               color=cmap(clr_ctr),
               label = 'Cumulative ConvexHull',
               alpha=alpha)

    plt.ylim(0,1)
    plt.xlim(t[0],t[-1])
    plt.xticks([])
    plt.yticks([])

#
def plot_convex_hull_single_only(pa, cmap, clr_ctr, alpha):


    t = np.arange(pa.ratio_single.shape[0])/30.-10

    ########### CUMSUM ############
    mean = np.mean(pa.ratio_random_single)

    #plt.scatter(t, pa.ratio_cumsum,
    ax1=plt.subplot(121)
    plt.scatter(t, pa.ratio_single,
               s=100, edgecolor='black', label= "ConvexHull",
               color=cmap(clr_ctr),
               alpha=alpha)
    plt.ylim(0,1)
    plt.xlim(t[0],t[-1])
    plt.show()

    ax2=plt.subplot(122)
    plt.scatter(t, pa.ratio_single/np.max(pa.ratio_single),
               s=100,edgecolor='black', label= "ConvexHull",
               color=cmap(clr_ctr),
               alpha=alpha)

    plt.ylim(0,1)
    plt.xlim(t[0],t[-1])
    plt.show()

# load triggers for each label
def load_starts_body_movements(pa
                              ):

    fname = os.path.join(pa.root_dir,
                        pa.animal_id,
                         'tif_files',
                         pa.session_corrected,
                         pa.session_corrected+"_3secNoMove_movements.npz"
                        )

    try:
        data = np.load(fname, allow_pickle=True)
    except:
        print (" No video/3sec movement files...")
        return []

    features = data['feature_quiescent']
    #print (features.shape)
    #labels = data['labels']
    #print (labels)
    #temp = np.vstack(features[0])

    starts = []
    for k in range(len(features)):
        temp = features[k]
        if len(temp)>0:
            #print (temp)
            starts.append(np.vstack(temp)[:,1])
        else:
            starts.append([])

    return starts


# Load the manual aligned shifts for each animal

def load_shift_ids(pa):

    fnames_good = '/media/cat/4TBSSD/yuki/'+pa.animal_id + '/tif_files/sessions_DLC_alignment_good.txt'

    import csv
    sessions = []
    shift_ids = []
    with open(fnames_good, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            sessions.append(str(row[0]).replace(',',''))
            shift_ids.append(row[1])

    #print ("SESSIONS: ", sessions)
    #print ("SHIFT IDS: ", shift_ids)
    #print ("pa.session_id: ", pa.session_id)

    # grab the session id from the ordered data to figure out correct ID to shift
    found = False
    for k in range(len(sessions)):
        # print ("sessions[k]", sessions[k])
        # print ("pa.sessoin_idd: ", pa.session_id)
        if sessions[k] in pa.session_id:
            ctr_plt = k
            found = True
            break
        # if pa.session_id in sessions[k]:
        #     ctr_plt = k
        #     found = True
        #     break

    if found:
        shift_id_str = shift_ids[ctr_plt]
    else:
        shift_id_str = None

    return shift_id_str


def load_shifts(pa):
    # load the DLC correlation shift
    fname_correlate = os.path.join(pa.root_dir, pa.animal_id,
                         'tif_files', pa.session,
                         'correlate.npz')

    try:
        data = np.load(fname_correlate,allow_pickle=True)
    except:
        print( " ... data missing", fname_correlate)
        asdf

    cors = data['cors'].squeeze().T

    #
    #print ("SELF SHIFT ID: ", pa.shift_id_str)
    if len(pa.shift_id_str)>1:
        pa.shift_id = int(pa.shift_id_str[0])
        pa.shift_additional = float(pa.shift_id_str[1:])
    else:
        pa.shift_id = int(pa.shift_id_str)
        pa.shift_additional = 0

    #print ( " using shift: ", pa.shift_id+pa.shift_additional)

    corr_featur_id = pa.shift_id

    temp_trace = cors[:,corr_featur_id]
    temp_trace[:2000] = 0
    temp_trace[-2000:] = 0
    DLC_shift = round(np.argmax(temp_trace)/1000. - 15.,2)+pa.shift_additional
    #print ("DLC SHIFT Loaded: ", DLC_shift)

    return DLC_shift


#
def pca_scatter_body_movements_fig3(pa, sessions):


    res_simplex = []
    for session in sessions:

        pa.session_id = session
        pa.session_corrected = session
        pa.session = session
        #print ("pa.session_id 1: ", pa.session_id )
        #print("session corrected: ", pa.session_corrected)

        #
        fname_simplex = os.path.join(pa.root_dir,pa.animal_id,'tif_files',pa.session,
                                 pa.session+ "_simplex.npy")

        if os.path.exists(fname_simplex)==False:
            ################################
            # load body movement starts
            starts = load_starts_body_movements(pa)
            if len(starts)==0:
                res_simplex.append([])
                continue

            #print ("pa.session_id 2: ", pa.session_id )
            #print ("starts: ", starts[0][:10])

            ################################
            # get the lever offset
            lever_offset = get_lever_offset(pa.root_dir,
                                            pa.animal_id,
                                            pa.session)
            #print ("lever offset: ", lever_offset)

            ###################################
            # load shift ids
            #pa.session_id = pa.session
            pa.shift_id_str = load_shift_ids(pa)
            if pa.shift_id_str is None:
                res_simplex.append([])
                continue

            ##################################
            # load shift
            DLC_shift = load_shifts(pa)
            shift_frames = int(DLC_shift*pa.frame_rate)
            shift_relative = -(shift_frames-lever_offset)
            #print ("DLC shift : ", DLC_shift,
            #       "  in frames: ", shift_frames,
            #       " after subtracting lever shift: ", shift_relative)

            ####################################
            # visualize
            clrs=['black','magenta','blue','green','red','yellow']

            # first load the raw data and lever triggers
            pa.X, pa.triggers = get_data_and_triggers(pa)
            if pa.X is None:
                res_simplex.append([])
                continue

            # get PCA or UMAP object and all points;
            pa.pca, pa.all_points = get_pca_object_and_all_points(pa)
            #print (" pca allpoints: ", pa.all_points.shape)

            ## use dim reduced data as input to UMAP
            if pa.umap_flag:
                pa.umap, pa.all_points = get_umap_object_and_all_points(pa)
                #print (" umap allpoints: ", pa.all_points.shape)

            #
            # loop over all features
            pa_array = []
            pa_simplex = []

            if pa.plot_flag:
                fig = plt.figure(figsize=(5,5))

            for k in range(5):
                pa.k=k

                #############################
                ######### PLOT LIMBS ########
                #############################
                pa.triggers = np.int32(np.array(starts[k])*pa.frame_rate) + shift_relative
                #print ("pa triggers: ", pa.triggers)

                if pa.triggers.shape[0]==0:
                    pa_array.append([])
                    pa_simplex.append([])
                    continue

                # FIG 2 E top
                # get pca projection first
                pa.p_lever =  project_data_pca(pa)
                #print ("  body movement pa.p_lever projection: ",
                #       pa.p_lever.shape)

                # get UMAP projection second
                if pa.umap_flag:
                    pa.p_lever = project_data_umap(pa)

                pa.plot_all = True

                #
                pa = plot_pca_scatter_multi(pa,
                                 n_frames=pa.n_frames,
                                 clr = clrs[k],
                                 plot_all=pa.plot_all,
                                 plot_3D = pa.plot_3D,
                                 plot_flag = pa.plot_flag)

                pa_array.append(pa.p_lever)
                pa_simplex.append(pa.points_simplex)

            #############################
            ######### PLOT LEVER ########
            #############################
            fname_lever = os.path.join(pa.root_dir, pa.animal_id,'tif_files',pa.session, pa.session+"_all_locs_selected.txt")
            #print ("FNAME LEVER: ", fname_lever)
            lever_times =np.loadtxt(fname_lever)
            pa.triggers = np.int32(np.array(lever_times)*pa.frame_rate)
            # print ("pa triggers: ", pa.triggers)

            #
            pa.p_lever =  project_data_pca(pa)

            # get UMAP projection second
            if pa.umap_flag:
                pa.p_lever =  project_data_umap(pa)

            pa.plot_all = True
            pa = plot_pca_scatter_multi(pa,
                             n_frames=pa.n_frames,
                             clr = 'brown',
                             plot_all=pa.plot_all,
                             plot_3D = pa.plot_3D,
                             plot_flag = pa.plot_flag)

            pa_simplex.append(pa.points_simplex)

            ########################################
            ######### ADD ALL POINTS TO END ########
            ########################################

            pa_simplex.append(pa.points_simplex_all_points)

            #
            if pa.plot_flag:
                if pa.umap_flag==False:
                    plt.ylim(-40000,60000)
                    plt.xlim(-60000,60000)
                plt.xticks([])
                plt.yticks([])

                plt.title(pa.session_corrected)

                if True:
                    plt.savefig('/home/cat/umap.png',dpi=300)
                    plt.close()
                else:
                    plt.show()

            res_simplex.append(pa_simplex)

            np.save(fname_simplex, pa_simplex)

        else:
            pa_simplex = np.load(fname_simplex, allow_pickle=True)
            res_simplex.append(pa_simplex)

    return res_simplex



def convexhull(p):
    p = np.array(p)
    hull = ConvexHull(p)
    return p[hull.vertices,:]



def plot_intersection_convex_hulls_lever_vs_bodyparts(res_simplex,
                                                      sessions,
                                                      animal_id,
                                                      root_dir):
    from shapely import geometry

    names = ['leftpaw','rightpaw','nose','jaw','ear','lever']
    clrs=['black','magenta','blue','green','red','brown']


    ctr = 0
    lever_vs_left_paw = []
    lever_vs_right_paw = []
    lever_vs_all = []
    n_trials = []
    for k in range(len(res_simplex)):

        # find # of lever pulls in sessions
        fname_session = os.path.join(root_dir, animal_id,'tif_files',sessions[k],
                                     sessions[k]+"_all_locs_selected.txt")


        #
        pa_simplex = res_simplex[k]
        if len(pa_simplex)==0:
            continue

        #
        if os.path.exists(fname_session):
            lever_pull_times = np.loadtxt(fname_session)
            n_pulls = lever_pull_times.shape[0]
        else:
            n_pulls = np.nan


        n_trials.append(n_pulls)

        if ctr<=45:
            ax=plt.subplot(5,10,ctr+1)

        ############################################
        ### PLOT POLYGONS FOR EACH BODY MOVEMENT ###
        ############################################
        polygons = []
        for p in range(len(pa_simplex)-1):
            pol_temp = []
            if len(pa_simplex[p])>0:
                for aa in range(pa_simplex[p].shape[0]):
                    temp = pa_simplex[p][aa].T

                    if ctr<=45:
                        # only use label once
                        if aa==0:
                            plt.plot(temp[0],
                                 temp[1],c=clrs[p],
                                 label=names[p],
                                 linewidth=5)
                        else:
                            plt.plot(temp[0],
                                 temp[1],
                                 c=clrs[p],
                                 linewidth=5)

                    pol_temp.append(temp.T[0])
                    pol_temp.append(temp.T[1])
                pol_temp = np.vstack(pol_temp)
                pol_temp = np.unique(pol_temp,axis=0)
                pol_temp = convexhull(pol_temp)
                polygons.append(pol_temp)
            else:
                polygons.append([])

        ########################################
        ### Compute all_points outer surface ###
        ########################################
        pts_simplex_all_points = pa_simplex[-1]
        pol_temp =[]
        for a in range(pts_simplex_all_points.shape[0]):
            temp = pts_simplex_all_points[a].T
            if a==0:
                plt.plot(temp[0],
                     temp[1],
                     c='grey',
                     label = 'all',
                    linewidth=5)
            else:
                plt.plot(temp[0],
                     temp[1],
                     c='grey',
                    linewidth=5)

            pol_temp.append(temp.T[0])
            pol_temp.append(temp.T[1])

        pol_temp = np.vstack(pol_temp)
        pol_temp = np.unique(pol_temp,axis=0)
        pol_temp = convexhull(pol_temp)
        polygons_all = pol_temp
        #

        ########################################
        # Compute areas of all neural spaces ###
        ########################################
        areas = []
        for p in range(len(polygons)):
            if len(polygons[p])>0:
                polygon1 = geometry.Polygon(polygons[p])
                polygon2 = geometry.Polygon(polygons[p])
                areas.append(polygon1.intersection(polygon2).area)
            else:
                areas.append(np.nan)

        print ("all areas: ", areas)
        polygon1 = geometry.Polygon(polygons_all)
        polygon2 = geometry.Polygon(polygons_all)
        area_all = polygon1.intersection(polygon2).area

        ############################################
        ##### COMPUTE Intersection lever and all ###
        ############################################
        polygon1 = geometry.Polygon(polygons[5])
        polygon2 = geometry.Polygon(polygons_all)

        intersection = polygon1.intersection(polygon2).area
        lever_vs_all.append(intersection/area_all)

        ####################################
        ###### compute intersections #######
        ####################################
        from shapely import geometry, ops
        for q in range(len(polygons)):
            for p in range(len(polygons)):
                if q==p:
                    continue

                polygon1 = geometry.Polygon(polygons[q])
                polygon2 = geometry.Polygon(polygons[p])

                intersection = polygon1.intersection(polygon2).area

                #
                if q == 0 and p == 5:
                    if np.isnan(areas[p])==False:
                        ratio = intersection/areas[q]
                        #ratio = ratio/n_pulls
                        lever_vs_left_paw.append(ratio)

                        print (names[q],names[p],
                           "% of region: ", round(ratio*100,2), "%",
                           "% of all space: ", round(intersection/area_all*100,2), "%")
                    else:
                        lever_vs_left_paw.append(np.nan)

                elif q == 1 and p == 5:
                    if np.isnan(areas[p])==False:
                        ratio = intersection/areas[q]
                        #ratio = ratio/n_pulls

                        lever_vs_right_paw.append(ratio)
                        print (names[q],names[p],
                           "% of region: ", round(ratio*100,2), "%",
                           "% of all space: ", round(intersection/area_all*100,2), "%")
                    else:
                        lever_vs_right_paw.append(np.nan)

            print ('')
        plt.xticks([])
        plt.yticks([])

        plt.title(sessions[k])

        #
        ctr+=1

    return lever_vs_left_paw, lever_vs_right_paw, lever_vs_all, n_trials


def plot_edt_distributions_box_plots(vis):

    #vis = Visualize.Visualize()
    vis.regular_flag = True
    vis.ideal_window_flag = False

    # 
    vis.animal_ids = ['IA1','IA2','IA3','IJ1','IJ2','AQ2']
    animal_names_elife = ["M1","M2","M3","M4","M5","M6"]
    
    fig = plt.figure(figsize=(20,4))
    legend_size = 22
    
    cons = []
    for ctr, animal_id in enumerate(vis.animal_ids):
        ax=plt.subplot(1,6,ctr+1)
        fnames = []

        # 
        fnames.append(os.path.join(vis.root_dir,
									animal_id,
									'first_decoding_time.npz'))
        fnames.append(os.path.join(vis.root_dir,
									animal_id,
									'first_decoding_time_concatenated.npz'))
            
        # data preprocessed slightly differently for single sessions vs. concatenated sessions
        shifts=[+15,0]

        # 
        linestyles = ['-','-','-']
        titles = ['single_sessions','concatenated']
        pvals = [0.05,0.01,0.001,0.0001,0.00001]
        edts = []
        yvals = []
        for ctr_type, fname in enumerate(fnames):
            data = np.load(fname, allow_pickle=True)

            # 
            clr = vis.colors[ctr_type]
            res = vis.plot_first_decoding_time_curves(data,
                                                      titles[ctr_type],
                                                      shifts[ctr_type],
                                                      linestyles[ctr_type],
                                                      clr, 
                                                      plotting=False)
            
            # 
            temp = res[0]

            #    
            edts.append(temp)
            print ("Mean edt: ", np.mean(temp))
            yvals.append(res[1])

        
        my_dict = dict(single_sessions = edts[0], 
                       concatenated = edts[1]
                      )
               
        # 
        #cons.append(edts[1])
        data = pd.DataFrame.from_dict(my_dict, orient='index')
        data = data.transpose()

        ###############################################
        ########## PLOT SCATTER #######################
        ###############################################
        clrs_local = ["blue","red"] #,"red"]
        for i,d in enumerate(data):
            y = data[d]
            x = np.random.normal(i+1, 0.04, len(y))
            if True:
                plt.plot(x, y, 
                     mfc =clrs_local[i], 
                     mec='k', 
                     ms=7, 
                     marker="o", 
                     linestyle="None",
                     alpha=.8
                    )
            else:
                plt.scatter(x, y, 
                       c=clrs_local[i],
                       edgecolor='black',
                       s=200,
                       alpha=.7)

            
        ###############################################
        ########## PLOT  #######################
        ###############################################
        flierprops = dict(marker='o', 
                          #markerfacecolor='g', 
                          markersize=10000,
                          linestyle='none', 
                          markeredgecolor='r')
        
        #
        bplot = data.boxplot(showfliers=True,
                    flierprops=flierprops,
                    grid=False) 
        
        if ctr==0:
            bplot.set_ylabel('EDTs (sec)')
        
        plt.title(animal_names_elife[ctr])
        temp = np.hstack(yvals)
        ymax = np.max(temp)

        #
        plt.ylim(-15,0)    

        #
        print ('')
    plt.show()
    
    

def plot_edt_distributions_box_plots_rois(vis):

    codes = ['_Retrosplenial', '_barrel', '_limb', '_visual','_motor','']
    #codes = ['limb, layer 1 - right', 'limb, layer 1 - left']
    clrs_local = ['magenta','brown','pink','lightblue','darkblue', 'blue']

   
    fig = plt.figure(figsize=(20,4))
    legend_size = 12
    for ctr, animal_id in enumerate(vis.animal_ids):
        ax=plt.subplot(1,6,ctr+1)
        fnames = []
        
        for code in codes:
            fnames.append('/media/cat/4TBSSD/yuki/'+animal_id+'/first_decoding_time'+code+'.npz')

        # 
        shifts=[+15,+15,+15,+15,+15,+15,+15]
        linestyles = ['-','-','-','-','-','-','-','-']
        
        # 
        pvals = [0.05,0.01,0.001,0.0001,0.00001]
        edts = []
        yvals = []

        # 
        for ctr_type, fname in enumerate(fnames):

            # 
            fname = fname.replace("time_","time__")
            res = np.load(fname, allow_pickle=True)['all_res_continuous']
            
            #    
            edts.append(res)
            yvals.append(res[1])

        edts_saved = np.array(edts, dtype=object).copy()
        
        if len(codes)!=2:
            my_dict = dict(ret = edts[0], 
                          barrel = edts[1],
                          limb = edts[2],
                          visual = edts[3],
                          motor = edts[4],                      
                          All = edts[5]+15,                  
                         )
        else:
            my_dict = dict(right = edts[0], 
                           left = edts[1])

        
        data = pd.DataFrame.from_dict(my_dict, orient='index')
        data = data.transpose()

        ######################################################### 
        ################## SCATTER PLOTS ########################
        ######################################################### 
        for i,d in enumerate(data):
            y = data[d]
            x = np.random.normal(i+1, 0.04, len(y))
            plt.plot(x, y, 
                     mfc =clrs_local[i], 
                     mec='k', 
                     ms=7, 
                     marker="o", 
                     linestyle="None",
                     alpha=.8
                    )

		#
        if animal_id=='IA1':
            patches = []
            for c,code in enumerate(codes):
                patches.append(mpatches.Patch(color=clrs_local[c], label=code.replace('_','')))
        
			#
            if vis.show_legend:
                plt.legend(handles=patches, fontsize=legend_size)

        plt.title(vis.manuscript_names[ctr])
            
        ######################################################### 
        ###################### BOX PLOTS ########################
        ######################################################### 
        flierprops = dict(marker='o', 
                          #markerfacecolor='g', 
                          #markersize=10000,
                          linestyle='none', 
                          markeredgecolor='r')
        
        #
        data.boxplot(showfliers=False,
                    flierprops=flierprops,
                    grid=False
                    )

        temp = np.hstack(yvals)
        ymax = np.max(temp)

        #
        plt.xlim(0.5, 6.5)
        plt.ylim(-15,0)    


        #
        plt.plot([0,6.5], [-3,-3],'--',linewidth=3,c='grey',alpha=.5)
        plt.plot([0,6.5],[-10,-10],'--',linewidth=3,c='grey',alpha=.5)

        patches = []
        if False:
            if len(codes)!=2:
                for p in [0,2,4]:
                    res = stats.ks_2samp(edts_saved[p], edts_saved[3])
                    #print ("res: ", res)
                    label_ = ''
                    for k in range(len(pvals)):
                        if res[1]<pvals[k]:
                            label_ = label_ + "*"
                        else:
                            break

                    patches.append(mpatches.Patch(label=label_))

                res = stats.ks_2samp(edts_saved[2], edts_saved[5])

                #
                label_ = ''
                for k in range(len(pvals)):
                    if res[1]<pvals[k]:
                        label_ = label_ + "*"
                    else:
                        break

                patches.append(mpatches.Patch(color='blue', label=label_))


            else:
                res = stats.ks_2samp(edts_saved[0], edts_saved[1])
                #print ("ks test: ", res)
                label_ = ''
                for k in range(len(pvals)):
                    if res[1]<pvals[k]:
                        label_ = label_ + "*"
                    else:
                        break

                patches.append(mpatches.Patch(color='blue', label=label_))


        ##########################################
        #print ("#####################")
        #print ("edts: ", edts_saved[0], edts_saved[2])
        res = stats.ks_2samp(edts_saved[0], edts_saved[5])
        print (animal_id, res)
        # label_ = ''
        # for k in range(len(pvals)):
        #     if res[1]<pvals[k]:
        #         label_ = label_ + "*"
        #     else:
        #         break

        patches.append(mpatches.Patch(color='red', label="limb vs. all" + 
                                      "pval: "+str(np.round(res,5))))

        #print ('')
        #ax2 = ax.twinx()  
        if vis.show_legend:
              ax.legend(handles=patches,fontsize=legend_size)

       
def plot_limb_movement_vs_time(root_dir,
                               animal_id, 
                               clrs,
                               codes):
    
    #
    data = np.load(os.path.join(root_dir,animal_id,
                                animal_id+'_edt_body_parts.npz'))
    edt = data['edt']
    n_trials = data['n_trials']

    for k in range(len(clrs)):
        ax=plt.subplot(1,2,k+1)

        idx = np.where(np.logical_and(edt[:,k]!=0,np.isnan(edt[:,k])==False))[0]
        t = np.array(n_trials[idx,k])
        temp = np.array(edt[idx,k])

        # 
        plt.scatter(temp,t,c=clrs[k],
                   s=500,edgecolor='black',alpha=1-np.arange(temp.shape[0])/temp.shape[0])

        # 
        coef = np.polyfit(temp,t,1)
        poly1d_fn = np.poly1d(coef) 
        plt.plot(temp, poly1d_fn(temp), c='black',
                linewidth=5,alpha=.5)    

        # 
        plt.xlim(-15,0)
        plt.ylim(0,200)
        plt.title(codes[k])
        plt.xlabel("EDT (sec)")
        plt.ylabel("# of trials in session")
    


def plot_edt_distributions_box_plots_body_vs_lever(vis):

    codes = [
             'left_paw',          # 0
             'right_paw',         # 1
             'nose',              # 2
             'jaw',               # 3
             'right_ear',         # 4
             'tongue'
            ]

    #codes = ['Retrosplenial', 'barrel', 'limb', 'visual','motor']
    #clrs_local = ['grey','magenta','blue','green','red','orange']
    clrs_local = ['grey','magenta','orange','blue']
    
    codes = [
    'left_paw',          # 0
    'right_paw',         # 1
    'nose',              # 2
    'jaw',               # 3
    'right_ear',         # 4
    'tongue']

    codes = [
    'left_paw',          # 0
    'right_paw',         # 1
    'nose',              # 2
    'jaw',               # 3
    'right_ear',         # 4
    'tongue']

    
    fig = plt.figure(figsize=(20,4))
    legend_size = 10
    for ctr, animal_id in enumerate(vis.animal_ids):
        ax=plt.subplot(1,6,ctr+1)
        fnames = []
        
        fname_edt_lever = '/media/cat/4TBSSD/yuki/'+animal_id+'/edts_single_session.npy'

        edts_lever_pulls = np.load(fname_edt_lever)
        
        #for code in codes:
        fname = '/media/cat/4TBSSD/yuki/'+animal_id+'_edt_body_parts.npz'

        # 
        #shifts=[+15,+15,+15,+15,+15,+15,+15]
        shifts=[0,0,0,0,0,0,0]
        linestyles = ['-','-','-','-','-','-','-','-']

        # 
        pvals = [0.05,0.01,0.001,0.0001,0.00001]
        edts = []
        yvals = []

        data = np.load(fname, allow_pickle=True)
        data = data['edt']

        # 
        for ctr_type in range(data.shape[1]):
            # 

            temp = data[:,ctr_type]

            #    
            edts.append(temp)

        edts_saved = np.array(edts).copy()
        my_dict = dict(left_paw = edts[0], 
                       right_paw = edts[1],
                       #nose = edts[2],
                       #ear = edts[3],
                       #jaw = edts[4],                      
                       tongue = edts[5],   
                       all = edts_lever_pulls
                      )
        
        data = pd.DataFrame.from_dict(my_dict, orient='index')
        data = data.transpose()

        # 
        flierprops = dict(marker='o', 
                          #markerfacecolor='g', 
                          #markersize=10000,
                          linestyle='none', 
                          markeredgecolor='r')
        
        #
        data.boxplot(showfliers=False,
                    flierprops=flierprops,
                    grid=False)
        
        
        for i,d in enumerate(data):
            y = data[d]
            x = np.random.normal(i+1, 0.04, len(y))
            plt.plot(x, y, 
                     mfc =clrs_local[i], 
                     mec='k', 
                     ms=7, 
                     marker="o", 
                     linestyle="None",
                     
                    )

        #
        plt.xlim(0.5, 4.5)
        plt.ylim(-15,0)    
        #plt.xticks([])
        #plt.yticks([])

        #
        plt.plot([0,6.5], [-3,-3],'--',linewidth=3,c='grey',alpha=.5)
        plt.plot([0,6.5],[-10,-10],'--',linewidth=3,c='grey',alpha=.5)

        # ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        if True:
            patches = []
            for p in [0,1,5]:
                try:
                    res = stats.ks_2samp(edts_saved[p], edts_lever_pulls)
                except:
                    continue
                    
                label_ = ''
                #print (p, res)
                for k in range(len(pvals)):
                    if res[1]<pvals[k]:
                        label_ = label_ + "*"
                    else:
                        break

                patches.append(mpatches.Patch(color='blue', label=label_))


            #print ('')
            if vis.show_legend:
                ax2 = ax.twinx()  
				
                ax2.legend(handles=patches, fontsize=8)
                plt.yticks([])
            

    ############################################################
    ############################################################
    ############################################################
    if True:
        plt.show()
    else:
        plt.savefig('/home/cat/first_scatter_body_movements.svg')
        plt.close()
        
        


def load_locaNMF_temporal(animal_id,
                          session_name,
                          root_dir,
                          session_id):
    #import
    loca = LocaNMFClass(root_dir, animal_id, session_name)

    #
    loca.get_sessions(session_name)
    #print ("sessions: ", loca.sessions.shape)
    #print ("selected session: ", loca.sessions[session_id])

    session = loca.sessions[session_id]

    # load data
    fname_locaNMF = os.path.join(root_dir, animal_id, 'tif_files',session,
                                 session + '_locanmf.npz')


    atlas, areas, names, locaNMF_temporal, locaNMF_temporal_random = load_locaNMF_data(fname_locaNMF)

    return atlas, areas, names, locaNMF_temporal, locaNMF_temporal_random



def load_locaNMF_data(fname_locaNMF):
    # order locaNMF components by plot color ORDER in Fig 4A
    ordered_names = np.array([15,0,14,1,   # retrosplenial areas
                          13,2,
                          12,3,
                          11,4,
                          10,5,
                          9,6,
                          8,7])[::-1]


    # load raw data
    try:
        d = np.load(fname_locaNMF)
    except:
        print ("file missing", fname_locaNMF)
        return None, None, None, None, None

    locaNMF_temporal = d['temporal_trial']
    locaNMF_temporal_random = d['temporal_random']
    locaNMF_temporal = locaNMF_temporal[:,ordered_names]
    locaNMF_temporal_random = locaNMF_temporal_random[:,ordered_names]
    # print ("locanmf data: ", locaNMF_temporal.shape)

    #
    areas = d['areas'][ordered_names]
    names = d['names'][ordered_names]
    #print ("original names: ", names.shape)

    #
    atlas = np.load('/home/cat/code/widefield/locanmf/atlas_fixed_pixel.npy')
    #print ("atlas: ",atlas.shape)
    # print (areas)
    # print (names)

    #print ("  # of ordered_names: ", ordered_names.shape)
    #print ("ORDERED NAMES: ", names[ordered_names])


    return atlas, areas, names, locaNMF_temporal, locaNMF_temporal_random

def get_angle_from_sinusoid(curve, index):
    # COMPUTE PHASE ANGLE FROM AMPLITUDE AND GRADIENT OF SINUSOID
    # Index is the loatoin of the point at which the phase is to be computed

    # must normalize and recentre curve to go from -1 to +1 as a standardized sinusoid
    curve = (curve-np.min(curve))/(np.max(curve)-np.min(curve))*2-1

    angle = np.arcsin(curve[index])
    if angle<0:
        angle+=np.pi

    #
    curve0 = curve[index]
    curve1 = curve[index+1]
    if curve0<0:
        if (curve1-curve0)>0:
            angle = np.pi+angle
        else:
            angle = 2*np.pi-angle
    else:
        if (curve1-curve0)<=0:
            angle = np.pi-angle
        else:
            angle = angle

    phase = angle

    return phase

def fit_sin(tt, yy):

    '''Fit sin to the input time sequence,
        and return fitting parameters
        "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"


    '''

    #
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):
        return A * np.sin(w*t + p) + c

    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c

    #
    return {"amp": A,
            "omega": w,
            "phase": p,
            "offset": c,
            "freq": f,
            "period": 1./f,
            "fitfunc": fitfunc,
            "maxcov": np.max(pcov),
            "rawres": (guess,popt,pcov)
           }



def plot_phases(root_dir, animal_id, session_id, show_area_id, random_flag):

    ''' function that fits phases to raw locanm components

    '''

    session_name = 'all'

    #
    areas_to_plot_phases = ['Retrosplenial area, dorsal part, layer 1 - left',
                            'Primary somatosensory area, barrel field, layer 1 - left',
                            'upper limb, layer 1 - left',
                            'Primary visual area, layer 1 - left',
                            'Primary motor area, Layer 1 - left',
                           ]

    #
    #show_area_id = 2  # upper limb cortex

    #
    #codes = ['_Retrosplenial', '_barrel', '_limb', '_visual','_motor','']
    clrs_local = ['magenta','brown','pink','lightblue','darkblue', 'blue']

    #
    #session_id = 21  # March 3 - IJ2 session

    #
    #random_flag = False

    start = 900 - 15*30   # start at -10 sec
    end = 900 + 0*30             # end at 0 sec

    #
    plotting=True

    #
    (atlas,
     areas,
     names,
     locaNMF_temporal,
     locaNMF_temporal_random) = load_locaNMF_temporal(animal_id,
                                                      session_name,
                                                      root_dir,
                                                      session_id)
    #
    if atlas is None:
        print ("session is empty ")
        return ([[],[],[],[],[]])

    plt.figure(figsize=(15,10))

    # loop over areas
    t0_phases=[]
    for ctr_area, area_sel in enumerate(areas_to_plot_phases):
        #
        t0_phases.append([])

        #
        areas_selected = []
        for k in range(len(names)):
            if area_sel in names[k]:
                areas_selected.append(k)

        # use random data instead
        if random_flag:
            locaNMF_temporal = locaNMF_temporal_random

        #
        areas_selected = np.array(areas_selected)
        locaNMF_temporal2 = locaNMF_temporal[:,areas_selected].squeeze()

        #
        missed_fit = 0
        t = np.arange(locaNMF_temporal2.shape[1])/30-30
        curves=[]

       # print ("# trials: ", locaNMF_temporal2.shape[0])
        for k in range(locaNMF_temporal2.shape[0]):
            if ctr_area==show_area_id:
                if plotting:
                    ax1=plt.subplot(2,1,1)
                    ax1.plot(t,locaNMF_temporal2[k],
                         c='black',
                         linewidth=3,
                         alpha=.1)

            # fit sinusoid to the single trial data
            tt = t[start:end]
            yy = locaNMF_temporal2[k][start:end]

            #
            try:
                tt2 = np.arange(0, locaNMF_temporal2[k].shape[0],1)/30.-30
                res = fit_sin(tt, yy)
                curve = res["fitfunc"](tt2)

                #print ("CURVE: ", curve.shape)
                # if curve fit, extrapolate all the way to -30sec

                if plotting:
                    if ctr_area==show_area_id:
                        ax2=plt.subplot(2,1,2)
                        curve2 = (curve-np.min(curve))/(np.max(curve)-np.min(curve))*2-1
                        curves.append(curve2)
                        ax2.plot(tt2,
                                 curve2,
                                 linewidth=2,
                                 #c=clrs_local[show_area_id],
                                 c='pink',
                                 alpha=.5
                            )
                        ax2.scatter(tt2[900],
                                 curve2[900],
                                 edgecolor='black',
                                 c=clrs_local[show_area_id],
                                 alpha=1
                            )

                # get peaks of curves
                phase = get_angle_from_sinusoid(curve, 900)

                #
                t0_phases[ctr_area].append(phase)
            except:
                pass


        ###############################
        ######## PLOT MEANS ###########
        ###############################
        if ctr_area==show_area_id:
            #curves=np.array(curves)
            mean = np.mean(locaNMF_temporal2,axis=0)
            if plotting:
                #ax=plt.subplot(2,1,1)
                ax1.plot(t,mean,
                         c='black',
                         linewidth=8,
                         alpha=1)

                #
                ax1.plot([-30,30],[0,0],'--',c='grey')
                ax1.set_xlim(-15,1)
                ax1.plot([0,0],[-10,10],'--',c='grey')
                ax1.set_ylim(-0.15,0.20)
                ax1.set_xlabel("Time (sec)")
                ax1.set_ylabel("DF/F")


            # fit sinusoid to trial average
            try:
                tt = t[start:end]
                curves=np.array(curves)
                mean = np.mean(curves,axis=0)
                tt2 = np.arange(0,mean.shape[0],1)/30.-30
                yy = mean[start:end]

                res = fit_sin(tt, yy)
                #print( "Amplitude=%(amp)s, Angular freq.=%(omega)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s" % res )

                # PLOT REAL AVERAGE NOT FIT

                if plotting:
                    mean = np.mean(curves,axis=0)
                    #ax2=plt.subplot(2,1,2)
                    #plt.plot(tt, yy, "-k", label="y", linewidth=2)
                    ax2.plot(tt2, mean, linewidth=8,
                            #c=clrs_local[show_area_id]
                            c='pink'
                            )

                    #
                    ax2.plot([-30,30],[0,0],'--',c='grey')
                    ax2.plot([0,0],[-10,10],'--',c='grey')
                    ax2.set_xlim(-15,1)
                    #ax2.ylim(-0.15,0.15)
                    ax2.set_ylim(-1,1)
                    ax2.set_xlabel("Time (sec)")
                    ax2.set_ylabel("Normalized sin fit")
            except:
                pass
    return t0_phases

#
class LocaNMFClass():

    def __init__(self, root_dir, animal_id, session):

        #
        self.min_trials = 10

        #
        self.root_dir = root_dir

        #
        self.animal_id = animal_id   # 'IJ1'

        #
        self.sessions = self.get_sessions(session)     # 'Mar3'

        #
        #fname_atlas = os.path.join(self.root_dir, 'yongxu/atlas_split.npy')
        fname_atlas = '/home/cat/code/widefield/locanmf/atlas_fixed_pixel.npy'
        self.atlas = np.load(fname_atlas)



    def get_sessions(self,session_id):
         # load ordered sessions from file
        sessions = np.load(os.path.join(self.root_dir,
                                        self.animal_id,
                                        'tif_files.npy'))
        # grab session names from saved .npy files
        data = []
        for k in range(len(sessions)):
            data.append(os.path.split(sessions[k])[1].replace('.tif',''))
        sessions = data

        #
        if session_id != 'all':
            final_session = []
            session_number = None
            for k in range(len(sessions)):
                if session_id in sessions[k]:
                    final_session = [sessions[k]]
                    session_number = k
                    break
            sessions = final_session

        # fix binary string files issues; remove 'b and ' from file names
        for k in range(len(sessions)):
            sessions[k] = str(sessions[k]).replace("'b",'').replace("'","")
            if sessions[k][0]=='b':
                sessions[k] = sessions[k][1:]

        sessions = np.array(sessions)

        return sessions

    def run_loca(self):

        #################################################
        #################################################
        #################################################
        # maxrank = how many max components per brain region. Set maxrank to around 4 for regular dataset.
        maxrank = 1

        # min_pixels = minimum number of pixels in Allen map for it to be considered a brain region
        # default min_pixels = 100
        min_pixels = 200

        # loc_thresh = Localization threshold, i.e. percentage of area restricted to be inside the 'Allen boundary'
        # default loc_thresh = 80
        loc_thresh = 75

        # r2_thresh = Fraction of variance in the data to capture with LocaNMF
        # default r2_thresh = 0.99
        r2_thresh = 0.96

        # Do you want nonnegative temporal components? The data itself should also be nonnegative in this case.
        # default nonnegative_temporal = False
        nonnegative_temporal = False


        # maxiter_hals = Number of iterations in innermost loop (HALS). Keeping this low provides a sort of regularization.
        # default maxiter_hals = 20
        maxiter_hals = 20

        # maxiter_lambda = Number of iterations for the lambda loop. Keep this high for finding a good solution.
        # default maxiter_lambda = 100
        maxiter_lambda = 150

        # lambda_step = Amount to multiply lambda after every lambda iteration.
        # lambda_init = initial value of lambda. Keep this low. default lambda_init = 0.000001
        # lambda_{i+1}=lambda_i*lambda_step. lambda_0=lambda_init. default lambda_step = 1.35
        # lambda_step = 1.25
        # lambda_init = 1e-4

        # NEW PARAMS SUGGESTED BY YONGXU July ~20
        lambda_step = 2.25
        lambda_init = 1e-1

        ######################################################
        ######################################################
        ######################################################
        for session in self.sessions:

            fname_out = os.path.join(self.root_dir,self.animal_id,'tif_files',
                                     session,session+'_locanmf.npz')
            if os.path.exists(fname_out)==False:

                fname_locs = os.path.join(self.root_dir, self.animal_id, 'tif_files',
                                          session, session + '_all_locs_selected.txt')
                if os.path.exists(fname_locs)==False:
                    print ("  no lever pulls, skipping ")
                    continue

                n_locs = np.loadtxt(fname_locs)
                print ("")
                print ("")
                print (session, " has n trials: ", n_locs.shape)
                if n_locs.shape[0]<self.min_trials:
                    print ("  too few trials, skipping ", n_locs.shape[0])
                    continue

                ###########################################################
                # load spatial footprints from PCA compressed data
                fname_spatial = os.path.join(self.root_dir,self.animal_id, 'tif_files',
                                             session,
                                             #session+'_code_04_trial_ROItimeCourses_15sec_pca_0.95_spatial.npy')
                                             session+'_code_04_trial_ROItimeCourses_30sec_pca_0.95_spatial.npy')

                spatial = np.load(fname_spatial)
                spatial = np.transpose(spatial,[1,0])
                denoised_spatial_name = np.reshape(spatial,[128,128,-1])
                # print ("denoised_spatial_name: ", denoised_spatial_name.shape)
                #

                ###########################################################
                # load temporal PC components
                temporal_trial = np.load(fname_spatial.replace('_spatial',''))

                #
                temporal_random = np.load(fname_spatial.replace('trial','random').replace('_spatial',''))

                # make sure there are same # of trials in random and trial dataset
                min_trials = min(temporal_trial.shape[0], temporal_random.shape[0])
                temporal_trial = temporal_trial[:min_trials]
                temporal_random = temporal_random[:min_trials]

                #
                temporal=np.concatenate((temporal_trial,temporal_random),axis=0)
                temporal = np.transpose(temporal,[1,0,2])

                denoised_temporal_name = np.reshape(temporal,[-1,temporal.shape[1]*temporal.shape[2]])
                #print('loaded data',flush=True)

                #######################################
                # Get data in the correct format
                V=denoised_temporal_name
                U=denoised_spatial_name

                #
                brainmask = np.ones(U.shape[:2],dtype=bool)

                # Load true areas if simulated data
                simulation=0

                # Include nan values of U in brainmask, and put those values to 0 in U
                brainmask[np.isnan(np.sum(U,axis=2))]=False
                U[np.isnan(U)]=0

                # Preprocess V: flatten and remove nans
                dimsV=V.shape
                keepinds=np.nonzero(np.sum(np.isfinite(V),axis=0))[0]
                V=V[:,keepinds]

                #
                if V.shape[0]!=U.shape[-1]:
                    print('Wrong dimensions of U and V!')

                print("Rank of video : %d" % V.shape[0])
                print("Number of timepoints : %d" % V.shape[1]);


                ##################################################
                ##################################################
                ##################################################
                # Perform the LQ decomposition. Time everything.
                t0_global = time.time()
                t0 = time.time()
                if nonnegative_temporal:
                    r = V.T
                else:
                    q, r = np.linalg.qr(V.T)
                # time_ests={'qr_decomp':time.time() - t0}

                # Put in data structure for LocaNMF
                video_mats = (np.copy(U[brainmask]), r.T)
                rank_range = (1, maxrank, 1)
                del U


                ##################################################
                ##################################################
                ##################################################

                #
                region_mats = LocaNMF.extract_region_metadata(brainmask,
                                                              self.atlas,
                                                              min_size=min_pixels)

                #
                region_metadata = LocaNMF.RegionMetadata(region_mats[0].shape[0],
                                                         region_mats[0].shape[1:],
                                                         device=device)

                #
                region_metadata.set(torch.from_numpy(region_mats[0].astype(np.uint8)),
                                    torch.from_numpy(region_mats[1]),
                                    torch.from_numpy(region_mats[2].astype(np.int64)))


                ##################################################
                ##################################################
                ##################################################

                # grab region names
                rois=np.load('/home/cat/code/widefield/locanmf/rois_50.npz')
                rois_name=rois['names']

                rois_ids=rois['ids']

                ##################################################
                ##################################################
                ##################################################

                # Do SVD as initialization
                if device=='cuda':
                    torch.cuda.synchronize()

                #
                print('v SVD Initialization')
                t0 = time.time()
                region_videos = LocaNMF.factor_region_videos(video_mats,
                                                             region_mats[0],
                                                             rank_range[1],
                                                             device=device)
                #
                if device=='cuda':
                    torch.cuda.synchronize()
                print("\'-total : %f" % (time.time() - t0))
                #time_ests['svd_init'] = time.time() - t0


                #
                low_rank_video = LocaNMF.LowRankVideo(
                    (int(np.sum(brainmask)),) + video_mats[1].shape, device=device
                )
                low_rank_video.set(torch.from_numpy(video_mats[0].T),
                                   torch.from_numpy(video_mats[1]))



                ##################################################
                ##################################################
                ##################################################
                if device=='cuda':
                    torch.cuda.synchronize()

                #
                print('v Rank Line Search')
                t0 = time.time()
                try:
                    # locanmf_comps,loc_save = LocaNMF.rank_linesearch(low_rank_video,
                    #                                              region_metadata,
                    #                                              region_videos,
                    #                                              maxiter_rank=maxrank,
                    #                                              maxiter_lambda=maxiter_lambda,      # main param to tweak
                    #                                              maxiter_hals=maxiter_hals,
                    #                                              lambda_step=lambda_step,
                    #                                              lambda_init=lambda_init,
                    #                                              loc_thresh=loc_thresh,
                    #                                              r2_thresh=r2_thresh,
                    #                                              rank_range=rank_range,
                    # #                                             nnt=nonnegative_temporal,
                    #                                              verbose=[True, False, False],
                    #                                              sample_prop=(1,1),
                    #                                              device=device
                    #                                             )

                    t0 = time.time()
                    locanmf_comps,loc_save,save_lam,save_scale,save_per,save_spa,save_scratch = LocaNMF.rank_linesearch(low_rank_video,
                                                                region_metadata,
                                                                region_videos,
                                                                maxiter_rank=maxrank,
                                                                maxiter_lambda=maxiter_lambda,
                                                                maxiter_hals=maxiter_hals,
                                                                lambda_step=lambda_step,
                                                                lambda_init=lambda_init,
                                                                loc_thresh=loc_thresh,
                                                                r2_thresh=r2_thresh,
                                                                rank_range=rank_range,
                    #                                             nnt=nonnegative_temporal,
                                                                verbose=[True, False, False],
                                                                sample_prop=(1,1),
                                                                device=device
                                                             )
                    if device=='cuda':
                        torch.cuda.synchronize()

                except Exception as e:
                    print (" locaNMF Failed, skipping")
                    print (e)
                    print ('')
                    print ('')
                    continue
                #
                if device=='cuda':
                    torch.cuda.synchronize()


                # C is the temporal components
                C = np.matmul(q,locanmf_comps.temporal.data.cpu().numpy().T).T
                print ("n_comps, n_time pts x n_trials: ", C.shape)
                qc, rc = np.linalg.qr(C.T)


                # Assigning regions to components
                region_ranks = []; region_idx = []

                for rdx in torch.unique(locanmf_comps.regions.data, sorted=True):
                    region_ranks.append(torch.sum(rdx == locanmf_comps.regions.data).item())
                    region_idx.append(rdx.item())

                areas=region_metadata.labels.data[locanmf_comps.regions.data].cpu().numpy()

                # Get LocaNMF spatial and temporal components
                A=locanmf_comps.spatial.data.cpu().numpy().T
                A_reshape=np.zeros((brainmask.shape[0],brainmask.shape[1],A.shape[1]));
                A_reshape.fill(np.nan)
                A_reshape[brainmask,:]=A

                # C is already computed above delete above
                if nonnegative_temporal:
                    C=locanmf_comps.temporal.data.cpu().numpy()
                else:
                    C=np.matmul(q,locanmf_comps.temporal.data.cpu().numpy().T).T

                # Add back removed columns from C as nans
                C_reshape=np.full((C.shape[0],dimsV[1]),np.nan)
                C_reshape[:,keepinds]=C
                C_reshape=np.reshape(C_reshape,[C.shape[0],dimsV[1]])

                # Get lambdas
                lambdas=np.squeeze(locanmf_comps.lambdas.data.cpu().numpy())


                # c_p is the trial sturcutre
                c_p=C_reshape.reshape(A_reshape.shape[2],int(C_reshape.shape[1]/1801),1801)

                #
                c_plot=c_p.transpose((1,0,2))
                c_plot.shape


                ##################################################
                ##################################################
                ##################################################
                # save LocaNMF data
                areas_saved = []
                for area in areas:
                    idx = np.where(rois_ids==np.abs(area))[0]
                    temp_name = str(rois_name[idx].squeeze())
                    if area <0:
                        temp_name += " - right"
                    else:
                        temp_name += " - left"

                    areas_saved.append(temp_name)

                # GET AREA NAMES
                def parse_areanames_new(region_name,rois_name):
                    areainds=[]; areanames=[];
                    for i,area in enumerate(region_name):
                        areainds.append(area)
                        areanames.append(rois_name[np.where(rois_ids==np.abs(area))][0])
                    sortvec=np.argsort(np.abs(areainds))
                    areanames=[areanames[i] for i in sortvec]
                    areainds=[areainds[i] for i in sortvec]
                    return areainds,areanames

                #
                region_name=region_mats[2]

                # Get area names for all components
                areainds,areanames_all = parse_areanames_new(region_name,rois_name)
                areanames_area=[]
                for i,area in enumerate(areas):
                    areanames_area.append(areanames_all[areainds.index(area)])

                ###################################
                np.savez(fname_out,
                          temporal_trial = c_plot[:int(c_plot.shape[0]/2),:,:],
                          temporal_random = c_plot[int(c_plot.shape[0]/2):,:,:],
                          areas = areas,
                          names = areas_saved,
                          A_reshape = A_reshape,
                          areanames_area = areanames_area
                         )


        print (" ... DONE ALL SESSIONS...")


    def run_loca_whole_session(self):

        #################################################
        #################################################
        #################################################
        # maxrank = how many max components per brain region. Set maxrank to around 4 for regular dataset.
        maxrank = 1

        # min_pixels = minimum number of pixels in Allen map for it to be considered a brain region
        # default min_pixels = 100
        min_pixels = 200

        # loc_thresh = Localization threshold, i.e. percentage of area restricted to be inside the 'Allen boundary'
        # default loc_thresh = 80
        loc_thresh = 75

        # r2_thresh = Fraction of variance in the data to capture with LocaNMF
        # default r2_thresh = 0.99
        r2_thresh = 0.96

        # Do you want nonnegative temporal components? The data itself should also be nonnegative in this case.
        # default nonnegative_temporal = False
        nonnegative_temporal = False


        # maxiter_hals = Number of iterations in innermost loop (HALS). Keeping this low provides a sort of regularization.
        # default maxiter_hals = 20
        maxiter_hals = 20

        # maxiter_lambda = Number of iterations for the lambda loop. Keep this high for finding a good solution.
        # default maxiter_lambda = 100
        maxiter_lambda = 150

        # lambda_step = Amount to multiply lambda after every lambda iteration.
        # lambda_init = initial value of lambda. Keep this low. default lambda_init = 0.000001
        # lambda_{i+1}=lambda_i*lambda_step. lambda_0=lambda_init. default lambda_step = 1.35
        # lambda_step = 1.25
        # lambda_init = 1e-4

        # NEW PARAMS SUGGESTED BY YONGXU July ~20
        lambda_step = 2.25
        lambda_init = 1e-1

        ######################################################
        ######################################################
        ######################################################
        for session in self.sessions:

            # output filename
            fname_out = os.path.join(self.root_dir,self.animal_id,'tif_files',
                                     session,session+'_locanmf_wholestack.npz')

            #
            if os.path.exists(fname_out)==False:

                fname_locs = os.path.join(self.root_dir, self.animal_id, 'tif_files',
                                          session, session + '_all_locs_selected.txt')
                if os.path.exists(fname_locs)==False:
                    print ("  no lever pulls, skipping ")
                    continue

                n_locs = np.loadtxt(fname_locs)
                print ("")
                print ("")
                print (session, " has n trials: ", n_locs.shape)
                if n_locs.shape[0]<self.min_trials:
                    print ("  too few trials, skipping ", n_locs.shape[0])
                    continue

                ###########################################################
                # load spatial footprints from PCA compressed data
                fname_spatial = os.path.join(self.root_dir,self.animal_id, 'tif_files',
                                             session,
                                             #session+'_code_04_trial_ROItimeCourses_15sec_pca_0.95_spatial.npy')
                                             session+
                                             #'_code_04_trial_ROItimeCourses_30sec_pca_0.95_spatial.npy')
                                             '_whole_stack_trial_ROItimeCourses_15sec_pca30components_spatial.npy')

                spatial = np.load(fname_spatial)
                spatial = np.transpose(spatial,[1,0])
                denoised_spatial_name = np.reshape(spatial,[128,128,-1])
                # print ("denoised_spatial_name: ", denoised_spatial_name.shape)
                #

                ###########################################################
                # load temporal PC components
                temporal_whole_stack = np.load(fname_spatial.replace('_spatial',''))

                ##
                # temporal_random = np.load(fname_spatial.replace('trial','random').replace('_spatial',''))

                # make sure there are same # of trials in random and trial dataset
                #min_trials = min(temporal_trial.shape[0], temporal_random.shape[0])
                #temporal_trial = temporal_trial[:min_trials]
                #temporal_random = temporal_random[:min_trials]

                #
                temporal= temporal_whole_stack
                #temporal = np.transpose(temporal,[1,0,2])  #feautures, n_trials, n_times

                # flatten whole stack
                #denoised_temporal_name = np.reshape(temporal,[-1,temporal.shape[1]*temporal.shape[2]])
                denoised_temporal_name = temporal.transpose(1,0)

                #print('loaded data',flush=True)

                #######################################
                # Get data in the correct format
                V=denoised_temporal_name
                U=denoised_spatial_name

                #
                brainmask = np.ones(U.shape[:2],dtype=bool)

                # Load true areas if simulated data
                simulation=0

                # Include nan values of U in brainmask, and put those values to 0 in U
                brainmask[np.isnan(np.sum(U,axis=2))]=False
                U[np.isnan(U)]=0

                # Preprocess V: flatten and remove nans
                dimsV=V.shape
                keepinds=np.nonzero(np.sum(np.isfinite(V),axis=0))[0]
                V=V[:,keepinds]

                #
                if V.shape[0]!=U.shape[-1]:
                    print('Wrong dimensions of U and V!')

                print("Rank of video : %d" % V.shape[0])
                print("Number of timepoints : %d" % V.shape[1]);


                ##################################################
                ##################################################
                ##################################################
                # Perform the LQ decomposition. Time everything.
                t0_global = time.time()
                t0 = time.time()
                if nonnegative_temporal:
                    r = V.T
                else:
                    q, r = np.linalg.qr(V.T)
                # time_ests={'qr_decomp':time.time() - t0}

                # Put in data structure for LocaNMF
                video_mats = (np.copy(U[brainmask]), r.T)
                rank_range = (1, maxrank, 1)
                del U


                ##################################################
                ##################################################
                ##################################################

                #
                region_mats = LocaNMF.extract_region_metadata(brainmask,
                                                              self.atlas,
                                                              min_size=min_pixels)

                #
                region_metadata = LocaNMF.RegionMetadata(region_mats[0].shape[0],
                                                         region_mats[0].shape[1:],
                                                         device=device)

                #
                region_metadata.set(torch.from_numpy(region_mats[0].astype(np.uint8)),
                                    torch.from_numpy(region_mats[1]),
                                    torch.from_numpy(region_mats[2].astype(np.int64)))


                ##################################################
                ##################################################
                ##################################################

                # grab region names
                rois=np.load('/home/cat/code/widefield/locanmf/rois_50.npz')
                rois_name=rois['names']

                rois_ids=rois['ids']

                ##################################################
                ##################################################
                ##################################################

                # Do SVD as initialization
                if device=='cuda':
                    torch.cuda.synchronize()

                #
                print('v SVD Initialization')
                t0 = time.time()
                region_videos = LocaNMF.factor_region_videos(video_mats,
                                                             region_mats[0],
                                                             rank_range[1],
                                                             device=device)
                #
                if device=='cuda':
                    torch.cuda.synchronize()
                print("\'-total : %f" % (time.time() - t0))
                #time_ests['svd_init'] = time.time() - t0


                #
                low_rank_video = LocaNMF.LowRankVideo(
                    (int(np.sum(brainmask)),) + video_mats[1].shape, device=device
                )
                low_rank_video.set(torch.from_numpy(video_mats[0].T),
                                   torch.from_numpy(video_mats[1]))



                ##################################################
                ##################################################
                ##################################################
                if device=='cuda':
                    torch.cuda.synchronize()

                #
                print('v Rank Line Search')
                t0 = time.time()
                try:
                    #
                    locanmf_comps,loc_save,save_lam,save_scale,save_per,save_spa,save_scratch = LocaNMF.rank_linesearch(low_rank_video,
                                                                region_metadata,
                                                                region_videos,
                                                                maxiter_rank=maxrank,
                                                                maxiter_lambda=maxiter_lambda,
                                                                maxiter_hals=maxiter_hals,
                                                                lambda_step=lambda_step,
                                                                lambda_init=lambda_init,
                                                                loc_thresh=loc_thresh,
                                                                r2_thresh=r2_thresh,
                                                                rank_range=rank_range,
                    #                                             nnt=nonnegative_temporal,
                                                                verbose=[True, False, False],
                                                                sample_prop=(1,1),
                                                                device=device
                                                             )
                    if device=='cuda':
                        torch.cuda.synchronize()

                except Exception as e:
                    print (" locaNMF Failed, skipping")
                    print (e)
                    print ('')
                    print ('')
                    continue
                #
                if device=='cuda':
                    torch.cuda.synchronize()


                # C is the temporal components
                C = np.matmul(q,locanmf_comps.temporal.data.cpu().numpy().T).T
                print ("n_comps, n_time pts x n_trials: ", C.shape)
                qc, rc = np.linalg.qr(C.T)


                # Assigning regions to components
                region_ranks = []; region_idx = []

                for rdx in torch.unique(locanmf_comps.regions.data, sorted=True):
                    region_ranks.append(torch.sum(rdx == locanmf_comps.regions.data).item())
                    region_idx.append(rdx.item())

                areas=region_metadata.labels.data[locanmf_comps.regions.data].cpu().numpy()

                # Get LocaNMF spatial and temporal components
                A=locanmf_comps.spatial.data.cpu().numpy().T
                A_reshape=np.zeros((brainmask.shape[0],brainmask.shape[1],A.shape[1]));
                A_reshape.fill(np.nan)
                A_reshape[brainmask,:]=A

                # C is already computed above delete above
                if nonnegative_temporal:
                    C=locanmf_comps.temporal.data.cpu().numpy()
                else:
                    C=np.matmul(q,locanmf_comps.temporal.data.cpu().numpy().T).T

                # Add back removed columns from C as nans
                C_reshape=np.full((C.shape[0],dimsV[1]),np.nan)
                C_reshape[:,keepinds]=C
                C_reshape=np.reshape(C_reshape,[C.shape[0],dimsV[1]])

                # Get lambdas
                lambdas=np.squeeze(locanmf_comps.lambdas.data.cpu().numpy())

                print ("A_reshape: ", A_reshape.shape)
                print ("C_reshape: ", C_reshape.shape)



                ##################################################
                ##################################################
                ##################################################
                # save LocaNMF data
                areas_saved = []
                for area in areas:
                    idx = np.where(rois_ids==np.abs(area))[0]
                    temp_name = str(rois_name[idx].squeeze())
                    if area <0:
                        temp_name += " - right"
                    else:
                        temp_name += " - left"

                    areas_saved.append(temp_name)

                # GET AREA NAMES
                def parse_areanames_new(region_name,rois_name):
                    areainds=[]; areanames=[];
                    for i,area in enumerate(region_name):
                        areainds.append(area)
                        areanames.append(rois_name[np.where(rois_ids==np.abs(area))][0])
                    sortvec=np.argsort(np.abs(areainds))
                    areanames=[areanames[i] for i in sortvec]
                    areainds=[areainds[i] for i in sortvec]
                    return areainds,areanames

                #
                region_name=region_mats[2]

                # Get area names for all components
                areainds,areanames_all = parse_areanames_new(region_name,rois_name)
                areanames_area=[]
                for i,area in enumerate(areas):
                    areanames_area.append(areanames_all[areainds.index(area)])

                ###################################
                np.savez(fname_out,
                          whole_stack = C_reshape,
                          areas = areas,
                          names = areas_saved,
                          A_reshape = A_reshape,
                          areanames_area = areanames_area
                         )


        print (" ... DONE ALL SESSIONS...")




    def show_ROIs(self, session=None):

        if session is None:
            session = self.sessions[0]

        fname_in = os.path.join(self.root_dir,self.animal_id,'tif_files',
                                  session,session+'_locanmf.npz')
        data = np.load(fname_in, allow_pickle=True)

        A_reshape = data["A_reshape"]
        areanames_area = data['areanames_area']

        ######################################################
        fig=plt.figure()
        for i in range(A_reshape.shape[2]):
            plt.subplot(4,4,i+1)
            plt.imshow(A_reshape[:,:,i])
            plt.title(areanames_area[i],fontsize=6)
        plt.tight_layout(h_pad=0.5,w_pad=0.5)
        plt.show()



def plot_polar_plots(t0_phases):

    clrs_local = ['magenta','brown','pink','lightblue','darkblue', 'blue']
    codes = ['_Retrosplenial', '_barrel', '_limb', '_visual','_motor','']

    fig2=plt.figure(figsize=(15,5))

    #
    for k in range(len(t0_phases)):
        ax = fig2.add_subplot(1,5,k+1, projection='polar')
        plt.title(codes[k].replace('_',''),fontsize=10)
        N = 16

        #
        conversion= 1

        #
        bins = np.linspace(0,2*np.pi*conversion, N+1)
        phases1 = np.array(t0_phases[k])*conversion
        y = np.histogram(phases1,
                        bins = bins
                        #bins = np.arange(0,360+45,45),
                        )
        #
        theta = y[0]#[:-1]
        theta = theta/np.max(theta)  # not necessary
        radii = y[1][1:]

        #
        width = (2*np.pi) / (N)
        ax.bar(radii-width/2.,
               theta,
               width=width,
               color=clrs_local[k])

        #
        ax.set_yticklabels([])
        #plt.title(codes[k])
        ax.yaxis.grid(False)

        #ax.set_xticks([])
        ax.xaxis.set_tick_params(labelsize=12)
        #if True:
        #    ax.set_xticklabels([])



def plot_histograms_schimdthetal(data_dir, animal_ids, feat):
    #
    clrs_anims = ['red','blue']

    #
    codes = ['_Retrosplenial', '_barrel', '_limb', '_visual','_motor','']
    clrs_local = ['red','blue','pink','lightblue','darkblue', 'green']
    #feat = 4

    #
    fig=plt.figure()
    for ctra, animal_id in enumerate(animal_ids):
        fname = os.path.join(data_dir,
                             'all_phases_all_trials_'+animal_id+'.npy')

        data2 = np.vstack(np.load(fname,
                      allow_pickle=True))

        #
        data = []
        for k in range(len(data2)):
            data.append(data2[k][feat])

        if True:  # all sessions
            data = np.hstack(data)
        else:
            data = np.hstack(t0_phases[feat])  #single session


        #########################################
        yy = data/np.pi*180
        width = 90
        offset = 0
        bins = np.arange(offset,361+offset,width)
        y = np.histogram(yy, bins=bins)

        yy = y[0]
        yy = np.roll(yy,3)
        shift = 90
        plt.bar(y[1][:-1]+width//2+ctra*30,
                yy/np.sum(yy),
                width//3,
                color=clrs_local[ctra],
                alpha=.8,
                label = animal_id)

        #
        plt.xlim(y[1][0],y[1][-2]+width*1.5)

          #
    plt.suptitle("Phases of "+codes[feat])
    labels2 = ['crest','falling','trough','rising']
    plt.xticks(np.arange(0,360,90)+60, labels2)
    plt.legend()



def plot_phases2(session_id,
                animal_id,
                session_name,
                root_dir,
                random_flag,
                areas_to_plot_phases,
                start,
                end,
                show_area_id,
                codes,
                clrs_local,
                plotting):

    ''' function that fits phases to raw locanm components

    '''

    if True:
        #print ("session_id: ", session_id)
        (atlas,
         areas,
         names,
         locaNMF_temporal,
         locaNMF_temporal_random) = load_locaNMF_temporal(animal_id,
                                                          session_name,
                                                          root_dir,
                                                          session_id)
    # except:
    #    # print (" couldn't load file")
    #     return ([[],[],[],[],[]])

        #     for ctr, name in enumerate(names):

#         print (ctr, name)

    #print ("GOT HERE")
    if atlas is None:
        #print ("session is empty ")
        return ([[],[],[],[],[]])
    #
    if plotting:
        fig=plt.figure(figsize=(15,10))
        colors = plt.cm.viridis(np.linspace(0,1,len(locaNMF_temporal)))

    # loop over areas
    t0_phases=[]

    for ctr_area, area_sel in enumerate(areas_to_plot_phases):
        #
        t0_phases.append([])

        #
        areas_selected = []
        for k in range(len(names)):
            if area_sel in names[k]:
                areas_selected.append(k)

        # use random data instead
        if random_flag:
            locaNMF_temporal = locaNMF_temporal_random

        #
        areas_selected = np.array(areas_selected)
        locaNMF_temporal2 = locaNMF_temporal[:,areas_selected].squeeze()

        #
        missed_fit = 0
        t = np.arange(locaNMF_temporal2.shape[1])/30-30
        curves=[]

       # print ("# trials: ", locaNMF_temporal2.shape[0])
        for k in range(locaNMF_temporal2.shape[0]):
            if ctr_area==show_area_id:
                if plotting:
                    ax=plt.subplot(2,1,1)
                    plt.plot(t,locaNMF_temporal2[k],
                         c='black',
                         linewidth=3,
                         alpha=.1)

            # fit sinusoid to the single trial data
            tt = t[start:end]
            yy = locaNMF_temporal2[k][start:end]

            #
            try:
                tt2 = np.arange(0, locaNMF_temporal2[k].shape[0],1)/30.-30
                res = fit_sin(tt, yy)
                curve = res["fitfunc"](tt2)

                #print ("CURVE: ", curve.shape)
                # if curve fit, extrapolate all the way to -30sec

                if plotting:
                    if ctr_area==show_area_id:
                        ax=plt.subplot(2,1,2)
                        curve2 = (curve-np.min(curve))/(np.max(curve)-np.min(curve))*2-1
                        curves.append(curve2)
                        plt.plot(tt2,
                                 curve2,
                                 linewidth=2,
                                 #c=clrs_local[show_area_id],
                                 c='pink',
                                 alpha=.5
                            )
                        plt.scatter(tt2[900],
                                 curve2[900],
                                 edgecolor='black',
                                 c=clrs_local[show_area_id],
                                 alpha=1
                            )

                # get peaks of curves
                phase = get_angle_from_sinusoid(curve, 900)

                #
                t0_phases[ctr_area].append(phase)

            #
            except:
                t0_phases[ctr_area].append(np.nan)
                missed_fit+=1


        ###############################
        ######## PLOT MEANS ###########
        ###############################
        if ctr_area==show_area_id:
            #curves=np.array(curves)
            mean = np.mean(locaNMF_temporal2,axis=0)
            if plotting:
                ax=plt.subplot(2,1,1)
                plt.plot(t,mean,
                         c='black',
                         linewidth=8,
                         alpha=1)

                #
                plt.plot([-30,30],[0,0],'--',c='grey')
                plt.xlim(-15,1)
                plt.plot([0,0],[-10,10],'--',c='grey')
                plt.ylim(-0.15,0.20)


            # fit sinusoid to trial average
            try:
                tt = t[start:end]
                curves=np.array(curves)
                mean = np.mean(curves,axis=0)
                tt2 = np.arange(0,mean.shape[0],1)/30.-30
                yy = mean[start:end]

                res = fit_sin(tt, yy)
                #print( "Amplitude=%(amp)s, Angular freq.=%(omega)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s" % res )

                # PLOT REAL AVERAGE NOT FIT

                if plotting:
                    mean = np.mean(curves,axis=0)
                    ax=plt.subplot(2,1,2)
                    #plt.plot(tt, yy, "-k", label="y", linewidth=2)
                    plt.plot(tt2, mean, linewidth=8,
                            #c=clrs_local[show_area_id]
                            c='pink'
                            )

                    #
                    plt.plot([-30,30],[0,0],'--',c='grey')
                    plt.plot([0,0],[-10,10],'--',c='grey')
                    plt.xlim(-15,1)
                    plt.ylim(-0.15,0.15)
                    plt.ylim(-1,1)
            except:
                pass


    ###############################################################
    ###############################################################
    ###############################################################

    my_dict = dict(restrosplenial = t0_phases[0],
                   barrel = t0_phases[1],
                   limb = t0_phases[2],
                   visual = t0_phases[3],
                   motor = t0_phases[4])

    data = pd.DataFrame.from_dict(my_dict, orient='index')
    data = data.transpose()

    #
    flierprops = dict(marker='o',
                      #markerfacecolor='g',
                      #markersize=10000,
                      linestyle='none',
                      markeredgecolor='r')

    #
    if False:
        ax = fig.add_subplot(313)
        data.boxplot(showfliers=False,
                flierprops=flierprops)

        #
        for i,d in enumerate(data):
            colors = plt.cm.viridis(np.linspace(0,1,len(t0_phases[i])))
            y = data[d]
            x = np.random.normal(i+1, 0.04, len(y))
            if False:
                plt.plot(x, y,
                     mfc = 'black',
                     mec='k',
                     ms=7,
                     marker="o",
                     linestyle="None",
                        )
            #
            else:
                x = np.random.normal(i+1, 0.04, len(t0_phases[i]))
                #print (i,d, ' y shape: ', y.shape)
                plt.scatter(x, t0_phases[i],
                           #c=clrs_local[i],
                           c=clrs_local[i],
                           edgecolor='black',
                           s=200,
                           #alpha=np.linspace(.2, 1.0, x.shape[0])
                           alpha=.2
                           )

        plt.ylim(0,2*np.pi)

    # print ("# of non-fit trias: ", missed_fit)

    return t0_phases

def plot_all_sessions_polar_plots(root_dir,
                                  animal_id,
                                  random_flag):
# #
#     #
#     areas_to_plot_phases = ['Retrosplenial area, dorsal part, layer 1 - left',
#                             'Primary somatosensory area, barrel field, layer 1 - left',
#                             'upper limb, layer 1 - left',
#                             'Primary visual area, layer 1 - left',
#                             'Primary motor area, Layer 1 - left',
#                            ]

    #
    codes = ['_Retrosplenial', '_barrel', '_limb', '_visual','_motor','']

    #
    if random_flag==False:
        fname = os.path.join(root_dir,'all_phases_all_trials_'+animal_id+'.npy')
    else:
        fname = os.path.join(root_dir,'all_phases_all_trials_random_'+animal_id+'.npy')

    #
    t0_array = np.load(fname, allow_pickle=True)

    #####################################
    ############ MAKE PLOTS #############
    #####################################
    t0_phases = []
    for k in range(5):
        t0_phases.append([])

    #
    #fig=plt.figure(figsize=(15,3))
    for k in range(len(t0_array)):
        for p in range(len(t0_array[0])):
            t0_phases[p].append(t0_array[k][p])

    for k in range(5):
        t0_phases[k] = np.hstack(t0_phases[k])


    plot_polar_plots(t0_phases)

    import pycircstat
    for k in range(len(t0_phases)):
        temp = t0_phases[k]
        idx = np.where(np.isnan(temp)==False)[0]
        temp = temp[idx]

        p, z = pycircstat.tests.rayleigh(temp)
        print (animal_id, codes[k], "rayleightest test : ", p,z)

    plt.show()
    # #
    # if False:
    #     plt.savefig('/home/cat/all_polar_'+animal_id+str(random_flag)+'.svg')
    #     plt.close()
    # else:
    #     plt.show()
    # #ctr_a+=1



from scipy.signal import butter, lfilter, filtfilt, hilbert, chirp

def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = filtfilt(b, a, data)
    return y



def plot_RP_and_hilbert_transform(data_dir, animal_id, session_id, random_flag):

    fname = os.path.join(data_dir,
                         animal_id,
                         session_id,
                         session_id+'_locanmf.npz')

    data = np.load(fname, allow_pickle=True)

    #
    trials = data['temporal_trial']
    random = data['temporal_random']

    #
    names = data['names']
    name = 'motor'


    t = []
    r = []
    for k in range(trials.shape[1]):
        if name in names[k]:
            t.append(trials[:,k].mean(0))
            r.append(random[:,k].mean(0))

    #
    t = np.array(t).mean(0)
    r = np.array(r).mean(0)

    if random_flag==False:
        pass
    else:
        t=r.copy()

    #
    filter_cutoff = 14
    t = butter_lowpass_filter(t, filter_cutoff,30)*100

    #
    x = np.arange(t.shape[0])/30.-30

    #
    analytic_signal = hilbert(t)
    amplitude_envelope = np.abs(analytic_signal)
    amplitude_envelope = butter_lowpass_filter(amplitude_envelope,.5,30)

    #

    start = 300
    end = 1050
    t = t[start:end]
    r = r[start:end]
    x = x[start:end]
    amplitude_envelope = amplitude_envelope[start:end]

    plt.plot(x, t, c='darkblue', linewidth=5, label='Average session neural signal')
    plt.plot(x, amplitude_envelope, '--', c='black', linewidth=1, label="Hilbert transform envelope")


    #
    plt.plot([x[0], x[-1]],[0,0],'--',c='black', linewidth=3)
    plt.plot([x[600], x[600]],[-5,7],'--',c='black', linewidth=3)

    plt.xlim(x[0],x[-1])
    plt.ylim(-5,8)
    plt.ylabel("DFF")
    plt.xlabel("Time (sec)")
    plt.legend()

#
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    #print ("box: ", box.shape)
    y_smooth = np.convolve(y, box, mode='same')
    #print ("y)mooth: ", y_smooth.shape)
    return y_smooth

#
def plot_svm_accuracy(data_dir,animal_id, session_id):

    #
    fname = os.path.join(data_dir,
                     animal_id,
                     'tif_files',
                     session_id,
                     "SVM_Scores_"+session_id+
                    "code_04_trial_ROItimeCourses_30sec_Xvalid10_Slidewindow30.npz")
    #
    data = np.load(fname, allow_pickle=True)
    accuracy = data['accuracy']

    # convert to percentage
    accuracy = accuracy *100

    # smooth accuracy using same SVM window length
    svm_window_len = 30
    for k in range(accuracy.shape[1]):
        accuracy[:,k] = smooth(accuracy[:,k], svm_window_len)

    #
    means = accuracy.mean(axis=1)
    std = np.std(accuracy,axis=1)

    #
    t=np.arange(means.shape[0])/30.-29
    plt.plot(t, means)
    plt.fill_between(t, means+std, means-std,
                     color='blue',
                     alpha=.2)
    #
    plt.plot([-20,4],[50,50],'--', c='black')
    plt.plot([0,0],[40,100],'--', c='black')
    plt.xlim(-20,4)
    plt.ylim(40,100)
    plt.ylabel("SVM Decoding Accuracy (%)")
    plt.xlabel("Time to lever pull (sec)")


#####################################
class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False
        super(LinearRegression, self)\
                .__init__(*args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([
            np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                                                    for i in range(sse.shape[0])
                    ])

        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        return self

def compute_significance2(data,
                     significance):

    #print ("self.data: ", data.shape)

    mean = data.mean(1)
    #
    sig = []
    for k in range(data.shape[0]):
        #res = stats.ks_2samp(self.data[k],
        #                     control)
        #res = stats.ttest_ind(first, second, axis=0, equal_var=True)

        #
        res = scipy.stats.ttest_1samp(data[k], 0.5)

        sig.append(res[1])


    sig_save = np.array(sig).copy()
    #print ("Self sig save: ", sig_save.shape)

    # multiple hypothesis test
    temp = np.array(sig)
    #print ("data into multi-hypothesis tes:", temp.shape)
    temp2 = multipletests(temp, alpha=significance, method='fdr_bh')
    sig = temp2[1]

    #
    sig=np.array(sig)[None]

    #
    thresh = significance
    idx = np.where(sig>thresh)
    sig[idx] = np.nan

    #
    idx = np.where(mean<0.5)
    sig[:,idx] = np.nan
    #print ("SIG: ", sig.shape)

    # find earliest
    earliest_continuous = 0
    for k in range(sig.shape[1]-1,0,-1):
        if sig[0][k]<=significance:
            earliest_continuous = k
        else:
            break

    earliest_continuous = -(sig.shape[1]-earliest_continuous)/30.

    return sig, earliest_continuous

#
def plot_longitudinal_svm_accuracy_concatenated_sessions(root_dir,
                                                         names):
    #
    fig = plt.subplots(figsize=(20,10))
    animal_ids = ["M1", "M2", "M3", "M4","M5",'M6']

    #
    smooth = True
    smooth_window = 30
    significance = 0.05
    #auc = []
    #early = []
    #from tqdm import tqdm, trange
    for k, name in enumerate(names):
        fnames = np.loadtxt(os.path.join(root_dir,name,
                                         'concatenated_svm.txt'),dtype='str')

        ax=plt.subplot(2,3,k+1)
        colors = plt.cm.viridis(np.linspace(0,1,len(fnames)))

        #
        fname_accs = os.path.join(root_dir,name,
                                         'acc_all_sessions_smooth.npy')
        fname_edts = os.path.join(root_dir,name,
                                         'edts_all_sessions_smooth.npy')
        fname_aucs = os.path.join(root_dir,name,
                                         'aucs_all_sessions_smooth.npy')
        acc_array = []
        edts_array = []
        auc_array = []
        if os.path.exists(fname_accs):
            acc_array = np.load(fname_accs)

             ##############################
            #
            for ctr,acc in enumerate(acc_array):
                mean = acc.mean(1)
                t = np.arange(mean.shape[0])/30-28
                ax.plot(t,mean,
                        linewidth=2,
                        color=colors[ctr])

        else:
            for ctr,fname in tqdm(enumerate(fnames)):

                #
                data = np.load(fname, allow_pickle=True)
                acc = data['accuracy']

                #print ('acc: ', acc.shape)
                if smooth:
                    data = []
                    for p in range(acc.shape[1]):
                        box = np.ones(smooth_window)/smooth_window

                        trace_smooth = np.convolve(acc[:,p],
                                                   box,
                                                   mode='valid')

                        data.append(trace_smooth)
                    data = np.array(data)
                    acc = np.array(data).copy().T

                acc_array.append(acc)

                ##############################
                #
                mean = acc.mean(1)
                t = np.arange(mean.shape[0])/30-28
                #print ("t.shape,", t.shape, 't: ', t[:10])
                ax.plot(t,mean,
                        linewidth=2,
                        color=colors[ctr])

                # append area under cruve up to t = -10sec
                temp = mean[-300:]  # last 10sec prior to movement
                auc_array.append(temp.sum())

                sig, earliest = compute_significance2(data.T,
                                                     significance)
                edts_array.append(earliest)


            np.save(fname_accs, acc_array)
            np.save(fname_edts, edts_array)
            np.save(fname_aucs, auc_array)

        plt.ylim(0.4,1.0)
        plt.xlim(-15,0)
        plt.plot([-30,0],[0.5,0.5],'--',c='black')

        #
        plt.xlabel("Time (sec)")
        plt.ylabel("SVM accuracy")
        plt.title(animal_ids[k])


#       
def plot_edts_longitudinal(root_dir, animal_ids):

    colors = ['black','blue','red','green','magenta','pink','cyan']
    fig=plt.figure(figsize=(6,6))
    names_biorxiv = ["M1", "M2", "M3", "M4","M5",'M6']

    #
    for k in range(len(animal_ids)):

        ax=plt.subplot(3,2,k+1)

        edts = np.load(os.path.join(root_dir,animal_ids[k],
                                   'edts_all_sessions_smooth.npy'))

        #temp = np.array(early[k])
        temp = edts

        #
        idx =np.where(temp<-20)[0]
        temp[idx]=temp[idx-1]
        t = np.arange(temp.shape[0])/temp.shape[0]

        #
        lr = LinearRegression()
        lr.fit(t.reshape(-1, 1), temp.reshape(-1, 1))

        #
        temp2 = np.poly1d(np.polyfit(t, temp, 1))(t)
        plt.plot(t, temp2,
                 linewidth=6,
                 #label=ids[k]+ " ***** ",
                 c='black')

        #
        corr = scipy.stats.pearsonr(t,temp)
        print ("corrL: ", corr)

        plt.scatter(t,
                    temp,
                    #label=names[k]+ " "+str(round(corr[0],2))+
                    #                        " ("+str("pval: {0:.1}".format(corr[1]))+")",
                    s=100,
                    #linewidth=4,
                    c='grey',
                    alpha=.8,
                   label = str(round(corr[0],2)))
        #t = np.arange(auc1.shape[0])/(auc1.shape[0]-1)

        plt.xticks([])
        #plt.yticks([])
        legend = plt.legend(handlelength=0, fontsize=12)

        #plt.legend(fontsize=12)
        plt.ylim(-12,0)
        plt.xlim(0,1)
        plt.title(names_biorxiv[k])

#
def generate_convex_hulls(pa):

    #pa.plot_all = True
    pa.plot_3D = False
    pa.alpha1 = 1.0
    pa.alpha2 = 0.02
    pa.knn_triage = .1
    pa.knn_triage_based_on_last_point = False           # this flag removes all traces based on t=0 triaging
                                                        # basically we use the outliers at t=0 to triage other time points;
    pa.cmap = matplotlib.cm.get_cmap('jet')
    pa.linewidth = 5
    pa.t0_hull = False
    pa.t0_dynamics = False
    pa.plot_dynamics = False

    #############################
    plot_pca_scatter(pa)

    #plt.title(session+ " triage: "+ (str(pa.knn_triage)))

    
    

def plot_convex_hulls_10sec(root_dir, animal_id, session):
#

    #
    t1 = np.arange(30,310,10)
    t2 = np.arange(0,270,10)

    #
    cmap = matplotlib.cm.get_cmap('Reds', len(t1))

    #
    plt.figure(figsize=(10,10))

    #
    fname_all_points = os.path.join(root_dir, animal_id,
                                    'tif_files',
                                     session,
                                    'pa_all_points.npy')
    fname_p_lever = os.path.join(root_dir, animal_id,
                                 'tif_files',
                                  session,
                                 'pa_p_lever.npy')
    # 
    pa_all_points = np.load(fname_all_points)
    pa_p_lever = np.load(fname_p_lever)
    #pa_all_points = []
    #pa_p_lever = []
    ctr=0
    for t1,t2 in zip(t1,t2):
        #print ("Times: ", t1,t2)

        pa = PCA_Analysis()
        pa.root_dir = root_dir #'/media/cat/4TBSSD/yuki/'
        pa.animal_id = animal_id

        #
        pa.use_pca_data = True    # this uses the PCA denoised STMs not Raw data!
        pa.recompute = True
        pa.n_pca = 20
        pa.sliding_window = 30    # how many frames to take into analysis window
        pa.n_frames = 30          # how many frames back in time to analyze:
        pa.session = session

        # #
        # pa.X, pa.triggers = get_data_and_triggers(pa)
        #
        # #
        # pa.pca, pa.all_points = get_pca_object_and_all_points(pa)
        #
        # #
        # pa_all_points.append(pa.all_points)

        pa.all_points = pa_all_points[ctr]

        #
        pa.t1 = t1
        pa.t2 = t2
        # pa.p_lever = project_data_pca2(pa)
        #
        # pa_p_lever.append(pa.p_lever)

        pa.p_lever = pa_p_lever[ctr]
        if t1 ==30:
            pa.plot_all=True
        else:
            pa.plot_all=False

        pa.hull_clr = cmap(ctr)

        #
        generate_convex_hulls(pa)

        ctr+=1

    #
    np.save(fname_all_points, pa_all_points)
    np.save(fname_p_lever, pa_p_lever)


def plot_area_under_curve_1sec_segments(root_dir,
                                       animal_id,
                                       session):
    #
    pa = PCA_Analysis()
    pa.root_dir = root_dir
    pa.animal_id = animal_id
    pa.session = session

    #
    pa.use_pca_data = True    # this uses the PCA denoised STMs not Raw data!
    pa.recompute = True
    pa.n_pca = 20
    pa.sliding_window = 30    # how many frames to take into analysis window
    pa.n_frames = 300          # how many frames back in time to analyze:

    #
    pa.t1 = 300
    pa.t2 = 270

    #
    pa.t0_hull = False
    pa.t0_dynamics = False
    pa.plot_dynamics = False

    data = np.load(os.path.join(root_dir, animal_id,
                                'tif_files',
                                session,
                                session+"_convex_hull.npz"),allow_pickle=True)

    #
    pa.ratio_cumsum = data['ratio_cumsum']
    pa.ratio_random_cumulative = data['ratio_random_cumulative']
    pa.ratio_random_single = data['ratio_random_single']
    pa.ratio_single = data['ratio_single']

    #
    plot_convex_hull22(pa)


def get_aucs_norms():
    animal_ids = ['IA1','IA2','IA3','IJ1',"IJ2","AQ2"]
    clrs = ['black','blue','red','green','magenta','pink']

    aucs = []
    aucs_norm = []

    plotting = False
    for ctr_animal, animal_id in enumerate(animal_ids):

        #
        aucs.append([])
        aucs_norm.append([])

        #
        pa = PCA_Analysis()
        pa.root_dir = '/media/cat/4TBSSD/yuki/'
        pa.session_id = 'all'
        pa.animal_id = animal_id

        #
        sessions = get_sessions(pa.root_dir,
                         pa.animal_id,
                         pa.session_id)
        #
        cmap = matplotlib.cm.get_cmap('Reds')

        clr_ctr=0
        n_sessions = len(sessions)
        alpha = .5

        if plotting:
            fig=plt.figure(figsize=(7,5))
        for session in sessions:
            print ('session:', session)
            #
            fname = os.path.join(pa.root_dir, pa.animal_id,'tif_files',session,
                                session+ '_convex_hull.npz')

            try:
                d = np.load(fname, allow_pickle = True)
            except:
                clr_ctr+=1
                continue

            pa.ratio_single = d['ratio_single'][::-1]
            pa.ratio_cumsum = d['ratio_cumsum'][::-1]
            pa.ratio_random_single = d['ratio_random_single'][::-1]
            pa.ratio_random_cumulative = d['ratio_random_cumulative'][::-1]

            clr_ctr+=1

            #
            aucs[ctr_animal].append(pa.ratio_cumsum.sum())
            aucs_norm[ctr_animal].append((pa.ratio_cumsum/np.max(pa.ratio_cumsum)).sum())


    return aucs_norm


def plot_longitudinal_area_under_curve(root_dir,
                                       animal_ids):
    #
    from sklearn.linear_model import LinearRegression as LR2

    aucs_norm = np.load(os.path.join(root_dir,'aucs_norm.npy'),allow_pickle=True)

    #
    fig=plt.figure(figsize=(6,6))
    for k in range(len(aucs_norm)):
        plt.subplot(3,2,k+1)

        #
        temp = aucs_norm[k]
        plt.scatter(np.arange(len(temp)), temp,
                    c='red',
                    edgecolor='black',
                    s=100,
                    alpha=.5)

        x = np.arange(len(temp))
        corr = scipy.stats.pearsonr(x,temp)
        print ("cor: ", corr)

        # fit
        model = LR2()
        y = np.array(temp).reshape(-1, 1)
        x = np.arange(y.shape[0]).reshape(-1, 1)
        model.fit(x, y)

        x2 = np.arange(0,y.shape[0],1).reshape(-1, 1)
        y_pred = model.intercept_ + model.coef_ * x2

        plt.plot(x2, y_pred, label= str(round(corr[0],2)),
                 c='black',
                 linewidth=6)



        plt.legend(handlelength=0, fontsize=16)
        plt.xlim(x[0],x[-1])
        plt.ylim(np.min(y), np.max(y))
        #print (" COMPUTE PEARSON CORR NOT T-TEST ON FIT")
        #print (np.min(y), np.max(y))
        #plt.ylim(0,2)

        plt.xticks([])
        plt.yticks([])

def plot_longitudinal_lever_hull(root_dir, animal_names):

    from sklearn.linear_model import LinearRegression as LR2

    animal_ids = np.arange(len(animal_names))
    names_biorxiv = ["M1", "M2", "M3", "M4","M5",'M6']

    areas = np.load(os.path.join(root_dir,
                                 'hull_areas.npy'),allow_pickle=True)
    overlaps = np.load(os.path.join(root_dir,
                                 'hull_overlaps.npy'),allow_pickle=True)

    fig=plt.figure(figsize=(10,10))
    clr='magenta'
    for animal_id in animal_ids:
        ax = plt.subplot(3,2,animal_id+1)
        ctr=0
        data=[]
        for k in range(len(areas[animal_id])):
            flag = False
            try:
                area_right = areas[animal_id][k][2]
                area_all = areas[animal_id][k][3]
                temp = area_right/area_all
                if np.isnan(temp)==False:
                    plt.scatter(ctr, temp,
                                s=300,
                                c=clr,
                               alpha=.6)
                    ctr+=1
                    data.append([ctr,temp])
            except:
                pass

            #print (k, ctr)

        #
        data = np.array(data)
        #print ("data: ", data.shape)

        # fit line
        x = data[:,0]
        y = data[:,1]
        corr = scipy.stats.pearsonr(x,y)
        print (animal_id, "cor: ", corr)


        # fit
        model = LR2()
        y=np.array(y).reshape(-1, 1)
        x = np.arange(y.shape[0]).reshape(-1, 1)
        model.fit(x, y)

        x2 = np.arange(0,y.shape[0],1).reshape(-1, 1)
        y_pred = model.intercept_ + model.coef_ * x2


        plt.plot(x2, y_pred, 
                 label= "pcorr: "+str(round(corr[0],5))+"\n"+
                 "pval: "+str(round(corr[1],5)),
                 c='black',
                 linewidth=6)


        legend = plt.legend(handlelength=0, fontsize=16)
        #plt.legend(fontsize=12)
        plt.xlim(x[0],x[-1])
        plt.ylim(np.min(y), np.max(y))
        #print (" COMPUTE PEARSON CORR NOT T-TEST ON FIT")
        #print (np.min(y), np.max(y))
        #plt.ylim(0,2)

        plt.xticks([])
        plt.yticks([])
        plt.ylim(0,0.5)

        #print (k, ctr)
        plt.title(names_biorxiv[animal_id])



def plot_longitudinal_right_paw_intersection_lever(root_dir, animal_names):

    from sklearn.linear_model import LinearRegression as LR2

    animal_ids = np.arange(len(animal_names))
    names_biorxiv = ["M1", "M2", "M3", "M4","M5",'M6']

    areas = np.load(os.path.join(root_dir,
                                 'hull_areas.npy'),allow_pickle=True)
    overlaps = np.load(os.path.join(root_dir,
                                 'hull_overlaps.npy'),allow_pickle=True)

    clr='brown'
    plt.figure(figsize=(10,10))
    for animal_id in animal_ids:
        ax=plt.subplot(3,2,animal_id+1)
        ctr=0
        data= []
        for k in range(len(overlaps[animal_id])):
            #ax = plt.subplot(2,2,a+1)
            try:
                temp = overlaps[animal_id][k]
                if np.isnan(temp)==False:
                    plt.scatter(ctr, temp,
                                s=300,
                                c=clr,
                               alpha=.6)
                    ctr+=1
                    data.append([ctr,temp])
            except:
                pass

        #
        data = np.array(data)

        # fit line
        x = data[:,0]
        y = data[:,1]
        corr = scipy.stats.pearsonr(x,y)
        print ("cor: ", corr)


        # fit
        model = LR2()
        y=np.array(y).reshape(-1, 1)
        x = np.arange(y.shape[0]).reshape(-1, 1)
        model.fit(x, y)

        x2 = np.arange(0,y.shape[0],1).reshape(-1, 1)
        y_pred = model.intercept_ + model.coef_ * x2


        plt.plot(x2, y_pred, 
                 label= "pcorr: "+str(round(corr[0],5))+"\n"+
                 "pval: "+str(round(corr[1],5)),
                 c='black',
                 linewidth=6)


        plt.legend(handlelength=0, fontsize=16)
        plt.xlim(x[0],x[-1])
        plt.ylim(np.min(y), np.max(y))

        plt.xticks([])
        plt.yticks([])
        plt.ylim(0,1)
        plt.title(names_biorxiv[animal_id])

#
def find_first_variance_decrease_point2(data_in, s1, e1, std_factor, ctr,
                                       animal_id,
                                       n_vals_below_thresh=30):

    #
    if False:
    #if ctr!=3:
        data_in = savgol_filter(data_in, 31, 2)

    # find std of up to 10 sec prior to pull
    std = np.std(data_in[s1:e1], axis=0)

    # find mean up to 10 sec prior to pull
    mean2 = np.mean(data_in[0:e1], axis=0)

    # do rolling evalution to find location when next N vals are belw threhsold
    idx_out = np.nan
    #n_vals_below_thresh = 30
    window = [20*30,30*30 ]
    for k in range(window[0],
                   window[1], 1):
        # ensure that several time points in a row are below
        temp = data_in[k:k+n_vals_below_thresh]
        #print ("TEMP: ", temp)
        #print ("mean2: ", mean2)
        #print ("std*std_factor[ctr]: ", std*std_factor[ctr])
        if animal_id !='IA2':
            if np.all(temp<=(mean2-std*std_factor[ctr])):
                idx_out = k
                break
        else:
            if np.all(temp>=(mean2+std*std_factor[ctr])):
                idx_out = k
                break

    #
    if idx_out>(900-30//2):
        idx_out=np.nan

    return idx_out

#
def plot_all_variances(root_dir,
                      animal_id):
    from scipy import signal

    #
    data = np.load(os.path.join(root_dir,
                                animal_id,
                                animal_id+"_variances.npz"))

    all_means = data['all_means']
    all_vars = data['all_vars']
    saved_names = data['saved_names']

    #
    session_ids = np.arange(len(all_vars))

    #
    roi_ids = [15,5,9,11,1]
    colors = plt.cm.viridis(np.linspace(0,1,len(all_vars)))

    ###############################################
    ###############################################
    ###############################################
    plot_varsS = [False, True]

    for plot_vars in plot_varsS:

        #
        plt.figure(figsize=(20,4))
        std_factor = [2,2,2,4,2]
        n_vals = [10,15,20,20,20,30]
        print ("NOTE THESE VALUES AND FUCNTIONS HAVE BEEN UPDATED IN THE MAIN REPO --- DO NOT USE UNTIL UPDATED")
        first_decay = []

        # plot first decay point
        s1 = 400
        e1 = 600

        #
        for ctr, roi_id in enumerate(roi_ids):

            first_decay.append([])

            #
            plt.subplot(1,5,ctr+1)
            all_ = []
            for ctr_sess, session_id in enumerate(session_ids):
                if plot_vars:
                    temp = all_vars[session_id][roi_id].copy()
                else:
                    temp = all_means[session_id][roi_id].copy()
                t = np.arange(temp.shape[0])/30.-30

                #
                if False:
                    temp = savgol_filter(temp, 31, 2)

                else:
                    fs = 30
                    fc = 5  # Cut-off frequency of the filter

                    w = fc / (fs / 2) # Normalize the frequency
                    b, a = signal.butter(5, w, 'low')
                    temp = signal.filtfilt(b, a, temp)

                #
                plt.plot(t, temp,
                         color=colors[ctr_sess],
                         alpha=.5,
                         linewidth=3)

                idx = find_first_variance_decrease_point2(temp, s1, e1, std_factor, ctr,
                                                         animal_id,
                                                         n_vals[ctr])

                first_decay[ctr].append(idx)

                all_.append(temp)

            #
            a = np.array(all_)

            if False:
                a = savgol_filter(a, 31, 2)

            a_mean = np.mean(a,axis=0)
            plt.plot(t, a_mean,c='red',
                     linewidth=5,
                     alpha=1)


            plt.xlim(-15,5)

            #
            if plot_vars:
                #plt.ylim(bottom=0)
                plt.ylim(0,0.01)
                plt.plot([-30,30],[0,0],'--',c='grey')
                plt.plot([0,0],[-.2,.2],'--',c='grey')
            else:
                plt.ylim(-.075,.15)
                plt.plot([-30,30],[0,0],'--',c='grey')
                plt.plot([0,0],[-.2,.2],'--',c='grey')
            plt.title(saved_names[roi_id],fontsize=8)
            if ctr==0:
                if plot_vars:
                    plt.ylabel("Variance")
                else:
                    plt.ylabel("DF/F")
            plt.xlabel("Time (sec)")

        plt.suptitle(animal_id)

def plot_first_decay_time(root_dir, animal_id):

        #
    first_decay = np.load(os.path.join(root_dir,
                                animal_id,
                                animal_id+"_first_decay.npy"))
    labels = ['retrosplenial', 'barrel', 'limb', 'visual','motor']

    #
    clrs_local = ['magenta','brown','pink','darkblue', 'blue']

    #
    d1 = []
    idx = [0,1,2,3,4]  # drop visual from these plots
    for k in idx:
        #temp = (np.array(first_decay[k])+700)/30-30
        temp = (np.array(first_decay[k]))/30-30
        idx = np.where(np.isnan(temp)==False)[0]
        temp = temp[idx]

        #print (k, "Times: ", temp)
        d1.append(temp)


    plt.figure(figsize=(10,10))
    my_dict = dict(
                   retrosplenial = d1[0],
                   barrel = d1[1],
                   somatosensory = d1[2],
                   v = d1[3],
                   motor = d1[4]
               )

    data = pd.DataFrame.from_dict(my_dict, orient='index')
    data = data.transpose()

    #
    flierprops = dict(marker='o',
                      #markerfacecolor='g',
                      #markersize=10000,
                      linestyle='none',
                      markeredgecolor='r')

    #
    data.boxplot(showfliers=False,
                flierprops=flierprops,
                grid=False)

    #
    for i,d in enumerate(d1):
        y = d1[i]
        x = np.random.normal(i+1, 0.04, len(y))

        #
        x = np.random.normal(i+1, 0.04, len(d1[i]))

        #
        plt.scatter(x,
                    d1[i],
                    c=clrs_local[i],
                    #c=colors,
                    edgecolor='black',
                    s=200,
                    label=labels[i],
                    alpha=.5
                   )

    plt.ylim(-10,0)
    plt.xlim(0.5,5.5)
    plt.plot([0,6],[-3,-3],'--')
    plt.plot([0,6],[-5,-5],'--')
    plt.xticks([])

    plt.legend()



def get_data_pie_charts(n_sec_lockout,
                        body_feats_lockout,
                        animal_id,
                        root_dir):

    # skip 1,2,3 n-sec-lockouts and the empty lockouts
    if len(body_feats_lockout)==0:    # if not using any lockout window
        if n_sec_lockout in [1,2, 3]:  #
            return [], [], []


    #
    fname = os.path.join(root_dir,
                         animal_id,
                         'super_sessions',
                         'alldata_body_and_nonreward_lockout_'+str(n_sec_lockout)+'secLockout_'+
                         str(body_feats_lockout)+'bodyfeats.npz')

    #
    data = np.load(fname, allow_pickle=True)

    #
    trials = data['trials']

    #
    behaviors = data['behaviors']

    #
    names = data['names']


    return trials, names, behaviors





def compute_lockout_single_animal(root_dir,animal_ids):

    from tqdm import tqdm

    #
    n_sec_lockouts = [0,1,2,3,6,9,12,15,18,21,24,27,30]

    #
    body_feats_lockouts = [[],[0],[5]]

    #
    labels = ['all', 'left paw', 'licking']

    clrs = ['black','red','green']

    #
    fig=plt.figure()
    ctr2 = 0

    #
    for ctr_animal, animal_id in enumerate(animal_ids):
        rewarded_array = np.load(os.path.join(root_dir,
                                           animal_id,
                                           'rewarded_array_'+animal_id+'.npy'),allow_pickle=True)
        nSec_array = np.load(os.path.join(root_dir,
                                           animal_id,
                                           'nSec_array_'+animal_id+'.npy'),allow_pickle=True)

        #
        ax=plt.subplot(1,1,1)

        #
        ctr=0



        #
        for cx, body_feats_lockout in enumerate(body_feats_lockouts):

            rewarded = rewarded_array[cx]
            nSec = nSec_array[cx]

            if nSec[0]==3:
                ntemp_rew = []
                ntemp_sec = []
                for k in range(3):
                    ntemp_rew.append(rewarded[0])
                    ntemp_sec.append(k)

                ntemp_sec.append(nSec)
                ntemp_rew.append(rewarded)

                nSec = np.hstack(ntemp_sec)
                rewarded = np.hstack(ntemp_rew)

            rewarded = np.hstack(rewarded)
            #rewarded = rewarded/1351
            plt.plot(nSec,
                     rewarded,
                     c=clrs[ctr],
                     label=labels[ctr],
                    linewidth=5)

            #all_array[ctr_animal].append(rewarded)

            ctr+=1

        plt.legend()
        plt.xlim(0,20)
        plt.ylim(bottom=0.01)
        ctr2+=1

    plt.xlabel("# of seconds of lockout")
    plt.ylabel("# of trials")

def plot_all_lockouts_all_animals(root_dir):

    all_array = np.load(os.path.join(root_dir,
                                     'all_array.npy'),allow_pickle=True)

    #
    clrs = ['black','red','green']
    #
    rew_all = []
    for k in range(6):
        temp = all_array[k][0]
        #temp = temp/np.max(temp)
        rew_all.append(temp)

    rew_all = np.vstack(rew_all)
    rew_all = rew_all[:,1:]
    x = np.array([1,2,3,6,9,12,15,18,21,24,27,30])

    #
    if False:
        mean = np.mean(rew_all, axis=0)
        std = np.std(rew_all, axis=0)
        plt.plot(x,
                 mean,
                 color=clrs[0],
                 linewidth=5)

        plt.fill_between(x, mean+std, mean-std,
                         color=clrs[0], alpha=.1)
    else:
        mean = np.mean(rew_all, axis=0)
        std = np.std(rew_all, axis=0)
        plt.plot(x,
                 mean,
                 color=clrs[0],
                 linewidth=5)
        for p in range(6):
            plt.scatter(x,rew_all[p],
                       c=clrs[0],
                        edgecolor='black',
                       alpha=.8)

    # ######################################################
    x = np.array([1,2,3,6,9,12,15,18,21,24,27,30])

    rew_all = []
    for k in range(6):
        temp = np.zeros(x.shape[0])
        temp2 = all_array[k][1]
        #temp2 = temp2/np.max(temp2)
        #print (temp2.shape)
        temp[:temp2.shape[0]]=temp2

        rew_all.append(temp)

    rew_all = np.vstack(rew_all)

    #
    if False:
        mean = np.mean(rew_all, axis=0)
        std = np.std(rew_all, axis=0)
        plt.plot(x,
             mean,
                 color=clrs[1],
             linewidth=5)

        plt.fill_between(x, mean+std, mean-std,
                         color=clrs[1], alpha=.1)
    else:
        mean = np.mean(rew_all, axis=0)
        std = np.std(rew_all, axis=0)
        plt.plot(x,
             mean,
                 color=clrs[1],
             linewidth=5)
        for p in range(6):
            plt.scatter(x,rew_all[p],
                       c=clrs[1],
                        edgecolor='black',
                       alpha=.8)

    # ######################################################
    x = np.array([1,2,3,6,9,12,15,18,21,24,27,30])

    rew_all = []
    for k in range(6):
        temp = np.zeros(x.shape[0])
        temp2 = all_array[k][2]
        #temp2 = temp2/np.max(temp2)
        #print (temp2.shape)
        temp[:temp2.shape[0]]=temp2

        rew_all.append(temp)

    rew_all = np.vstack(rew_all)

    #
    mean = np.mean(rew_all, axis=0)
    std = np.std(rew_all, axis=0)
    plt.plot(x,
         mean,
             color=clrs[2],
         linewidth=5)

    for p in range(6):
        plt.scatter(x,rew_all[p],
                   c=clrs[2],
                    edgecolor='black',
                   alpha=.8)

    #
    plt.plot([3,3],[0,10000],'--',c='black')
    plt.semilogy()
    plt.fill_between(x, np.zeros(x.shape[0]), np.zeros(x.shape[0])+200,
                         color='black', alpha=.05)
    plt.xlim(0,20)

    #plt.legend()
    plt.ylim(bottom=1)
    plt.xlabel("# of seconds of lockout")
    plt.ylabel("# of trials")

#
def plot_no_of_hrs_of_recordings_vs_no_of_trials(root_dir):
    animal_ids = ['IA1','IA2','IA3','IJ1','IJ2','AQ2']
    animal_names = ['M1','M2','M3','M4','M5','M6']

    hrs_array = np.load(os.path.join(root_dir,
                                     'hrs_array.npy'),allow_pickle=True)
    rew_array = np.load(os.path.join(root_dir,
                                     'rew_array.npy'),allow_pickle=True)

    #
    plt.figure()
    ax1=plt.subplot(111)
    ax2 = ax1.twinx()
    for ctr, animal_id in enumerate(animal_ids):
       # n_hrs, n_rew = get_no_of_hours(root_dir,animal_id)

        n_hrs = hrs_array[ctr]
        n_rew = rew_array[ctr]

        n_hrs = n_hrs*22/60.

        ax1.scatter(ctr, n_hrs, c='black',
                    s=200,
                    label=animal_id)

        ax2.scatter(ctr, n_rew, c='blue',
                    s=200,
                    label=animal_id)


    #
    ax1.set_ylabel("# of hours of recording")
    ax2.set_ylabel("# of rewarded pulls")
    plt.xticks(np.arange(6), animal_names)
    ax1.set_ylim(0,45)
    ax2.set_ylim(0,7500)

def plot_pie_charts_behaviors(root_dir, animal_id):
    all_rewarded = []
    plotting=True

    # trials = np.load(os.path.join(root_dir, animal_id,
    #                               animal_id+'_trials.npy'),allow_pickle=True)
    # names = np.load(os.path.join(root_dir, animal_id,
    #                               animal_id+'_trials_names.npy'),allow_pickle=True)
    behaviors = np.load(os.path.join(root_dir, animal_id,
                                  animal_id+'_behaviors.npy'),allow_pickle=True)


    # get # rewarded pulls:
    n_rew = 0
    n_nonrew = 0
    n_left_paw = 0
    n_right_paw = 0
    n_licking = 0

    #
    feat_rew = 0
    feat_nonrew = 1
    feat_left_paw = 2
    feat_right_paw = 3
    feat_licking = 7

    n_sess = 0
    min_quiescence = 30 # in frame times
    for sess in trange(len(behaviors)):
        try:
            if len(behaviors[sess][feat_left_paw])>0:
                n_rew+= behaviors[sess][feat_rew].shape[0]
                n_nonrew+= behaviors[sess][feat_nonrew].shape[0]

                # left paw
                temp = behaviors[sess][feat_left_paw]
                diff = temp[1:]-temp[:-1]
                #print (diff)
                idx = np.where(diff>=min_quiescence)[0]
                n_left_paw+= idx.shape[0]

                # left paw
                temp = behaviors[sess][feat_right_paw]
                diff = temp[1:]-temp[:-1]
                idx = np.where(diff>=min_quiescence)[0]
                n_right_paw+= idx.shape[0]

                # left paw
                temp = behaviors[sess][feat_licking]
                diff = temp[1:]-temp[:-1]
                idx = np.where(diff>=min_quiescence)[0]
                n_licking+= idx.shape[0]
            n_sess+=1

        except:
            pass

    sizes = np.array([n_rew, n_nonrew, n_left_paw, n_right_paw, n_licking])
    sizes = sizes/np.sum(sizes)*100

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = ['rewarded lever pulls','non-rewarded lever pulls','left paw','right paw', 'licking']

    #
    all_rewarded.append(sizes[0])

    #
    if plotting:

        fig1, ax1 = plt.subplots()
        colors = ['blue','lightgrey','wheat','paleturquoise','pink']
        patches, texts,_ = ax1.pie(sizes,
                                #explode=explode,
                                #labels=labels,
                                colors=colors,
                                autopct='%1.1f%%',
                                #shadow=True,
                                pctdistance=1.1,
                                labeldistance=1.2,
                                startangle=180,
                                textprops={'fontsize': 20}
                               )

        #patches[0][0].set_alpha(.1)

        labels = [f'{l}, {s:0.1f}%' for l, s in zip(labels, sizes)]

        ax1.legend(patches, labels, loc="best")

        # Set aspect ratio to be equal so that pie is drawn as a circle.
        ax1.axis('equal')
        plt.tight_layout()
        plt.title(animal_id)
        #

#
def plot_all_animal_lever_pulls_percentages(root_dir, animal_ids):
    animal_names = ['M1','M2','M3','M4','M5','M6']

    all_rewarded = np.load(os.path.join(root_dir,
                                        'all_rewarded_percentages.npy'),allow_pickle=True)
    #
    fig=plt.figure()
    ax1=plt.subplot(111)
    for ctr, animal_id in enumerate(animal_ids):
        n_hrs = all_rewarded[ctr]
        ax1.scatter(ctr, n_hrs, c='blue',
                    s=200,
                    label=animal_id)

    #
    plt.plot([0,6],[np.mean(all_rewarded),np.mean(all_rewarded)],'--')
    ax1.set_ylabel("% behaviors == lever pulls")
    plt.xticks(np.arange(6), animal_names)
    ax1.set_ylim(0,10)


def plot_svm_accuracy_with_lockouts(root_dir):

    import pandas as pd
    from scipy import stats
    import matplotlib.patches as mpatches


    #
    def box_plots(edts,
                 licking_flag=False):

        pvals = [0.05,0.01,0.001,0.0001,0.00001]

        clrs_local = ['magenta','brown','pink','lightblue','darkblue', 'blue']

        #
        if licking_flag==False:
            my_dict = dict(
                       #three = edts[0][1],
                       #six = edts[1][1],
                       #nine = edts[2][1],
                       twelve = edts[3][1],
                       #fifteen = edts[4][1],
                      #All = edts[5]+15,
                     )
        else:
            my_dict = dict(one = edts[0][1],
                   two = edts[1][1],
                   three = edts[2][1],
                   six = edts[3][1],
                   #nine = edts[4][1],
                   #twelve = edts[3][1],
                   #fifteen = edts[4][1],
                  #All = edts[5]+15,
                 )

        cmap = plt.cm.get_cmap('Reds', len(my_dict))    # 11 discrete colors


        #
        data = pd.DataFrame.from_dict(my_dict, orient='index')
        data = data.transpose()

        #print ("DATA: ", data.shape)
        #print ("means: ", np.mean(data,0))
        means =np.mean(data,0)
        #########################################################
        ################## SCATTER PLOTS ########################
        #########################################################
        for i,d in enumerate(data):
            y = data[d]
            x = np.random.normal(i+1, 0.04, len(y))
            #print (x)
            #print (y)
            if licking_flag==False:
                plt.scatter(x,y,
                       c=clrs_local[i],
                       #c=cmap(i),
                        s=100,
                       edgecolor='black')
            else:
                plt.scatter(x,y,
                       #c=clrs_local[i],
                       c=cmap(i),
                        s=100,
                       edgecolor='black')

        #########################################################
        ###################### BOX PLOTS ########################
        #########################################################
        flierprops = dict(#marker='o',
                          #markerfacecolor='g',
                          markersize=10000,
                          linestyle='none',
                          markeredgecolor='r')

        #
        data.boxplot(showfliers=False,
                     flierprops=flierprops,
                     grid=False)



        #
        plt.xlim(0.5, 6.5)
        plt.ylim(-15,0)
        plt.xticks([])
        #plt.yticks([])

        #
        plt.plot([0,6.5], [-3,-3],'--',linewidth=3,c='grey',alpha=.5)
        plt.plot([0,6.5], [-5,-5],'--',linewidth=3,c='grey',alpha=.5)
        plt.plot([0,6.5],[-10,-10],'--',linewidth=3,c='grey',alpha=.5)

        # ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

        patches = []
        for p in range(1,len(my_dict),1):
            res = stats.ks_2samp(edts[p][1], edts[0][1])
            #print ("res: ", res)
            label_ = ''
            for k in range(len(pvals)):
                if res[1]<pvals[k]:
                    label_ = label_ + "*"
                else:
                    break

            patches.append(mpatches.Patch(label=label_))

        #plt.legend(handles=patches,fontsize=12)

        return means

    ############################
    animal_ids = ['IA1','IA2','IA3','IJ1','IJ2','AQ2']
    biorxiv_names = ['M1','M2','M3','M4','M5','M6']

    licking_flag = False
    means = []
    plt.figure(figsize=(20,5))
    for ctr,animal_id in enumerate(animal_ids):
        ax=plt.subplot(1,6,ctr+1)

        if ctr==0:
            plt.ylabel("EDTs with lever pull lockouts (sec)")
        if licking_flag==False:
            edts = np.load(os.path.join(root_dir,
                                        animal_id,
                                    animal_id +
                                    '_edts_lockedout.npy'), allow_pickle=True)
        else:
            edts = np.load(os.path.join(root_dir,animal_id,
                                    animal_id +
                                    '_edts_lockedout_licking.npy'), allow_pickle=True)


        #
        plt.title(biorxiv_names[ctr])
        m_ = box_plots(edts, licking_flag)
        means.append(m_)

    #np.save('/home/cat/lever_lockout_means.npy',means)

def plot_trends_lever_lockouts(root_dir):
    biorxiv_names = ['M1','M2','M3','M4','M5','M6']

    means = np.load(os.path.join(root_dir,'lever_lockout_means.npy'),allow_pickle=True)
    fig=plt.figure()
    t = np.array([3,6,9,12,15])
    for k in range(6):
        mm = means[k]#.to_numpy()
        #print (mm)
        plt.plot(t, mm,
                 label=biorxiv_names[k],
                 linewidth=5)
        plt.scatter(t, mm,
                   s=100)

    plt.xlabel("# sec leverlpull lockout")
    plt.ylabel("EDTs (sec)")
    plt.plot([3,15],[-3,-3],'--',c='grey')
    plt.plot([3,15],[-5,-5],'--',c='grey')

    plt.ylim(-6.5,0)
    plt.legend()

#
def plot_area_averages(main_dir, animal_id, session_ids):
    # FIG 6A example of visual vs. limb averages
    #animal_id = 'IJ2'
    #session_ids = ['Feb1_', 'Mar2_','Mar31_','Apr4_']
    #session_ids = ['Feb3_','Mar3_'] #,'Apr4_']

    colors = plt.cm.viridis(np.linspace(0,1,3))
    #colors=['black','blue','red','magenta']
    #
    plt.figure(figsize=(20,5))
    area_ids = [13, 8, 12, 11]
    linewidth = 3
    plot_roi_averages(main_dir,
                         animal_id,
                         session_ids,
                         colors,
                         area_ids,
                         linewidth)



def load_raw_data(spatial_fname, temporal_fname):
    # GRAB AND RECONSTRUCT DATA
    spatial = np.load(spatial_fname)
    temporal = np.load(temporal_fname)
    temporal = temporal.transpose(0,2,1)

    #
    print (spatial.shape)
    print (temporal.shape)

    #
    print ("reconstructing data: ")
    data = np.matmul( temporal, spatial)
    print (data.shape)

    #
    print ("getting mean of data: ")
    data_mean = data.mean(0)
    print ("data_mean: ", data_mean.shape)
    # compute variance in raw data- not used
    # var = np.var(data2d, axis=0)
    # print ("var: ", var.shape)

    ######################################
    ###### COMPUTE RAW ROI ACTIVITY ######
    ######################################
    data2D = data_mean.reshape(data_mean.shape[0], 128,128)
    print ("Data mean 2D: ", data2D.shape)

    #
    means = []
    ctr=0
    for id_ in ordered_names:
        area_id = areas[id_]
        idx = np.where(atlas==area_id)
        print (ctr, "areaId: ", area_id, names[id_], idx[0].shape)
        mask = np.zeros((128,128),'float32') + np.nan
        mask[idx] = 1

        temp = data2D*mask
        roi = np.nanmean(temp, axis=1)
        roi = np.nanmean(roi, axis=1)
        means.append(roi)

        ctr+=1

    #
    raw_means = np.array(means)
    print ("Raw data means: ", raw_means.shape)

    return raw_means


def load_locaNMF_data(fname_locaNMF):
    # order locaNMF components by plot color ORDER in Fig 4A
    ordered_names = np.array([15,0,14,1,   # retrosplenial areas
                              13,2,
                              12,3,
                              11,4,
                              10,5,
                              9,6,
                              8,7])[::-1]


    # load raw data
    try:
        d = np.load(fname_locaNMF)
    except:
        print ("file missing", fname_locaNMF)
        return None, None, None, None, None

    locaNMF_temporal = d['temporal_trial']
    locaNMF_temporal_random = d['temporal_random']
    locaNMF_temporal = locaNMF_temporal[:,ordered_names]
    locaNMF_temporal_random = locaNMF_temporal_random[:,ordered_names]
    #print ("locanmf data: ", locaNMF_temporal.shape)

    #
    areas = d['areas'][ordered_names]
    names = d['names'][ordered_names]
    #print ("original names: ", names.shape)

    #
    atlas = np.load('/media/cat/4TBSSD/yuki/yongxu/atlas_split.npy')
    #print ("atlas: ",atlas.shape)
    # print (areas)
    # print (names)

    print ("  # trials ", locaNMF_temporal.shape)
    #print ("ORDERED NAMES: ", names[ordered_names])


    return atlas, areas, names, locaNMF_temporal, locaNMF_temporal_random


def plot_locanmf_vs_raw(locaNMF_temporal, raw_means):
# FIg 2B locaNMF vs. raw

    locaNMF_temporal_means_clip = np.mean(locaNMF_temporal,axis=0)[:,:900]
    print ('locaNMF_temporal_means: ',
            locaNMF_temporal_means_clip.shape)

    #
    raw_means_clip = raw_means[:,:900]
    print ('raw temporal_means: ', raw_means.shape)

    #
    colors = plt.cm.jet(np.linspace(0,1,raw_means.shape[0]))
    scale1 = 1
    scale2 = scale4 = .075
    scale3 = 1

    #
    t = np.arange(raw_means_clip.shape[1])/30-30
    fig=plt.figure(figsize=(10,10))
    linewidth=3
    for k in range(raw_means.shape[0]):
        ax1=plt.subplot(121)
        plt.xticks([])
        plt.yticks([])
        plt.xlim(-15,0)
        plt.ylim(-.1,1.25)

        temp1 = raw_means_clip[k]#/np.max(raw_means[k])
        if k==0:
            plt.plot(t,temp1*scale1+k*scale2,c=colors[k],
                     linewidth= linewidth,
                     label = 'raw')
        else:
            plt.plot(t,temp1*scale1+k*scale2,c=colors[k],
                     linewidth= linewidth)

        #plt.plot([-15,0], [scale3+k*scale2, scale3+k*scale2],'--',c='black',alpha=.2)


        # locanmf traces
        temp2 = locaNMF_temporal_means_clip[k]#/np.max(locaNMF_temporal_means[k])

        if k ==0:
            plt.plot(t,temp2*scale3+k*scale2,'--',
                     linewidth= linewidth,
                     c=colors[k],
                    label='locaNMF')
        else:
            plt.plot(t,temp2*scale3+k*scale2,'--',
                     linewidth= linewidth,
                     c=colors[k])

        plt.plot([-15,0], [k*scale2,k*scale2],'--',c='black',alpha=.2)

    #     #
    #     ax2=plt.subplot(122)
    #     plt.xlim(-15,0)
    #     temp3 = temp2*scale3-temp1*scale1
    #     plt.plot(t, temp3*scale3+k*scale4,'--', c=colors[k])

    ax1.legend()
    if False:
        plt.savefig('/home/cat/fano.png',dpi=300)
        plt.close()
    else:
        plt.show()






#
def variance_locaNMF(locaNMF_temporal):
    #
    t = np.arange(locaNMF_temporal.shape[2])/30 - 30
    means = []
    var = []
    #for k in ordered_names:
    for k in range(locaNMF_temporal.shape[1]):
        temp = locaNMF_temporal[:,k].mean(0)
        means.append(temp)

        #
        temp = np.var(locaNMF_temporal[:,k],axis=0)
        var.append(temp)

    #
    means = np.array(means)#[:,:900]
    var = np.array(var)#[:,:900]
    #print (means.shape, var.shape)

    return means, var



def plot_longitudinal_roi_loca(n_trials, saved_names, all_means):
    print ('n trials: ', n_trials)
    colors = plt.cm.viridis(np.linspace(0,1,len(all_means)))
    area_ids = [0,1,6,7,8,9,12,13]
    time= np.arange(all_means[0].shape[1])/30-30.

    #
    ctr=1
    min_trials = 10
    aucs = []
    saved = []
    fig=plt.figure(figsize=(10,6))
    for ctr, area_id in enumerate(area_ids):
        ax = plt.subplot(2,4,ctr+1)

        #
        aucs.append([])
        counter = 0
        for t in range(len(all_means)):
            temp = all_means[t][area_id]
            if n_trials[t]<min_trials:
                continue

            if np.max(np.abs(temp))<0.2:
                plt.plot(time, temp,
                         color=colors[t],
                        alpha=.8)

                auc = np.nansum(np.abs(temp), axis=0)

                aucs[ctr].append([t,auc])
                counter+=1

        print (ctr, 'area_id: ', area_id, counter)
        #
        #if ctr==5:

        plt.xticks([])
        plt.yticks([])
        plt.xlim(-15,0)
        plt.title(saved_names[area_id],fontsize=8)

        # cmap = matplotlib.cm.viridis
        #norm = matplotlib.colors.Normalize(vmin=5, vmax=10)

        # cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
         #                               orientation='vertical')

    #
    if False:
        plt.savefig('/home/cat/'+str(animal_id)+'_loca_longitudinal.png',dpi=300)
        plt.close()
    else:
        plt.show()


def plot_locanmf_temporal_averages(fig,
                                   locaNMF_temporal,
                                  clr):
    locaNMF_temporal_means_clip = np.mean(locaNMF_temporal,axis=0)[:,:900]
    print ('locaNMF_temporal_means: ',
            locaNMF_temporal_means_clip.shape)

    #
    colors = plt.cm.jet(np.linspace(0,1,locaNMF_temporal_means_clip.shape[0]))
    scale1 = 1
    scale2 = scale4 = .075
    scale3 = 1

    #
    t = np.arange(locaNMF_temporal_means_clip.shape[1])/30-30
    linewidth=5
    for k in range(locaNMF_temporal_means_clip.shape[0]):
        #ax1=plt.subplot(121)
        #plt.xticks([])
        #plt.yticks([])
        plt.xlim(-30,0)
        #plt.ylim(-.1,1.25)

        # locanmf traces
        temp2 = locaNMF_temporal_means_clip[k]#/np.max(locaNMF_temporal_means[k])

        if k ==0:
            plt.plot(t,temp2*scale3+k*scale2,
                     linewidth= linewidth,
                     c=clr)
        else:
            plt.plot(t,temp2*scale3+k*scale2,
                     linewidth= linewidth,
                     c=clr)

        plt.plot([-30,0], [k*scale2,k*scale2],'--',c='black',alpha=.2)



##################
def plot_variance_locaNMF(fig, var,
                         clr):

    scale1 = 1
    scale2 = .004
    linewidth=5
    # scale3 = np.nan
    t = np.arange(var.shape[1])/30-30


    #
    for k in range(var.shape[0]):
        #
        plt.xticks([])
        #plt.yticks([])

        #
        temp = var[k]
        temp = temp*scale1+k*scale2

        # plot the variance
        plt.plot(t, temp, c=clr,
                linewidth=linewidth)


        #
        #
        print ("temp var: ", temp.shape)
        idx = np.argmin(temp[750:900])
        print ("idx: ", idx)
        print (t[idx+750], temp[idx+750])
        plt.scatter(t[idx+750], temp[idx+750],
                    s=1000,
                    color=clr,
                   alpha=.8)

        # plot lines on top of plot
        plt.plot([-30,0], [k*scale2,k*scale2],'--',c='black',alpha=.5)

        #
        mean2 = np.mean(temp.squeeze())
        plt.plot([-30,0], [mean2,mean2],'--',c=clr,alpha=.5)

    #
    plt.xlim(-30,0)


def load_locaNMF_temporal(animal_id, session_name, root_dir,
                         session_id):

    loca = analysis_fig4.LocaNMFClass(root_dir, animal_id, session_name)

    #
    loca.get_sessions(session_name)
    print ("sessions: ", loca.sessions.shape)
    print ("selected session: ", loca.sessions[session_id])

    session = loca.sessions[session_id]

    # load data
    fname_locaNMF = os.path.join(root_dir, animal_id, 'tif_files',session,
                                 session + '_locanmf.npz')


    atlas, areas, names, locaNMF_temporal, locaNMF_temporal_random = load_locaNMF_data(fname_locaNMF)

    return atlas, areas, names, locaNMF_temporal, locaNMF_temporal_random


def plot_roi_averages(main_dir,
                     animal_id,
                     session_ids,
                     colors,
                     area_ids,
                     linewidth):
    ctr_session=0

    for session_id in session_ids:
        sessions = get_sessions(main_dir,
                                         animal_id,
                                         session_id)

        loca = np.load(os.path.join(main_dir, animal_id, 'tif_files',sessions[0],
                                    sessions[0]+'_locanmf.npz'))

        trials = loca['temporal_trial']
        random = loca['temporal_random']
        names = loca['names']

        #
        ctr=0
        t=np.arange(900)/30.-30
        for area_id in area_ids:
            plt.subplot(1,4,ctr+1)
            plt.title(names[area_id])

            plt.plot(t,trials[:43,area_id,:900].mean(0),
                      linewidth=linewidth,
                      color=colors[ctr_session],
                     label='session average')

            #
            ran = random[:43,area_id,:900]


            for p in range(ran.shape[0]):
                idx = np.random.choice(np.arange(-300,300,1))
                ran[p] = np.roll(ran[p], idx)

            plt.plot(t,ran.mean(0),'--',
                     linewidth=linewidth,
                     color=colors[ctr_session],
                     label='random average')


            plt.ylim(-0.10, 0.10)
            plt.xlim(-15,t[-1])
            plt.plot([-30,0],[0,0],'--',c='grey')

            if ctr==0:
                plt.legend()
                plt.ylabel("DFF")
            plt.xlabel("Time (sec)")
            ctr+=1
        ctr_session+=1

def plot_spectra_longintudinally(colors, plotting, fs, selected_areas,
                                all_areas,
                                all_means,
                                all_means_random):
    from scipy import signal
    from scipy.signal import argrelmax

    plt.figure(figsize=(20,5))

    #
    norm = False
    ctr2=0
    averages = []
    for k in range(len(selected_areas)):
        averages.append([])

    spectra = np.zeros((len(all_means),len(selected_areas),451))
    # loop over all sessions
    for k in range(len(all_means)):

        # get all traces within a session
        all_traces = all_means[k]

        #
        all_traces_random = all_means_random[k]

        temp_list = []
        temp_list_random = []
        pxx_list = []
        # loop over all areas selected
        for ctr_area, selected_area in enumerate(selected_areas):

            #########################################
            ######### PLOT INDIVIDUAL SPECGRAMS #####
            #########################################
            temp = all_traces[selected_area]
            temp_random = all_traces_random[selected_area]
            #temp = savgol_filter(temp, 7, 1)
            temp_list.append(temp)
            temp_list_random.append(temp_random)
            if True: #np.max(np.abs(temp))<0.2:
                temp = np.float64(temp)
                temp_random = np.float64(temp_random)
                f, Pxx_den = signal.periodogram(temp, fs)
                f, Pxx_den_random = signal.periodogram(temp_random, fs)

            if norm:
                Pxx_den = Pxx_den/Pxx_den_random

            # SHOW ONLY FIRST 451 datapoints
            x = np.arange(0,451,1)/30.
            Pxx_den = Pxx_den[:451]

            if plotting:
                ax = plt.subplot(1,len(selected_areas),ctr_area+1)
                if norm == False:
                    plt.ylim(1E-8,1E-1)
                else:
                    plt.ylim(1E-3,1E6)

                plt.xlabel('frequency [Hz]')
                plt.xlim(2E-2,6)
                #ax.set_yticks([])
                plt.semilogy()
                plt.semilogx()

                plt.plot(x,
                     Pxx_den,
                     linewidth=3,
                     c= colors[ctr_area][ctr2],
                        alpha=.5)
            #
            xx = argrelmax(Pxx_den, order=100)

            pxx_list.append(Pxx_den)
            averages[ctr_area].append(Pxx_den)

            spectra[k,ctr_area]=Pxx_den

        ctr2+=1

    #########################################
    ######### PLOT AVERAGES #################
    #########################################
    for ctr_area in range(len(selected_areas)):
        y = np.median(averages[ctr_area],axis=0)
        argmax = np.argmax(y)

        if plotting:
            ax = plt.subplot(1,len(selected_areas),ctr_area+1)

            plt.plot(x,
                 y,
                 linewidth=3,
                 c= 'black')

            plt.plot([x[argmax], x[argmax]],[1E-8,1E5], '--',c='red')
            #print ("max freq: ", x[argmax])
            #
            plt.plot([0.1, 0.1],[1E-8,1E5], '--', c='black',alpha=.5)
            plt.plot([1, 1],[1E-8,1E5], '--',c='black',alpha=.5)

            for k in range(-8,-1,1):
                plt.plot([1E-4,20],[10**k,10**k], '--',c='black',alpha=.5)
        plt.title("ROI: "+ str(selected_areas[ctr_area]))


    return spectra


def plot_box_plots(peaks):

    #
    codes = ['Retrosplenial', 'barrel', 'limb', 'visual','motor']
    #codes = ['limb, layer 1 - right', 'limb, layer 1 - left']
    clrs_local = ['black','blue','red','magenta', 'pink','brown']

    #
    #bin_width = 0.01
    #bins = np.arange(0.1,2.0,bin_width)

    #
    edts = []
    for a in range(len(peaks)):
        good_vals = np.hstack(peaks[a])
        edts.append(good_vals)
#     #
    roi_names = ['V1-L',
     'SomF-L',
     'M1-L',
     'RD-L']
    my_dict = dict(V1L = edts[0],
                   SomFL = edts[1],
                   M1L = edts[2],
                   RDL = edts[3],
#                    M5 = edts[4],
#                    M6 = edts[5]
                     )

    data = pd.DataFrame.from_dict(my_dict, orient='index')
    data = data.transpose()

    #
    flierprops = dict(marker='o',
                      #markerfacecolor='g',
                      #markersize=10000,
                      linestyle='none',
                      markeredgecolor='r')

    #
    data.boxplot(showfliers=False,
                flierprops=flierprops)

    # manually define a new patch
    labels = []
    for i,d in enumerate(data):
        y = data[d]#+np.random.uniform(data[d].shape[0])/200.-1/400.
        x = np.random.normal(i+1, 0.04, len(y))

        #
        colors = plt.cm.viridis(np.linspace(0,1,len(edts[i])))
        x = np.random.normal(i+1, 0.04, len(edts[i]))
        print (i,d, ' y shape: ', y.shape)
        plt.scatter(x, edts[i],
                   #c=clrs_local[i],
                   c=colors,
                   edgecolor='black',
                   s=200,
                   #alpha=np.linspace(.2, 1.0, x.shape[0])
                   alpha=.5
                   )

        # compute correlation between time and location
        from scipy import stats
        # Y and Z are numpy arrays or lists of variables
        #print (np.arange(edts[i]).shape, y.shape)
        corr = stats.pearsonr(np.arange(y.shape[0]), y)
        #res = scipy.stats.normaltest(edts[i])
        print (i, "corr", corr)

        #
        patch = mpatches.Patch(color='grey', label=roi_names[i]+ ":  "+str(round(corr[0],2)))

        # handles is a list, so append manual patch
        labels.append(patch)

        # plot the legend
    plt.legend(handles=labels, loc='upper center')




    plt.xticks([])
    plt.yticks([])

    plt.semilogy()
    #plt.ylim(1E-1,1E1)

#
def plot_power_spectra_multi_area(root_dir, animal_id):

    all_means = np.load(os.path.join(root_dir,
                                     animal_id,
                                     'all_means_'+animal_id+'.npy'))
    all_means_random = np.load(os.path.join(root_dir,
                                     animal_id,
                                     'all_means_random_'+animal_id+'.npy'))

    #
    colors = []
    for p in range(4):
        colors.append(plt.cm.viridis(np.linspace(0,1,len(all_means))))

    #
    fs = 30
    plotting=True
    selected_areas = [11,9,1,15]
    all_areas = np.arange(all_means[0].shape[0])

    #
    spectra = plot_spectra_longintudinally(colors,
                                           plotting,
                                           fs,
                                           selected_areas,
                                           all_areas,
                                           all_means,
                                           all_means_random)

#
def plot_peak_frequency_all_animals_all_sessions(root_dir, plot_power, plot_freq):
    from sklearn.linear_model import LinearRegression
    from scipy import stats

    all_power = np.load(os.path.join(root_dir, 'all_power.npy'),allow_pickle=True)
    all_peaks = np.load(os.path.join(root_dir, 'all_peaks.npy'),allow_pickle=True)


    #
    ids = [0,2,3]
    clrs_local = ['magenta','brown','pink','lightblue','darkblue']

    feats_ = ['visual','limb','motor','Retro']
    clrs_local_ = ['brown','lightblue','magenta','pink']
    feat_names = ['retrosplenial','visual','limb','motor']
    #ids_ = np.array([1,2,3])
    #
    animal_ids = ['IA1','IA2','IA3','IJ1','IJ2','AQ2']
    #animal_ids = ['IA1']#,'IA2','IA3','IJ1','IJ2','AQ2']
    biorxiv_names = ["M1", "M2", "M3", "M4","M5",'M6']

    #
    fig=plt.figure(figsize=(10,10))
    for ctr_animal, animal_id in enumerate(animal_ids):
        if plot_power:
            maxes = all_power[ctr_animal]
        elif plot_freq:
            maxes = all_peaks[ctr_animal]

        ax=plt.subplot(3,2,ctr_animal+1)

        ctr=0
        for id_ in ids:
            m = np.array(maxes[id_])

            ##################################
            ##################################
            ##################################
            corr = stats.pearsonr(np.arange(m.shape[0]), m)
            print (animal_id, id_, "corr: ", corr)

            model = LinearRegression()
            y=np.array(m).reshape(-1, 1)
            x = np.arange(y.shape[0]).reshape(-1, 1)
            model.fit(x, y)

            x2 = np.arange(0,y.shape[0],1).reshape(-1, 1)
            y_pred = model.intercept_ + model.coef_ * x2


            # compute correlation between time and location
            if corr[1]>0.05:
                plt.plot(x2, y_pred,
                     #label= str(np.round(corr,2)),
                      c=clrs_local[id_],
                      #label=feat_names[id_],
                     linewidth=6)
            else:
                plt.plot(x2, y_pred,
                     #label= str(np.round(corr,2)),
                      c=clrs_local[id_],
                      #label=feat_names[id_]+"pcor: "+str(round(corr[0],5))+
                      label="pcor: "+str(round(corr[0],5))+
                     "\npval: "+str(round(corr[1],5)),
                     linewidth=6)

            ##################################
            ##################################
            ##################################
            #
            plt.scatter(np.arange(m.shape[0]), m,
                       c=clrs_local[id_],
                        edgecolor='black',
                        s=100,
                       alpha=.3)

            ctr+=1
        #if ctr_animal==0:
        plt.legend()
        if ctr_animal>5:
            plt.xlabel("Session ID")
        plt.ylabel("Freq (hz)")

        plt.title(biorxiv_names[ctr_animal])
        print ('')
