import matplotlib

# 
import numpy as np
import matplotlib.pyplot as plt
import os


#########################################
########### HELPER FUNCTIONS ############
#########################################
def load_data_PCA(fname_whole_stack,
                 fname_lever_times,
                 fname_blue_light):

    '''
    Function takes in names of lever triggers, blue light and whole stack and
    returns the continuos as well as triggered segments

    *** NOTE: this function has been superseeded by the next one that loads ROI based data from
    whole stack locaNMF


    '''

    # this is the entire session PCA stack
    d1 = np.load(fname_whole_stack)
    
    # zero out beginning and end, usually they are screwed up by excitation light
    d1[:200]=d1[200]
    d1[-200:]=d1[-200]

    # these are the times in seconds for each of the rewarded lever pulls
    imaging_rate = 30
    times = np.loadtxt(fname_lever_times)
    times = np.int32(times*imaging_rate)
    #print ("trigger times in frame times: ", times)
    
    # this is the calcium excitation light, we need to shift the dataset to match it
    blue = np.load(fname_blue_light)
    start = blue['start_blue']
    d1 = np.roll(d1,-start,axis=0)
    
    # make a time triggered stack to visualize there is indeed a peak in the correct place
    stack = []
    for t in times:
        temp = d1[t-450:t+450]
        if len(temp)==900:
            stack.append(temp)

    
    stack = np.array(stack)

    return stack, d1, times, start


###################################################
def load_data_ROIs_lever_pulls(fname_whole_stack,
                           fname_lever_times,
                           fname_blue_light):

    '''
    Function takes as input the whole stack locaNMF data, lever trigger file names and blue-light shift

    Exports:
        stack: trigger centred segments of locaNMF data
        stack_random: same as above but for random sampled trigger times
        continuous:  locaNMF processed data shifted to match blue light shift
        times: the trigger times;
        times_nonrewarded: the
    '''


    # this is the entire session PCA stack
    dall = np.load(fname_whole_stack)
    
    d1 = dall['whole_stack'].T
    names = dall['names']
    
    # zero out beginning and end, usually they are screwed up by excitation light
    d1[:200]=d1[200]
    d1[-200:]=d1[-200]

    # these are the times in seconds for each of the rewarded lever pulls
    imaging_rate = 30
    times = np.loadtxt(fname_lever_times)
    times = np.int32(times*imaging_rate)
    
    nonrewarded_times = np.loadtxt(fname_lever_times.replace('rewarded','nonrewarded'))
    nonrewarded_times = np.int32(nonrewarded_times*imaging_rate)

    #print ("trigger times in frame times: ", times)
    
    # this is the calcium excitation light, we need to shift the dataset to match it
    if True:
        blue = np.load(fname_blue_light)
        start = blue['start_blue']
        d1 = np.roll(d1,-start,axis=0)
    
    # make a time triggered stack to visualize there is indeed a peak in the correct place
    stack = []
    stack_random = []
    min_lockout = 30*3  # min 3 seconds away from each trigger
    for t in times:
        temp = d1[t-900:t+900]
        if len(temp)==1800:
            stack.append(temp)

            # grab random segments also at the same time;
            while True:
                t_random = np.random.randint(1000,38000)
                min_diff = np.min(np.abs(times-t_random))
                if min_diff>min_lockout:

                    temp2 = d1[t_random-900:t_random+900]
                    if len(temp2)==1800:
                        stack_random.append(temp2)
                    break
    #
    stack = np.array(stack)
    stack_random = np.array(stack_random)

    #    
    return stack, stack_random, d1, times, nonrewarded_times, names


################################################################
################################################################
################################################################
def load_data_ROIs_body_movements(root_dir,
                                  animal_id,
                                  session,
                                  fname_whole_stack,
                                  fname_blue_light):

    '''
    Function takes as input the whole stack locaNMF data, body movement trigger file name and blue-light shift

    Exports:
        stack: trigger centred segments of locaNMF data
        stack_random: same as above but for random sampled trigger times
        continuous:  locaNMF processed data shifted to match blue light shift
        times: the trigger times;
    '''


    ############################################
    ##### WHOLE STACK LOCANMF ##################
    ############################################
    dall = np.load(fname_whole_stack)

    #
    d1 = dall['whole_stack'].T

    # zero out beginning and end, usually they are screwed up by excitation light
    d1[:900]=d1[900]
    d1[-900:]=d1[-900]

    # this is the calcium excitation light, we need to shift the dataset to match it
    if True:
        blue = np.load(fname_blue_light)
        start = blue['start_blue']
        d1 = np.roll(d1,-start,axis=0)

    ############################################
    ##### BODY MOVEMENT TRIGGERS ###############
    ############################################
    fname_ = os.path.join(root_dir,
                                animal_id,
                                'tif_files',
                                session,
                                session+'_0secNoMove_movements.npz')

    data = np.load(fname_, allow_pickle=True)

    #
    names = data['labels']

    # load video shift peaks from left paw, right paw and lever DLC traces
    shift = data['video_shift']
    print (shift)

    # load binarized body initiation matrices
    inits = data['feature_initiations']
    print (inits.shape)
    print ("INITS: ", inits)

    #
    times = []
    for k in range(inits.shape[0]):
        idx = np.where(inits[k]==1)[0]
        print (k, idx.shape)
        print (idx)
        idx = idx*2
        print (idx)
        times.append(idx)

    ############################################
    ######### FIGURE OUT SHIFT INDEX ###########
    ############################################
    # load the manual shift id from file to select one of the shifts from above
    good_alignments = np.loadtxt(os.path.join(root_dir,
                                             animal_id,
                                             'tif_files',
                                             'sessions_DLC_alignment_good.txt'),
                                 dtype='str')
    # print ("good alignmnets", good_alignments)
    for k in range(len(good_alignments)):
        temp = good_alignments[k][0].replace(',','')
        if temp in session:
            shift_id = int(good_alignments[k][1])
            shift = shift[shift_id]
            print ("Ideal DLC shift: ", shift )
            break


    ############################################
    ##### BODY MOVEMENT TRIGGERS ###############
    ############################################
    # make a time triggered stack to visualize there is indeed a peak in the correct place
    stack = []
    stack_random = []
    min_lockout = 30*3  # min 3 seconds away from each trigger

    from tqdm import trange, tqdm
    # loop over all body parts
    window_width = 900
    
    for f in range(len(times)):
        stack.append([])
        stack_random.append([])

        # loop over each time point
        remove_ids = []  # keep track of which times cannot generate a proper ca stack
        stack_temp = []
        stack_random_temp = []
        ctr2=0
        for t in tqdm(times[f]):
            temp = d1[t-window_width:t+window_width]
            #print ("temp: ", temp.shape, t-window_width,"..",t+window_width)
            if len(temp)==window_width*2:
                stack_temp.append(temp)

                # grab random segments also at the same time;
                while True:
                    t_random = np.random.randint(window_width,40000-window_width)
                    min_diff = np.min(np.abs(times[f]-t_random))
                    if min_diff>min_lockout:
                        temp2 = d1[t_random-window_width:t_random+window_width]
                        #print ("temp2: ", temp2.shape)
                        if len(temp2)==window_width*2:
                            #stack_random[f].append(temp2)
                            stack_random_temp.append(temp2)
                        break
            else:
                remove_ids.append(ctr2)
            ctr2+=1

        stack[f]= np.array(stack_temp)
        stack_random[f] = np.array(stack_random_temp)

        # delete all the times where calcium data couldn't be generated
        if len(remove_ids)>0:
            remove_ids = np.array(remove_ids)
            tt = np.array(times[f])
            tt = np.delete(tt, remove_ids)
            times[f] = tt
        #

        print ("area: ", f, " ", stack[f].shape, stack_random[f].shape)

    stack = np.array(stack)
    stack_random = np.array(stack_random)
    print ("stacK: ", stack.shape, stack[0].shape, ' random: ', stack_random.shape)
    #
    return stack, stack_random, d1, times, names
#



#
def get_04_02_triggers(root_dir, recording):

    # make sure locs
    try:
        locs_44threshold = np.load(os.path.join(root_dir,
                                                    'tif_files',
                                                    recording,
                                                    recording + '_locs44threshold.npy'))
        #
        codes = np.load(os.path.join(root_dir,
                                     'tif_files',
                                     recording,
                                     recording + '_code44threshold.npy'))
    except:
        print ("files missing")

        return False

    # 
    code = b'04'
    idx = np.where(codes==code)[0]
    locs_selected_04 = locs_44threshold[idx]

    #
    code = b'02'
    idx = np.where(codes==code)[0]
    locs_selected_02 = locs_44threshold[idx]
                    

    #
    np.savetxt(os.path.join(root_dir,
                             'tif_files',
                             recording,
                             "rewarded_times.txt"),
               locs_selected_04)

    np.savetxt(os.path.join(root_dir,
                             'tif_files',
                             recording,
                             "nonrewarded_times.txt"),
               locs_selected_02)

    return True
    
def visualize_loaded_data(data_continuous, 
                          data_segments,
                          rewarded_pull_times,
                          nonrewarded_pull_times,
                          names):

    ax=plt.subplot(2,1,1)
    t=np.arange(data_continuous.shape[0])/30.


    # 
    start_ = 400
    end_ = 39000

    ############################################################## 
    ############ PLOT CONTINUOUS DATA + TRIGGER TIMES ############ 
    ##############################################################
    ''' This plots should show trigger times occuring right before strong
        peaks in [Ca++] activity; this is a very robust finding, so if
        the triggers don't occur right before high peaks, something is wrong
    '''

    #
    fs = [2,3,4]  # selecting limb features
    for f in fs:
        plt.plot(t[start_:end_],
                 data_continuous[start_:end_,f],
                 label=names[f],
                 alpha=.8)
    plt.plot([t[0],t[-1]],
        [0,0],'--')

    # rewarded pull times
    plt.scatter(rewarded_pull_times/30., 
                np.zeros(rewarded_pull_times.shape[0]),
                s=200,
                c='red', 
                label='rewarded_pull_times')
    
    # rewarded pull times
    plt.scatter(nonrewarded_pull_times/30., 
                np.zeros(nonrewarded_pull_times.shape[0]),
                s=200,
                c='magenta', 
                label='non-rewarded_pull_times')    

    plt.xlim(start_/30.,end_/30.)
    plt.title("Whole session + triggers")
    plt.legend()

    ##################################################### 
    ############ PLOT SEGMENTED AVERAGE DATA ############ 
    #####################################################
    ''' This plot should always have a strong peak in features just after
        t=0sec; if not, then it means alignment is incorrect
    '''
    #
    ax=plt.subplot(2,1,2)
    t=np.arange(data_segments.shape[1])/30.-30
    print ("Data segments: ", data_segments.shape)
    for f in fs:
        plt.plot(t,data_segments[:,:,f].mean(0).T, label=names[f])

    plt.plot([t[0],t[-1]],[
        0,0],'--')

    plt.title("All trial averages")
    plt.legend()
    plt.xlim(t[0],t[-1])
    plt.xlabel("time (sec)")
    plt.show()

  
# 
def load_rois_lever_pulls(root_dir, animal_id, session):
    # ROI FILENAME
    fname_whole_stack_ROI = os.path.join(root_dir,
                                   animal_id, 
                                   'tif_files',
                                   session,
                                   session+'_locanmf_wholestack.npz')

    # # PCA FILENAME
    # fname_whole_stack_PCA = os.path.join(root_dir,
    #                                 animal_id,
    #                                 'tif_files',
    #                                 session,
    #                                 session + '_whole_stack_trial_ROItimeCourses_15sec_pca30components.npy')

    # same for both PCA and ROI
    fname_lever_times = os.path.join(root_dir,
                                    animal_id,
                                    'tif_files',
                                    session,
                                    'rewarded_times.txt')

    fname_blue_light = os.path.join(root_dir,
                                    animal_id,
                                    'tif_files',
                                    session,
                                    'blue_light_frame_triggers.npz')

    ########################################
    ######### LOAD PCA DATA ################
    ########################################
    # NOT USED CURRENTLY FOR SUPER_session processing
    # data_segments, data_continuous, trigger_times, shift = load_data_PCA(fname_whole_stack_PCA,
    #                                                                      fname_lever_times,
    #                                                                      fname_blue_light)


    ########################################
    ######### LOAD ROI DATA ################
    ########################################
    (data_segments,
     data_segments_random,
     data_continuous, 
     rewarded_times,
     nonrewarded_times,
     names) = load_data_ROIs_lever_pulls(fname_whole_stack_ROI,
                                         fname_lever_times,
                                         fname_blue_light)

    #
    data_continuous -= data_continuous.mean(0)
    
	# get trial average
	# subtract mean to bring it to 0
    for k in range(data_segments.shape[0]):
        data_segments[k] -= data_segments[k].mean(0)
        data_segments_random[k] -= data_segments_random[k].mean(0)


    return data_segments, data_segments_random, data_continuous, rewarded_times, nonrewarded_times, names


#
def load_rois_body_movements(root_dir, animal_id, session):
    # ROI FILENAME
    fname_whole_stack_ROI = os.path.join(root_dir,
                                   animal_id,
                                   'tif_files',
                                   session,
                                   session+'_locanmf_wholestack.npz')


    # #
    # fname_body_movements = os.path.join(root_dir,
    #                                 animal_id,
    #                                 'tif_files',
    #                                 session,
    #                                 session + '_0secNoMove_movements.npz')

    #
    fname_blue_light = os.path.join(root_dir,
                                    animal_id,
                                    'tif_files',
                                    session,
                                    'blue_light_frame_triggers.npz')

    ########################################
    ######### LOAD PCA DATA ################
    ########################################
    # NOT USED CURRENTLY FOR SUPER_session processing
    # data_segments, data_continuous, trigger_times, shift = load_data_PCA(fname_whole_stack_PCA,
    #                                                                      fname_lever_times,
    #                                                                      fname_blue_light)


    ########################################
    ######### LOAD ROI DATA ################
    ########################################
    (data_segments,
     data_segments_random,
     data_continuous,
     times,
     names) = load_data_ROIs_body_movements(root_dir,
                                            animal_id,
                                            session,
                                            fname_whole_stack_ROI,
                                            fname_blue_light)

    #
    data_continuous -= data_continuous.mean(0)

	# get trial average
	# subtract mean to bring it to 0
    # print ("Data segemnts before subtracgtin mean: ", data_segments.shape)
    for f in range(len(data_segments)):
        for k in range(len(data_segments[f])):
            data_segments[f][k] -= data_segments[f][k].mean(0)
            data_segments_random[f][k] -= data_segments_random[f][k].mean(0)


    return data_segments, data_segments_random, data_continuous, times, names
