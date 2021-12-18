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

    # this is the calcium excitation light, we need to shift the dataset to match it
    if True:
        blue = np.load(fname_blue_light)
        start = blue['start_blue']
        d1 = np.roll(d1,-start,axis=0)

    ##################################################################################
    ##################################################################################
    ##################################################################################




    ##################################################################################
    ########################### LOAD CA DATA #########################################
    ##################################################################################

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
            #if len(shift)>1:
            #    shift=shift[0]
            #print ("Ideal DLC shift: ", shift )
            break


    #####################################################
    ########## LOAD BODY MOVEMENT CA DATA ###############
    #####################################################
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
    print ("stacK: ",
           stack.shape,
           stack[0].shape,
           ' random: ',
           stack_random.shape)
    #
    return stack, stack_random, d1, times, names


################################################################
################################################################
################################################################
def load_data_rois_lever_pulls_body_lockout(
                                             root_dir,
                                             animal_id,
                                             session,
                                             fname_whole_stack,
                                             fname_lever_times,
                                             fname_blue_light,
                                             n_sec_lockout,
                                             body_feats_lockout
                                            ):

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
    #

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

    ##########################################################
    ############ LOAD REWARDED AND NONREWARDED TIMES #########
    ##########################################################
    imaging_rate = 30
    times = np.loadtxt(fname_lever_times)
    rewarded_times = np.int32(times*imaging_rate)

    nonrewarded_times = np.loadtxt(fname_lever_times.replace('rewarded','nonrewarded'))
    nonrewarded_times = np.int32(nonrewarded_times*imaging_rate)


    ##########################################################
    ############### LOAD BODY MOVEMENT TIMES #################
    ##########################################################
    fname_ = os.path.join(root_dir,
                                animal_id,
                                'tif_files',
                                session,
                                session+'_0secNoMove_movements.npz')

    # load the body movement datasets
    if os.path.exists(fname_):
        data = np.load(fname_, allow_pickle=True)
        names = data['labels']

        ############################################
        ##### FIGURE OUT BETWEEN DLC AND CA ########
        ############################################
        # load video shift peaks from left paw, right paw and lever DLC traces
        shift = data['video_shift']

        # load the manual shift id from file to select one of the shifts from above
        good_alignments = np.loadtxt(os.path.join(root_dir,
                                                  animal_id,
                                                  'tif_files',
                                                  'sessions_DLC_alignment_good.txt'),
                                     dtype='str')

        # find from text file which shift was the correct one
        found = False
        for k in range(len(good_alignments)):
            temp = good_alignments[k][0].replace(',','')
            if temp in session:
                shift_id = int(good_alignments[k][1])
                shift = shift[shift_id]
                found = True
                break

        # take the right paw time
        if found == False:
            shift = shift[1]

        print (session, " shift: ", shift)

        # convert the shift in sec into frames at 30Hz
        shift_frames = shift * 30

        ###############################################
        ##### LOAD BODY MOVEMENT TRIGGER TIMES ########
        ###############################################
        # load binarized body initiation matrices
        inits = data['feature_initiations']

        # loading times array for all the body triggers
        times_body_parts = []
        for k in range(inits.shape[0]):
            idx = np.where(inits[k]==1)[0]
            idx = np.int32(idx*2-shift_frames)   #multiple by 2 to convert from 15fps to 30fps; and shift back
            times_body_parts.append(idx)

    # no body movement data
    else:
        print ("0sec mov files missing ", session)
        times_body_parts = [[],[],[],[],[],[]]

        if len(body_feats_lockout)>0:
            print ("exiting early as no video: ", session)
            print ('body_feats_lockout: ', body_feats_lockout)
            print ('')
            return np.zeros(0,'int32'),np.zeros(0,'int32'),np.zeros(0,'int32'),np.zeros(0,'int32')

    ##################################################################################
    ############### FIND LEVER PULLS THAT ARE LOCKED OUT BY N_SEC ####################
    ##################################################################################

    # use only selected features as confounds;
    body_part_times = []
    for f in body_feats_lockout:
        body_part_times.extend(times_body_parts[f])

    # pool all non rewarded times together:
    temp = np.array(body_part_times).flatten()

    # check if there are indeed body features requested as lockout
    # if so, then we must condition on body features and there should be some feature data
    # otherwise return empty arrays as we don't want to compute on the data with
    # nonrewarded lever pulls as the only confounds


    # always use nonrewarded times as confounds even if no features are being used
    confound_times = np.int32(np.hstack((nonrewarded_times, temp)))

    # also add rewarded times as they are also confounds as they can occur anytime after 3 sec lockout
    confound_times = np.int32(np.hstack((rewarded_times, confound_times)))

    # search across rewarded_times for those that are locked out from other body movements
    def get_locked_out_times(rewarded_times,
                             confound_times,
                             n_sec_lockout):

        #
        times= []
        for t in rewarded_times:
            diff = t-confound_times

            # select only confound times preceding current time t:
            #idx = np.where(diff>=0)[0]
            idx = np.where(diff>0)[0]   # use strict > 0 to exclude the current t time that is present in the
                                        #    confound data as rewarded time t also
                                        # also, it's ok if the other body part movements occur exactly in the same
                                        # frame...
            if idx.shape[0]==0:
                continue
            diff = diff[idx]

            # check that the nearest time is at least n-sec away
            if np.min(diff)>=n_sec_lockout:
                times.append(t)

        times = np.array(times)

        return times

    #
    n_sec_lockout = n_sec_lockout*30
    times = get_locked_out_times(rewarded_times,
                                 confound_times,
                                 n_sec_lockout)


    ##################################################################################
    ########################### LOAD CA DATA #########################################
    ##################################################################################

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

    ######## finalize data #############

    stack = np.array(stack)
    stack_random = np.array(stack_random)
    #print ("stacK: ", stack.shape, ' random: ', stack_random.shape)

    # also save all behaviors and names
    names = [
             'rewarded_lever_pulls',
             'unrewarded_lever_pulls',
             'left_paw',
             'right_paw',
             'nose',
             'jaw',
             'right_ear',
             'tongue',
             'lever_DLC'
             ]

    all_behaviors = []
    all_behaviors.append(rewarded_times)
    all_behaviors.append(nonrewarded_times)
    for k in range(len(times_body_parts)):
        all_behaviors.append(times_body_parts[k])

    #
    return stack, stack_random, names, all_behaviors
#

def generate_body_movement_lockout_data(n_sec_lockouts,
                                       body_feats_lockouts,
                                       animal_ids,
                                       root_dir,
                                       ):

    # loop over sec lockout
    for n_sec_lockout in n_sec_lockouts:

        # loop over body movement lockouts
        for body_feats_lockout in body_feats_lockouts:

            if len(body_feats_lockout)==0:
                if n_sec_lockout in [1,2,3]:
                    print ("skipping redundant computation")
                    print ("         animal id: ", animal_id,
                           "  n-sec: ", n_sec_lockout,
                           "  body feats: ", body_feats_lockout)
                    continue


            # loop over all animals
            for animal_id in animal_ids:

                fname_out_animal = os.path.join(root_dir,
                         animal_id,
                         'super_sessions',
                         "alldata_body_and_nonreward_lockout_"+
                         str(n_sec_lockout)+"secLockout_"+
                         str(body_feats_lockout)+"bodyfeats.npz"
                         )
                if os.path.exists(fname_out_animal):
                    print ("File already exists: ", fname_out_animal)
                    continue

                sessions = np.loadtxt(os.path.join(root_dir,
                                                   animal_id,
                                                  'tif_files',
                                                  "sessions.txt"),dtype='str')
                #print ("sessions: ", sessions)
                #
                all_trials = []
                all_random = []
                all_behaviors = []
                all_names = []
#                 for session in tqdm(sessions, desc=animal_id+"_"+
#                                     str(n_sec_lockout)+"_"+
#                                     str(body_feats_lockout)):

                for session in (sessions):
                    session = os.path.split(session)[1]
                    # check that code04 exists in file
                    times04 = get_04_02_triggers(os.path.join(root_dir,animal_id), session)
                    #print ("Times 04; ", times04)

                    #
                    if times04.shape[0]<10:
                        print (session, "  < 10 trials")
                        continue

                    ###########################################################################
                    ############### LOAD LEVER PULLS EXCLUDING LOCKOUTS #######################
                    ###########################################################################
                    (data_segments,
                     data_segments_random,
                     names,
                     behaviors) = load_rois_lever_pulls_body_lockout(root_dir,
                                                                     animal_id,
                                                                     session,
                                                                     n_sec_lockout,
                                                                     body_feats_lockout)

                    ######################################
                    ######### MAKE TRAINING DATA #########
                    ######################################

                    if data_segments.shape[0]==0:
                        continue

                    #
                    fname_out = os.path.join(root_dir,
                                             animal_id,
                                             'tif_files',
                                             session,
                                             session+"_body_and_nonreward_lockout_"+
                                             str(n_sec_lockout)+"secLockout_"+
                                             str(body_feats_lockout)+"bodyfeats.npz"
                                             )

                    #
                    np.savez(
                             fname_out,
                             trials = data_segments,
                             random = data_segments_random,
                             behaviors = behaviors,
                             names = names
                            )

                    #
                    all_trials.append(data_segments)
                    all_random.append(data_segments_random)
                    all_behaviors.append(behaviors)
                    all_names.append(names)

                # save whole animal stack:
                try:
                    os.mkdir(os.path.join(root_dir,
                                          animal_id,
                                          'super_sessions'))
                except:
                    pass


                np.savez(
                         fname_out_animal,
                         trials = all_trials,
                         random = all_random,
                         behaviors = all_behaviors,
                         names = all_names
                        )


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
        print ("code04 files missing")

        return np.zeros(0,'int32')

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

    if len(locs_selected_04)==0:
        locs_selected_04 = np.zeros(0,'int32')

    return locs_selected_04
    
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
def load_rois_lever_pulls_body_lockout(root_dir,
                                       animal_id,
                                       session,
                                       n_sec_lockout,
                                       body_feats_lockout):

    # ROI FILENAME
    fname_whole_stack_ROI = os.path.join(root_dir,
                                         animal_id,
                                         'tif_files',
                                         session,
                                         session+'_locanmf_wholestack.npz')

    if os.path.exists(fname_whole_stack_ROI)==False:
        print ("whole stack missing", session)
        return np.zeros(0,'int32'),np.zeros(0,'int32'),np.zeros(0,'int32'),np.zeros(0,'int32')


    # same for both PCA and ROI
    fname_lever_times = os.path.join(root_dir,
                                    animal_id,
                                    'tif_files',
                                    session,
                                    'rewarded_times.txt')

    #
    fname_blue_light = os.path.join(root_dir,
                                    animal_id,
                                    'tif_files',
                                    session,
                                    'blue_light_frame_triggers.npz')

    ########################################
    ######### LOAD ROI DATA ################
    ########################################
    (data_segments,
     data_segments_random,
     names,
     all_behaviors) = load_data_rois_lever_pulls_body_lockout(
                                                             root_dir,
                                                             animal_id,
                                                             session,
                                                             fname_whole_stack_ROI,
                                                             fname_lever_times,
                                                             fname_blue_light,
                                                             n_sec_lockout,
                                                             body_feats_lockout
                                                             )

	# get trial average
	# subtract mean to bring it to 0
    for k in range(data_segments.shape[0]):
        data_segments[k] -= data_segments[k].mean(0)
        data_segments_random[k] -= data_segments_random[k].mean(0)

    #
    return data_segments, data_segments_random, names, all_behaviors

  
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
