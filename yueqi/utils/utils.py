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
def load_data_ROIs(fname_whole_stack,
                   fname_lever_times,
                   fname_blue_light):

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
    for t in times:
        temp = d1[t-900:t+900]
        if len(temp)==1800:
            stack.append(temp)

        min_lockout = 30*3  # # of seconds away from

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
def load_rois_and_pca(root_dir, animal_id, session):
    # ROI FILENAME
    fname_whole_stack_ROI = os.path.join(root_dir,
                                   animal_id, 
                                   'tif_files',
                                   session,
                                   session+'_locanmf_wholestack.npz')

    # PCA FILENAME
    fname_whole_stack_PCA = os.path.join(root_dir,
                                    animal_id,
                                    'tif_files',
                                    session,
                                    session + '_whole_stack_trial_ROItimeCourses_15sec_pca30components.npy')

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
     names) = load_data_ROIs(fname_whole_stack_ROI,
                             fname_lever_times,
                             fname_blue_light)

    #
    data_continuous -= data_continuous.mean(0)
    
	# get trial average
    #data_segments = data_segments.mean(0)

	# subtract mean to bring it to 0
    # print ("Data segemnts before subtracgtin mean: ", data_segments.shape)
    for k in range(data_segments.shape[0]):
        data_segments[k] -= data_segments[k].mean(0)
        data_segments_random[k] -= data_segments_random[k].mean(0)


    return data_segments, data_segments_random, data_continuous, rewarded_times, nonrewarded_times, names
