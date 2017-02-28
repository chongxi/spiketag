#--------------------------------------------------------------
# Here is for all correlote method
#--------------------------------------------------------------
import numpy as np

def correlate(spike_time, cluster_with_idx, fs=25000, window_size=50, bin_size=1):
    '''
        calculte the correlate  for every pair of cluster. The main step is :
        1. convert target to spike train:
            r = [2,6]
            t = [0,3,4,7] ---> [1,0,0,1,1,0,0,1,0,0]
            window = [1,1,1,1] # window_size = 4, binsize and offset = 1
        2. iterate the reference where has spike (value = 1):
            r = [2,6]
            t = [1,0,0,1,1,0,0,1,0,0]
                     |       |
                 x x | x x   |
                     |       | 
                [1,1   1,1]  |
                             |
                         x x | x x
                             |
                        [1,1   1,1]
               --> counter += [1,0,1,1] * [1,1,1,1] + [1,0,1,0] * [1,1,1,1]

        return array(shape=[i,j,counter]): counter sum the spikes which happend in every bin respectly
        
        Parameter
        ---------------------
        spike_time : numpy array 
            this array includes time of spikes
        cluster_with_idx : dist 
            key is the clu_no, values are the index of spike time of all spikes belong to this clu_no within spike_time array above, eg:
            {0:{0,1},1:{2,3}}
        fs : int
            sample rate
        window_size : int  
            the unit is ms
        bin_size : int
            the unit is ms
    
    '''
    assert window_size % 2 == 0
    assert window_size % bin_size == 0
  
    samples_in_bin  = int(fs / 1e3 * bin_size)
    samples_in_window = window_size / bin_size * samples_in_bin
   
    assert spike_time is not  None
    assert cluster_with_idx is not None

    clus_nums = len(cluster_with_idx.keys())
    ccg = np.zeros(shape=(clus_nums, clus_nums, window_size / bin_size), dtype='int32')
    
    for i in reversed(range(clus_nums)):
        
        t = spike_time[cluster_with_idx[i]]
        t_train  = _to_train(t, spike_time[-1] + 1)
        
        for j in range(i + 1):
            r = spike_time[cluster_with_idx[j]]
            ccg[i][j] = ccg[j][i] = _do_correlate(r, t_train, samples_in_window
                    , samples_in_bin)

    return ccg
            
def _do_correlate(reference, target, samples_in_window, samples_in_bin):
    '''
        do correlate one by one. plus 1 make the window center the reference
    '''
    window = np.ones(samples_in_window + 1, dtype='int32')
    counter = np.zeros(samples_in_window + 1 , dtype='int32')
    radius =  samples_in_window / 2

    for r in reference:
        cliped_t = _clip(target, r, radius)
        counter += cliped_t * window
    
    counter = np.delete(counter,radius)
    return counter.reshape(samples_in_window / samples_in_bin, samples_in_bin).sum(axis=1)

def _clip(source, center, radius):
    '''
       clip the source array from center - radius to center + radius, 
       automatic left align zero or right align zero if index out of bound
    '''
       
    start,end = center - radius, center + radius + 1
    cliped = None

    if start >= 0 and end <= len(source):
        cliped = source[start:end]
    elif start < 0:
        left_align_zero = np.zeros(abs(start), dtype='int32')
        cliped = np.hstack((left_align_zero,source[0:end]))
    elif end > len(source):
        right_align_zero = np.zeros(end - len(source), dtype='int32')
        cliped = np.hstack((source[start:len(source)],right_align_zero))

    assert len(cliped) == radius * 2 + 1
    
    return cliped
   
            
def _to_train(idx, sample_length):
    '''
      according the size of data, raster the absolute pos to relative pos:
      len(data) = 6
      r = [1,5] -> [0,1,0,0,0,1]
    '''
    t =  np.zeros(sample_length, dtype='int32')
    t[idx] = 1
    return t 

  
