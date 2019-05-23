import numpy as np

def get_spkwav(filename):
    '''
    spk_wav.bin
    '''
    spkwav  = np.fromfile(filename, dtype=np.int32).reshape(-1, 20, 4)
    return spkwav

def get_spkinfo(filename):
    '''
    spk.bin
    '''
    spkinfo = np.fromfile(filename, dtype=np.int32).reshape(-1,2)
    return spkinfo