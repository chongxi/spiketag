import numpy as np
import pandas as pd
from sympy import binomial_coefficients


class UNIT(object):
    """
    bin_len: float (s) of bin length
    """
    def __init__(self, bin_len=0.1):
        super(UNIT, self).__init__()
        self._bin_len = bin_len

    def load_all(self, filename):
        pass

    def load_unitpacket(self, filename):
        '''
        1. pd dataframe
        2. fet.bin

        Both follows table structure:
        ['time', 'group_id', 'fet0', 'fet1', 'fet2', 'fet3', 'spike_id']
        '''
        if filename.split('.')[-1]=='pd':
            self.df = pd.read_pickle(filename)
            self.df['frame_id'] /= 25000.
            self.df.rename(columns={'frame_id':'time'}, inplace=True)
            self.df['group_id'] = self.df['group_id'].astype('int')
            self.df['spike_id'] = self.df['spike_id'].astype('int')
            # self.df.set_index('spike_id', inplace=True)
            # self.df.index = self.df.index.astype(int)
            # self.df.index -= self.df.index.min()
            # self.df['spk'] = self.df
            # self.spk_time_dict = {i: self.df.loc[i]['time'].to_numpy() 
            #                       for i in self.df.index.unique().sort_values()}
            # self.df['spk'].reset_index(inplace=True)
            # self.n_units = np.sort(self.df.spike_id.unique()).shape[0]
            # self.n_groups = np.sort(self.df.group_id.unique()).shape[0]

        elif filename.split('.')[-1]=='bin':
            fet = np.fromfile(filename, dtype=np.int32).reshape(-1, 7).astype(np.float32)
            fet[:, 2:6] /= float(2**16)
            self.df = pd.DataFrame(fet,
                      columns=['time', 'group_id', 'fet0', 'fet1', 'fet2', 'fet3', 'spike_id'])
            self.df['time'] /= 25000.
            self.df['group_id'] = self.df['group_id'].astype('int')
            self.df['spike_id'] = self.df['spike_id'].astype('int')

        self.assign_bin()

    @property
    def bin_len(self):
        return self._bin_len
    
    @bin_len.setter
    def bin_len(self, value):
        self._bin_len = value
        self.assign_bin()

    def assign_bin(self):
        '''
        critical function to 
        1. assign bin (and end time of each bin) to each spike
        2. assign bin_index to each bin
        '''
        # assign `bin` number to each spike 
        self.df['bin'] = self.df.time.apply(lambda x: int(x//self.bin_len))
        self.df['bin_end_time'] = self.df.time.apply(
            lambda x: (int(x//self.bin_len)+1)*self.bin_len)
        # assign bin_index to each bin (bin_index will be used to index scv.bin stored in BMI experiment)
        self.bin_index = np.unique(self.df['bin'].to_numpy())[7:-1]

    def load_behavior(self, filename):
        pass
