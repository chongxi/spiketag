import numpy as np
import pandas as pd
from sympy import binomial_coefficients
from .FET import FET
from .CLU import CLU
from ..view import scatter_3d_view, grid_scatter3d

class UNIT(object):
    """
    UNIT class load, visualize and analyze unit structured data
    unit structure in which each unit is: {time, group_id, fet0, fet1, fet2, fet3, spike_id}, each section uses a 32-bit integer to represent.

    - UNIT class bins the unit data.
    - UNIT class convert the feature part of the unit data to Spike FET, which can call sorting, and spiketag feature view. 

    Inputs:
        bin_len: float (s) of bin length
        nbins: int of number of bins
        binpoint: bits used to encode the interger part of a 32-bit number (default 13)
    """
    def __init__(self, bin_len=0.1, nbins=8, binpoint=13, sampling_rate=25000.0):
        super(UNIT, self).__init__()
        self._bin_len = bin_len
        self._nbins   = nbins
        self.binpoint = binpoint
        self.sampling_rate = sampling_rate

    def load_all(self, filename):
        pass

    def load_unitpacket(self, filename):
        '''
        1. pd dataframe
        2. fet.bin

        Both follows table structure:
        ['time', 'group_id', 'fet0', 'fet1', 'fet2', 'fet3', 'spike_id']
        '''
        self.filename = filename
        if filename.split('.')[-1]=='pd':
            self.df = pd.read_pickle(filename)
            self.df['frame_id'] /= self.sampling_rate
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
            fet[:, 2:6] /= float(2**self.binpoint)
            self.df = pd.DataFrame(fet,
                      columns=['time', 'group_id', 'fet0', 'fet1', 'fet2', 'fet3', 'spike_id'])
            self.df['time'] /= self.sampling_rate
            self.df['group_id'] = self.df['group_id'].astype('int')
            self.df['spike_id'] = self.df['spike_id'].astype('int')

        self.assign_fet()
        self.assign_bin()

    def assign_fet(self):
        fet_dict = {}
        self.groups = self.df.group_id.sort_values().unique()
        self.n_grp = len(self.groups)
        for g in range(self.n_grp):
            fet_dict[g] = self.df[self.df.group_id==g][['fet0', 'fet1', 'fet2', 'fet3']].to_numpy()
        self.fet = FET(fet_dict)

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
        self.bin_index = self.df['bin'].unique() 
        if self.df.bin.min() != 0 and self.df.bin.iloc[0] < self._nbins:
            self.bin_index = np.insert(self.bin_index, 0, self._nbins-1) # insert 7 (if nbins==8) at 0 position
        # We still need to remove the first 7 bins (if nbins==8) and the last bin was never trigger the binner to send out command
        self.bin_index = self.bin_index[self._nbins-1:-1] 

    def show(self, g=0):
        if g is None:
            gd = grid_scatter3d()
            gd.from_file(self.filename)
            gd.show()
        else:
            fet_view = scatter_3d_view()
            fet_view.show()
            fet_view.set_data(self.fet[g])
            fet_view.title = f'group {g}: {self.fet[g].shape[0]} spikes'

    def load_behavior(self, filename):
        pass
