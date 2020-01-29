import numpy as np
import pandas as pd


class UNIT(object):
    """
    neural units: could be single units or any other
    """
    def __init__(self):
        super(UNIT, self).__init__()

    def load_all(self, filename):
        pass

    def load_unitpacket(self, filename):
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

    def load_behavior(self, filename):
        pass
        