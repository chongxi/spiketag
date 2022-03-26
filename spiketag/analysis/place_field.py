import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.pyplot import cm
from scipy.interpolate import interp1d
from .core import spk_time_to_scv, firing_pos_from_scv, smooth
from ..base import SPKTAG
from ..utils import colorbar
from ..utils.plotting import colorline



def info_bits(Fr, P):
    Fr[Fr==0] = 1e-25
    MFr = sum(P.ravel()*Fr.ravel())
    return sum(P.ravel()*(Fr.ravel()/MFr)*np.log2(Fr.ravel()/MFr))


def info_sparcity(Fr, P):
    Fr[Fr==0] = 1e-25
    MFr = sum(P.ravel()*Fr.ravel())
    return sum(P.ravel()*Fr.ravel()**2/MFr**2)


class place_field(object):
    '''
    place cells class contains `ts` `pos` `scv` for analysis
    load_log for behavior
    load_spktag for spike data
    get_fields for computing the representaions using spike and behavior data
    '''
    def __init__(self, pos, v_cutoff=5, bin_size=2.5, ts=None, t_step=None, maze_range=None):
        '''
        resample the trajectory with new time interval
        reinitiallize with a new t_step (dt)
        '''
        if ts is None:
            ts = np.arange(0, pos.shape[0]*t_step, t_step)
            self.t_step = t_step
        self.ts, self.pos = ts, pos
        self._ts_restore, self._pos_restore = ts, pos
        self.spk_time_array, self.spk_time_dict = None, None
        self.df = {}

        # key parameters for initialization (before self.initialize we need to align behavior with ephys) 
        self.bin_size = bin_size
        self.v_cutoff = v_cutoff
        self.maze_range = maze_range
        self.initialize(bin_size=self.bin_size, v_cutoff=self.v_cutoff)

    def __call__(self, t_step):
        '''
        resample the trajectory with new time interval
        reinitiallize with a new t_step (dt)
        '''
        fs = self.fs 
        new_fs = 1/t_step
        self.t_step = t_step
        self.ts, self.pos = self.interp_pos(self.ts, self.pos, self.t_step)
        self.initialize(bin_size=self.bin_size, v_cutoff=self.v_cutoff)


    def restore(self):
        self.ts, self.pos = self._ts_restore, self._pos_restore

    @property
    def dt(self):
        return self.ts[1] - self.ts[0]

    @property
    def fs(self):
        self._fs = 1/(self.ts[1]-self.ts[0])
        return self._fs

    def interp_pos(self, t, pos, new_dt):
        '''
        convert irregularly sampled pos into regularly sampled pos
        N is the dilution sampling factor. N=2 means half of the resampled pos
        example:
        >>> new_fs = 200.
        >>> pc.ts, pc.pos = pc.interp_pos(ts, pos, N=fs/new_fs)
        '''
        dt = t[1] - t[0]
        x, y = interp1d(t, pos[:,0], fill_value="extrapolate"), interp1d(t, pos[:,1], fill_value="extrapolate")
        new_t = np.arange(t[0], t[-1], new_dt)
        new_pos = np.hstack((x(new_t).reshape(-1,1), y(new_t).reshape(-1,1)))
        return new_t, new_pos 


    def align_with_recording(self, recording_start_time, recording_end_time, replay_offset=0):
        '''
        replay_offset should be 0 if and only if pc.ts[0] is the actual ephys start time 0
        the ephys actual start time 0 = recording_start_time + replay_offset

        ts before alignment   |--------------------|
        behavior  start:      |
        behavior    end:                           |
        recording start:         |------------
        recording   end:          ------------|
        replay_offset  :             |
        ts after alignment           |------------| 
        '''
        self.ts += replay_offset   # 0 if the ephys is not offset by replaying through neural signal generator
        self.pos = self.pos[np.logical_and(self.ts>=recording_start_time, self.ts<=recording_end_time)]
        self.ts  =  self.ts[np.logical_and(self.ts>=recording_start_time, self.ts<=recording_end_time)]
        self.t_start = self.ts[0]
        self.t_end   = self.ts[-1]
        self._ts_restore, self._pos_restore = self.ts, self.pos


    def initialize(self, bin_size, v_cutoff):
        self.v_cutoff = v_cutoff
        self.get_maze_range()
        self.get_speed() 
        self.occupation_map(bin_size)
        self.pos_df = pd.DataFrame(np.hstack((self.ts.reshape(-1,1), self.pos)),
                                   columns=['time', 'x', 'y'])
        # self.binned_pos = (self.pos-self.maze_original)//self.bin_size


    def get_maze_range(self):
        if self.maze_range is None:
            self.maze_range = np.vstack((self.pos.min(axis=0), self.pos.max(axis=0))).T
            self._maze_original = self.maze_range[:,0] # the left, down corner location
        else:
            self.maze_range = np.array(self.maze_range)
            self._maze_original = self.maze_range[:,0] # the left, down corner location

    @property
    def maze_center(self):
        self._maze_center = self.maze_original[0]+self.maze_length[0]/2, self.maze_original[1]+self.maze_length[1]/2
        return self._maze_center

    @property
    def maze_original(self):
        return self._maze_original

    @property
    def maze_length(self):
        return np.diff(self.maze_range, axis=1).ravel()

    @property
    def maze_ratio(self):
        return self.maze_length[0]/self.maze_length[1]

    @property
    def binned_pos(self):
        return (self.pos-self.maze_original)//self.bin_size

    def binned_pos_2_real_pos(self, binned_pos):
        pos = binned_pos*self.bin_size + self.maze_original
        return pos

    def real_pos_2_binned_pos(self, real_pos, interger_output=True):
        if interger_output:
            binned_pos = (real_pos - self.maze_original)//self.bin_size
        else:
            binned_pos = (real_pos - self.maze_original)/self.bin_size
        return binned_pos

    def get_speed(self):
        '''
        self.ts, self.pos is required

        To consider that some/many place cells start firing before moving, and stop firing a few seconds after moving, we 
        need a wider smoothing window. 

        v_smoothed_wide is a larger window to calculate the low_speed_idx

        The `low_speed_idx` are thos index of `ts` and `pos` that are too slow to be considered to calculate the place field. 
        '''
        self.v = np.linalg.norm(np.diff(self.pos, axis=0), axis=1)/np.diff(self.ts)
        self.v = np.hstack((self.v[0], self.v))
        self.v_smoothed = smooth(self.v.reshape(-1,1), int(np.round(self.fs))).ravel()
        self.v_smoothed_wide = 2 * smooth(self.v.reshape(-1,1), 4*int(np.round(self.fs))).ravel()
        self.low_speed_idx = np.where(self.v_smoothed_wide < self.v_cutoff)[0]

        self.df['pos'] = pd.DataFrame(data=np.hstack((self.pos, self.v_smoothed.reshape(-1,1))), index=self.ts, 
                                        columns=['x','y','v'])
        self.df['pos'].index.name = 'ts'

        '''
        # check speed:
        f, ax = plt.subplots(1,1, figsize=(18,8))
        offset=20000
        plot(ts[offset:1000+offset], v[offset:1000+offset])
        plot(ts[offset:1000+offset], v_smoothed[offset:1000+offset])
        ax.axhline(5, c='m', ls='-.')
        '''
        # return v_smoothed, v

    def plot_speed(self, start=None, stop=None, v_cutoff=5):
        if start is None:
            start = self.ts[0]
        if stop is None:
            stop = self.ts[-1]
        fig, ax = plt.subplots(1,1, figsize=(18,5))
        period = np.logical_and(self.ts>start, self.ts<stop)
        plt.plot(self.ts[period], self.v[period], alpha=.7)
        plt.plot(self.ts[period], self.v_smoothed[period], lw=3)
        ax.axhline(v_cutoff, c='m', ls='-.')
        sns.despine()
        return fig
        

    def occupation_map(self, bin_size=4, time_cutoff=None):
        '''
        f, ax = plt.subplots(1,2,figsize=(20,9))
        ax[0].plot(self.pos[:,0], self.pos[:,1])
        ax[0].plot(self.pos[0,0], self.pos[0,1], 'ro')
        ax[0].plot(self.pos[-1,0], self.pos[-1,1], 'ko')
        ax[0].pcolormesh(self.X, self.Y, self.O, cmap=cm.hot_r)
        sns.heatmap(self.O[::-1]*self.dt, annot=False, cbar=False, ax=ax[1])
        '''
        # if maze_range != 'auto':
        #     self.maze_range = maze_range
        self.maze_size = np.array([self.maze_range[0][1]-self.maze_range[0][0], self.maze_range[1][1]-self.maze_range[1][0]])
        self.bin_size  = bin_size
        self.nbins = self.maze_size/bin_size
        self.nbins = self.nbins.astype(int)
        # occupation, self.x_edges, self.y_edges = np.histogram2d(x=self.pos[1:,0], y=self.pos[1:,1], 
        #                                                         bins=self.nbins, range=self.maze_range)
        idx = np.where(self.v_smoothed >= self.v_cutoff)[0]
        if time_cutoff is not None:
            idx = np.delete(idx, np.where(self.ts[idx]>time_cutoff)[0])
        occupation, self.x_edges, self.y_edges = np.histogram2d(x=self.pos[idx,0], y=self.pos[idx,1], 
                                                                bins=self.nbins, range=self.maze_range)
        self.X, self.Y = np.meshgrid(self.x_edges, self.y_edges)
        self.O = occupation.T.astype(int)  # Let each row list bins with common y range.
        self.P = self.O/float(self.O.sum()) # occupation prabability

        #### parameter used to calculate the fields
        self.kernlen = 18
        self.kernstd = 2.5


    def plot_occupation_map(self, cmap=cm.viridis):
        f, ax = plt.subplots(1,2,figsize=(20,9))
        ax[0].plot(self.pos[:,0], self.pos[:,1])
        ax[0].plot(self.pos[0,0], self.pos[0,1], 'ro')
        ax[0].plot(self.pos[-1,0], self.pos[-1,1], 'go')
        ax[0].pcolormesh(self.X, self.Y, self.O, cmap=cmap)
        ax[1].pcolormesh(self.X, self.Y, self.O, cmap=cmap)
        plt.show()

    @property
    def map_binned_size(self):
        return np.array(np.diff(self.maze_range)/self.bin_size, dtype=np.int).ravel()[::-1]

    @staticmethod
    def gkern(kernlen=21, std=2):
        """Returns a 2D Gaussian kernel array."""
        gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
        gkern2d = np.outer(gkern1d, gkern1d)
        gkern2d /= gkern2d.sum()
        return gkern2d


    def _get_field(self, spk_times):
        '''
        spk_times: spike times in seconds (an Numpy array), for example:
        array([   1.38388,    1.6384 ,    1.7168 , ..., 2393.72648, 2398.52484, 2398.538  ])

        Can be read out of spk_time_dict, which is loaded from pc.load_spktag() or pc.load_spkdf() methods.
        spk_times = pc.spk_time_dict[2]

        Used by `get_fields` method to calculate the place fields for all neurons in pc.spk_time_dict.
        '''
        spk_ts_idx = np.searchsorted(self.ts, spk_times) - 1
        spk_ts_idx = spk_ts_idx[spk_ts_idx>0]
        idx = np.array([_ for _ in spk_ts_idx if _ not in self.low_speed_idx], dtype=np.int)
        # idx = np.setdiff1d(spk_ts_idx, self.low_speed_idx)
        self.firing_ts  = self.ts[idx]
        self.firing_pos = self.pos[idx]        
        self.firing_map, x_edges, y_edges = np.histogram2d(x=self.firing_pos[:,0], y=self.firing_pos[:,1], 
                                                           bins=self.nbins, range=self.maze_range)
        self.firing_map = self.firing_map.T
        np.seterr(divide='ignore', invalid='ignore')
        self.FR = self.firing_map/(self.O*self.dt)
        self.FR[np.isnan(self.FR)] = 0
        self.FR[np.isinf(self.FR)] = 0
        self.FR_smoothed = signal.convolve2d(self.FR, self.gkern(self.kernlen, self.kernstd), boundary='symm', mode='same')
        return self.FR_smoothed


    def firing_map_from_scv(self, scv, t_step, section=[0,1]):
        '''
        firing heat map constructed from spike count vector (scv) and position
        '''
        # assert(scv.shape[1]==self.pos.shape[0])
        scv = scv.T.copy()
        n_neurons, total_bin = scv.shape
        valid_bin = np.array(np.array(section)*total_bin, dtype=np.int)
        firing_map_smoothed = np.zeros((n_neurons, *self.map_binned_size))
        for neuron_id in range(n_neurons):
            firing_pos = firing_pos_from_scv(scv, self.pos, neuron_id, valid_bin)
            firing_map, x_edges, y_edges = np.histogram2d(x=firing_pos[:,0], y=firing_pos[:,1], 
                                                          bins=self.nbins, range=self.maze_range)
            firing_map = firing_map.T/self.O/t_step
            firing_map[np.isnan(firing_map)] = 0
            firing_map[np.isinf(firing_map)] = 0
            firing_map_smoothed[neuron_id] = signal.convolve2d(firing_map, self.gkern(self.kernlen, self.kernstd), boundary='symm', mode='same')
            firing_map_smoothed[firing_map_smoothed==0] = 1e-25

        self.fields = firing_map_smoothed
        self.n_fields = self.fields.shape[0]
        self.n_units  = self.n_fields


    def get_field(self, spk_time_dict, neuron_id, start=None, end=None):
        '''
        wrapper of _get_field method
        calculate the place field of a single neuron (neuron_id) in a dictionary of spike times (spk_time_dict)

        Also, can restrict the time range of the place field by `start` and `end`.
        '''
        spk_times = spk_time_dict[neuron_id]
        ### for cross-validation and field stability check
        ### calculate representation from `start` to `end`
        if start is not None and end is not None:
            spk_times = spk_times[np.logical_and(start<=spk_times, spk_times < end)]
        self._get_field(spk_times)


    def _plot_field(self, trajectory=False, cmap='viridis', marker=True, alpha=0.5, markersize=5, markercolor='m'):
        f, ax = plt.subplots(1,1,figsize=(13,10));
        pcm = ax.pcolormesh(self.X, self.Y, self.FR_smoothed, cmap=cmap);
        plt.colorbar(pcm, ax=ax, label='Hz');
        if trajectory:
            ax.plot(self.pos[:,0], self.pos[:,1], alpha=0.8);
            ax.plot(self.pos[0,0], self.pos[0,1], 'ro');
            ax.plot(self.pos[-1,0],self.pos[-1,1], 'ko');
        if marker:
            ax.plot(self.firing_pos[:,0], self.firing_pos[:,1], 'o', 
                    c=markercolor, alpha=alpha, markersize=markersize);
        return f,ax


    def get_fields(self, spk_time_dict=None, start=None, end=None, v_cutoff=None, rank=True):
        '''
        spk_time_dict is dictionary start from 0: (each spike train is a numpy array) 
        {0: spike trains for neuron 0
         1: spike trains for neuron 1 
         2: spike trains for neuron 2
         ...
         N: spike trains for neuron N}
        '''

        if spk_time_dict is None:
            spk_time_dict = self.spk_time_dict

        self.n_fields = len(spk_time_dict.keys())
        self.n_units  = self.n_fields
        self.fields = np.zeros((self.n_fields, self.O.shape[0], self.O.shape[1]))
        self.firing_pos_dict = {}

        if v_cutoff is None:
            self.get_speed()    # ! critical for generating `low_speed_idx`
        else:
            self.v_cutoff = v_cutoff
            self.get_speed()    # ! critical for generating `low_speed_idx`

        print(spk_time_dict.keys())

        for i in spk_time_dict.keys():
            ### get place fields from neuron i
            self.get_field(spk_time_dict, i, start, end)
            self.fields[i] = self.FR_smoothed
            self.firing_pos_dict[i] = self.firing_pos
            self.fields[i] = self.FR_smoothed
            ### metrics for place fields

        self.fields[self.fields==0] = 1e-25

        if rank is True:
            self.rank_fields(metric_name='spatial_bit_smoothed_spike')


    def plot_fields(self, idx=None, nspks=None, N=10, size=3, cmap='hot', marker=False, markersize=1, alpha=0.8, order=False):
        '''
        order: if True will plot with ranked fields according to the metric 
        '''
        if idx is None: # plot all fields
            nrow = self.n_fields/N + 1
            ncol = N
            fig = plt.figure(figsize=(ncol*size, nrow*size));
            # plt.tight_layout();
            plt.subplots_adjust(wspace=None, hspace=None);
            for i in range(self.n_fields):
                ax = fig.add_subplot(nrow, ncol, i+1);
                if order:
                    field_id = self.sorted_fields_id[i]
                else:
                    field_id = i
                pcm = ax.pcolormesh(self.X, self.Y, self.fields[field_id], cmap=cmap);
                ax.set_title('#{0}: {1:.2f}Hz'.format(field_id, self.fields[field_id].max()), fontsize=20)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect(self.maze_ratio)
                if marker:
                    ax.plot(self.firing_pos_dict[field_id][:,0], self.firing_pos_dict[field_id][:,1], 
                                                              'mo', markersize=markersize, alpha=alpha)
            plt.grid(False)
            plt.show();

        else:
            nrow = len(idx)/N + 1
            ncol = N
            fig = plt.figure(figsize=(ncol*size, nrow*size));
            plt.subplots_adjust(wspace=None, hspace=None);
            for i, field_id in enumerate(idx):
                ax = fig.add_subplot(nrow, ncol, i+1);
                pcm = ax.pcolormesh(self.X, self.Y, self.fields[field_id], cmap=cmap);
                ax.set_title('#{0}: {1:.2f}Hz'.format(field_id, self.fields[field_id].max()))
                ax.set_xticks([])
                ax.set_yticks([])
                if nspks is not None:
                    ax.set_xlabel('{} spikes'.format(nspks[i]))
                if marker:
                    ax.plot(self.firing_pos_dict[field_id][:,0], self.firing_pos_dict[field_id][:,1], 
                                                              'mo', markersize=markersize, alpha=alpha)
                ax.set_aspect(self.maze_ratio)
            plt.grid(False)
            plt.show();

        return fig


    def plot_field(self, i=0, cmap=None, alpha=.3, markersize=10, markercolor='#66f456', trajectory=True):
        '''
        plot ith place field with information in detail, only called after `pc.get_fields(pc.spk_time_dict, rank=True)`
        example:

        @interact(i=(0, pc.n_units-1, 1))
        def view_fields(i=0):
            pc.plot_field(i)

        '''
        if cmap is None:
            cmap = sns.cubehelix_palette(as_cmap=True, dark=0.05, light=1.2, reverse=True);
        neuron_id = self.sorted_fields_id[i]
        self._get_field(self.spk_time_dict[neuron_id])
        f,ax = self._plot_field(cmap=cmap, alpha=alpha, markersize=markersize, 
                         markercolor=markercolor, trajectory=trajectory);
        n_bits = self.metric['spatial_bit_spike'][neuron_id]
        p_rate = self.metric['peak_rate'][neuron_id]
        ax.set_title('neuron {0}: max firing rate {1:.2f}Hz, {2:.3f} bits'.format(neuron_id, p_rate, n_bits))
        return f,ax


    def rank_fields(self, metric_name):
        '''
        metric_name: spatial_bit_spike, spatial_bit_smoothed_spike, spatial_sparcity
        '''
        self.metric = {}
        self.metric['peak_rate'] = np.zeros((self.n_fields,))
        self.metric['spatial_bit_spike'] = np.zeros((self.n_fields,))
        self.metric['spatial_bit_smoothed_spike'] = np.zeros((self.n_fields,))
        self.metric['spatial_sparcity'] = np.zeros((self.n_fields,))

        for neuron_id in range(self.fields.shape[0]):
            self.metric['peak_rate'][neuron_id] = self.fields[neuron_id].max()
            self.metric['spatial_bit_spike'][neuron_id] = info_bits(self.fields[neuron_id], self.P) 
            self.metric['spatial_bit_smoothed_spike'][neuron_id] = info_bits(self.fields[neuron_id], self.P)
            self.metric['spatial_sparcity'][neuron_id] = info_sparcity(self.fields[neuron_id], self.P)

        self.sorted_fields_id = np.argsort(self.metric[metric_name])[::-1]


    def raster(self, ls, colorful=False, xlim=None, ylim=None):
        color_list = ['C{}'.format(i) for i in range(self.n_units)]
        fig, ax = plt.subplots(1,1, figsize=(15,10));
        if colorful:
            ax.eventplot(positions=self.spk_time_array, colors=color_list, ls=ls, alpha=.2);
        else:
            ax.eventplot(positions=self.spk_time_array, colors='k', ls=ls, alpha=.2);
        if xlim is not None:
            ax.set_xlim(xlim);
        if ylim is not None:
            ax.set_ylim(ylim);
        ax.set_ylabel('unit')
        ax.set_xlabel('time (secs)')
        sns.despine()
        return fig


    def load_spkdf(self, df_file, fs=25000., replay_offset=0, show=False):
        '''
        core function: load spike dataframe in spktag folder (to get Spikes)
        This function also align ephys with behavior and compute the place fields of each found units in the `df_file`

        Example:
        ------------
        pc = place_field(pos=pos, ts=ts)
        pc.load_spkdf(spktag_file_df)
        pc.report()
        '''
        print('--------------- place cell object: load spktag dataframe ---------------\r\n')
        # try:
        self.spike_df = pd.read_pickle(df_file)
        self.spike_df['frame_id'] /= fs
        self.spike_df.set_index('spike_id', inplace=True)
        self.spike_df.index = self.spike_df.index.astype(int)
        self.spike_df.index -= self.spike_df.index.min()
        self.spike_df.index.name = 'spike_id'
        self.df['spk'] = self.spike_df
        self.spk_time_dict = {i: self.spike_df.loc[i]['frame_id'].to_numpy() 
                              for i in self.spike_df.index.unique().sort_values()}
        self.df['spk'].reset_index(inplace=True)
        self.n_units = np.sort(self.spike_df.spike_id.unique()).shape[0]
        self.n_groups = np.sort(self.spike_df.group_id.unique()).shape[0]
        print('1. Load the spktag dataframe\r\n    {} units are found in {} electrode-groups\r\n'.format(self.n_units, self.n_groups))
        # except:
            # print('! Fail to load spike dataframe')

        start, end = self.spike_df.frame_id.iloc[0], self.spike_df.frame_id.iloc[-1]
        self.align_with_recording(start, end, replay_offset)
        # after align_with_recording we have the correct self.ts and self.pos
        self.total_spike = len(self.spike_df)
        self.total_time = self.ts[-1] - self.ts[0]
        self.mean_mua_firing_rate = self.total_spike/self.total_time

        print('2. Align the behavior and ephys data with {0} offset\r\n    starting at {1:.3f} secs, end at {2:.3f} secs, step at {3:.3f} ms\r\n    all units mount up to {4:.3f} spikes/sec\r\n'.format(replay_offset, start, end, self.dt*1e3, self.mean_mua_firing_rate))

        print('3. Calculate the place field during [{},{}] secs\r\n    spatially bin the maze, calculate speed and occupation_map with {}cm bin_size\r\n    dump spikes when speed is lower than {}cm/secs\r\n'.format(start, end, self.bin_size, self.v_cutoff))                      
        self.initialize(bin_size=self.bin_size, v_cutoff=self.v_cutoff)
        self.get_fields(self.spk_time_dict, rank=True)

        try:
            self.df['spk']['x'] = np.interp(self.df['spk']['frame_id'], self.ts, self.pos[:,0])
            self.df['spk']['y'] = np.interp(self.df['spk']['frame_id'], self.ts, self.pos[:,1])
            self.df['spk']['v'] = np.interp(self.df['spk']['frame_id'], self.ts, self.v_smoothed)
            print('4. Interpolate the position and speed to each spikes, check `pc.spike_df`\r\n')
        except:
            print('! Fail to fill the position and speed to the spike dataframe')
        if show is True:
            self.field_fig = self.plot_fields();     
        print('------------------------------------------------------------------------')   


    def report(self, cmap='hot', order=False):
        print('occupation map from {0:.2f} to {1:.2f}, with speed cutoff:{2:.2f}'.format(self.ts[0], self.ts[-1], self.v_cutoff))
        self.plot_occupation_map();
        self.plot_speed(self.ts[0], self.ts[-1]//10, v_cutoff=self.v_cutoff);
        self.plot_fields(N=10, cmap=cmap, order=order);


    def load_spktag(self, spktag_file, show=False):
        '''
        1. load spktag
        2. extract unit time stamps
        3. calculate the place fields
        4. rank based on its information bit
        5. (optional) plot place fields of each unit
        check pc.n_units, pc.n_fields and pc.metric after this
        '''
        spktag = SPKTAG()
        spktag.load(spktag_file)
        self.spktag_file = spktag_file
        self.spk_time_array, self.spk_time_dict = spktag.spk_time_array, spktag.spk_time_dict
        self.get_fields(self.spk_time_dict, rank=True)
        if show is True:
            self.field_fig = self.plot_fields();


    def get_scv(self, t_window):
        '''
        The offline binner to calculate the spike count vector (scv)
        run `pc.load_spktag(spktag_file)` first
        t_window is the window to count spikes
        t_step defines the sliding window size
        '''
        # if t_step is None:
        self.scv = spk_time_to_scv(self.spk_time_dict, t_window=t_window, ts=self.ts)
        self.mua_count = self.scv.sum(axis=1)
        # scv = scv[self.sorted_fields_id]
        return self.scv
        # else:
        #     new_ts = np.arange(self.t_start, self.t_end, t_step)
        #     scv = spk_time_to_scv(self.spk_time_dict, delta_t=t_window, ts=new_ts)
        #     # scv = scv[self.sorted_fields_id]
        #     x, y = interp1d(self.ts, self.pos[:,0], fill_value="extrapolate"), interp1d(self.ts, self.pos[:,1], fill_value="extrapolate")
        #     new_pos = np.hstack((x(new_ts).reshape(-1,1), y(new_ts).reshape(-1,1))) 
        #     return scv, new_ts, new_pos


    def plot_epoch(self, time_range, figsize=(5,5), marker=['ro', 'wo'], markersize=15, alpha=.5, cmap=None, legend_loc=None):
        '''
        plot trajactory within time_range: [[a0,b0],[a1,b1]...]
        with color code indicate the speed.  
        '''
        
        gs = dict(height_ratios=[20,1])
        fig, ax = plt.subplots(2,1,figsize=(5, 5), gridspec_kw=gs)

        for i, _time_range in enumerate(time_range):  # ith epoches
            epoch = np.where((self.ts<_time_range[1]) & (self.ts>=_time_range[0]))[0]
            
            if cmap is None:
                cmap = mpl.cm.cool
            norm = mpl.colors.Normalize(vmin=self.v_smoothed.min(), vmax=self.v_smoothed.max())

            ax[0] = colorline(x=self.pos[epoch, 0], y=self.pos[epoch, 1], 
                              z=self.v_smoothed[epoch]/self.v_smoothed.max(), #[0,1] 
                              cmap=cmap, ax=ax[0])
            if i ==0:
                ax[0].plot(self.pos[epoch[-1], 0], self.pos[epoch[-1], 1], marker[0], markersize=markersize, alpha=alpha, label='end')
                ax[0].plot(self.pos[epoch[0], 0], self.pos[epoch[0], 1], marker[1], markersize=markersize, alpha=alpha, label='start')
            else:
                ax[0].plot(self.pos[epoch[-1], 0], self.pos[epoch[-1], 1], marker[0], markersize=markersize, alpha=alpha)
                ax[0].plot(self.pos[epoch[0], 0], self.pos[epoch[0], 1], marker[1], markersize=markersize, alpha=alpha)                

        ax[0].set_xlim(self.maze_range[0]);
        ax[0].set_ylim(self.maze_range[1]);

        # ax[0].set_title('trajectory in [{0:.2f},{1:.2f}] secs'.format(_time_range[0], _time_range[1]))
        if legend_loc is not None:
            ax[0].legend(loc=legend_loc)
        
        cb = mpl.colorbar.ColorbarBase(ax[1], cmap=cmap,
                                        norm=norm,
                                        orientation='horizontal')
        cb.set_label('speed (cm/sec)')
        return ax


    def to_file(self, filename):
        df_all_in_one = pd.concat([self.pos_df, self.spike_df], sort=True)
        df_all_in_one.to_pickle(filename+'.pd')


    def to_dec(self, t_step, t_window, type='bayesian', t_smooth=2, **kwargs):
        '''
        kwargs example:
        - training_range: [0, 0.5]
        - valid_range: [0.5, 0.7]
        - testing_range: [0.7, 1.0]
        - low_speed_cutoff: {'training': True, 'testing': True}
        '''
        if type == 'bayesian':
            from spiketag.analysis import NaiveBayes
            dec = NaiveBayes(t_step=t_step, t_window=t_window)
            dec.connect_to(self)
            dec.resample(t_step=t_step, t_window=t_window)
            training_range = kwargs['training_range'] if 'training_range' in kwargs.keys() else [0.0, 1.0]
            valid_range    = kwargs['training_range'] if 'valid_range'    in kwargs.keys() else [0.0, 1.0]
            testing_range  = kwargs['training_range'] if 'testing_range'  in kwargs.keys() else [0.0, 1.0]
            low_speed_cutoff = kwargs['low_speed_cutoff'] if 'low_speed_cutoff' in kwargs.keys() else {'training': True, 'testing': True}
            dec.partition(training_range=training_range, valid_range=valid_range, testing_range=testing_range,
                          low_speed_cutoff=low_speed_cutoff)
            dec.drop_neuron([0]) # drop the neuron with id 0 which is noise
            score = dec.score(smooth_sec=t_smooth, remove_first_neuron=False)
            return dec, score

        if type == 'LSTM':
            # TODO
            pass
