from playground.base import logger
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import cm
from scipy.interpolate import interp1d
from .core import spk_time_to_scv, firing_pos_from_scv, smooth
from ..base import SPKTAG



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
    def __init__(self, pos, t_step):
        '''
        resample the trajectory with new time interval
        reinitiallize with a new t_step (dt)
        '''
        ts = np.arange(0, pos.shape[0]*t_step, t_step)
        self.ts, self.pos = ts, pos
        self._ts_restore, self._pos_restore = ts, pos

    def __call__(self, t_step):
        '''
        resample the trajectory with new time interval
        reinitiallize with a new t_step (dt)
        '''
        fs = self.fs 
        new_fs = 1/t_step
        self.ts, self.pos = self.interp_pos(self.ts, self.pos, N=fs/new_fs)

    def restore(self):
        self.ts, self.pos = self._ts_restore, self._pos_restore

    @property
    def fs(self):
        self._fs = 1/(self.ts[1]-self.ts[0])
        return self._fs

    def interp_pos(self, t, pos, N=1):
        '''
        convert irregularly sampled pos into regularly sampled pos
        N is the dilution sampling factor. N=2 means half of the resampled pos
        example:
        >>> new_fs = 200.
        >>> pc.ts, pc.pos = pc.interp_pos(ts, pos, N=fs/new_fs)
        '''
        dt = np.mean(np.diff(t))
        x, y = interp1d(t, pos[:,0], fill_value="extrapolate"), interp1d(t, pos[:,1], fill_value="extrapolate")
        new_t = np.arange(0.0, dt*len(t), dt*N)
        new_pos = np.hstack((x(new_t).reshape(-1,1), y(new_t).reshape(-1,1)))
        return new_t, new_pos 


    def align_with_recording(self, recording_start_time, recording_end_time, replay_offset=0):
        '''
        ts before alignment   |--------------------|
        behavior  start:      |
        behavior    end:                           |
        recording start:         |------------
        recording   end:          ------------|
        replay_offset  :             |
        ts after alignment           |------------| 
        '''
        self.ts += replay_offset   # 0 if the ephys is not offset by replaying through neural signal generator
        self.pos = self.pos[np.logical_and(self.ts>recording_start_time, self.ts<recording_end_time)]
        self.ts  =  self.ts[np.logical_and(self.ts>recording_start_time, self.ts<recording_end_time)]
        self.t_start = self.ts[0]
        self.t_end   = self.ts[-1]
        self._ts_restore, self._pos_restore = self.ts, self.pos


    def load_log(self, logfile=None, session_id=0, v_cutoff=5, maze_range=[[-100,100], [-100,100]], bin_size=4, sync=True):
        '''
        """getting the place fields from a log"""
        # default mode:
        # 1: ts, pos, dt 
        # 2: v_smoothed, v_cutoff
        # 3: maze_range, bin_size 
        '''
        if logfile is None:
            pass
             
        # logfile mode
        else:
            self.logfile = logfile
            self.log = logger(self.logfile, sync=sync)
            self.ts, self.pos = self.log.to_trajectory(session_id)
            self.pos[:,1] = -self.pos[:,1]
            

            # self.v_smoothed, self.v = self.log.get_speed(self.ts, self.pos, smooth_window=60, std=15)
            # self.v_cutoff = v_cutoff
            # self.get_still_idx() 

            # self.get_speed(smooth_window=60, std=15, v_cutoff=5) 
            # self.maze_range = maze_range
            # self.occupation_map(bin_size)
            self.initialize(bin_size=bin_size, v_cutoff=v_cutoff, maze_range=maze_range)

        ### place fields parameters ###
        self.n_fields = 0


    def initialize(self, bin_size, v_cutoff, maze_range=None):
        self.dt = self.ts[1] - self.ts[0]
        self.get_maze_range(maze_range)
        self.get_speed(smooth_window=60, std=15, v_cutoff=v_cutoff) 
        self.occupation_map(bin_size)
        self.binned_pos = smooth((self.pos-self.maze_original)//self.bin_size, 3)


    def get_maze_range(self, maze_range=None):
        if maze_range is None:
            self.maze_range = np.vstack((self.pos.min(axis=0), self.pos.max(axis=0))).T
            self._maze_original = self.maze_range[:,0] # the left, down corner location
        else:
            self.maze_range = np.array(maze_range)
            self._maze_original = self.maze_range[:,0] # the left, down corner location

    @property
    def maze_original(self):
        return self._maze_original

    def binned_pos_2_real_pos(self, binned_pos):
        pos = binned_pos*self.bin_size + self.maze_original
        return pos

    def get_speed(self, smooth_window=59, std=6, v_cutoff=5):
        '''
        self.ts, self.pos is required
        '''
        v = np.linalg.norm(np.diff(self.pos, axis=0), axis=1)/np.diff(self.ts)
        w = signal.gaussian(smooth_window, std) # window size 59 frame (roughly 1 sec), std = 6 frame
        w /= sum(w)
        v_smoothed = np.convolve(v, w, mode='same')

        self.v = np.hstack((0.01, v))
        self.v_smoothed = np.hstack((0.01, v_smoothed))
        self.v_cutoff   = v_cutoff
        self.low_speed_idx = np.where(self.v_smoothed < self.v_cutoff)[0]
        '''
        # check speed:
        f, ax = plt.subplots(1,1, figsize=(18,8))
        offset=20000
        plot(ts[offset:1000+offset], v[offset:1000+offset])
        plot(ts[offset:1000+offset], v_smoothed[offset:1000+offset])
        ax.axhline(5, c='m', ls='-.')
        '''
        # return v_smoothed, v

    def plot_speed(self, start, stop, thres=5):
        fig, ax = plt.subplots(1,1, figsize=(18,8))
        period = np.logical_and(self.ts>start, self.ts<stop)
        plt.plot(self.ts[period], self.v[period], alpha=.7)
        plt.plot(self.ts[period], self.v_smoothed[period], lw=3)
        ax.axhline(thres, c='m', ls='-.')
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


    def plot_occupation_map(self):
        f, ax = plt.subplots(1,2,figsize=(20,9))
        ax[0].plot(self.pos[:,0], self.pos[:,1])
        ax[0].plot(self.pos[0,0], self.pos[0,1], 'ro')
        ax[0].plot(self.pos[-1,0], self.pos[-1,1], 'go')
        ax[0].pcolormesh(self.X, self.Y, self.O, cmap=cm.hot)
        # sns.heatmap(self.O[::-1]*self.dt, annot=False, cbar=False, ax=ax[1])
        ax[1].pcolormesh(self.X, self.Y, self.O, cmap=cm.hot)
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
        spk_ts = np.searchsorted(self.ts, spk_times) - 1
        idx = np.setdiff1d(spk_ts, self.low_speed_idx)
        self.firing_ts  = self.ts[spk_ts] #[:,1]
        self.firing_pos = self.pos[idx]        
        self.firing_map, x_edges, y_edges = np.histogram2d(x=self.firing_pos[:,0], y=self.firing_pos[:,1], 
                                                           bins=self.nbins, range=self.maze_range)
        self.firing_map = self.firing_map.T
        np.seterr(divide='ignore', invalid='ignore')
        self.FR = self.firing_map/self.O/self.dt
        # self.FR = np.nan_to_num(self.FR)
        self.FR[np.isnan(self.FR)] = 0
        self.FR[np.isinf(self.FR)] = 0
        self.FR_smoothed = signal.convolve2d(self.FR, self.gkern(self.kernlen, self.kernstd), boundary='symm', mode='same')
        return self.FR_smoothed


    def firing_map_from_scv(self, scv, section=[0,1]):
        '''
        firing heat map constructed from spike count vector (scv) and position
        '''
        # assert(scv.shape[1]==self.pos.shape[0])
        n_neurons, total_bin = scv.shape
        valid_bin = np.array(np.array(section)*total_bin, dtype=np.int)
        firing_map_smoothed = np.zeros((n_neurons, *self.map_binned_size))
        for neuron_id in range(n_neurons):
            firing_pos = firing_pos_from_scv(scv, self.pos, neuron_id, valid_bin)
            firing_map, x_edges, y_edges = np.histogram2d(x=firing_pos[:,0], y=firing_pos[:,1], 
                                                          bins=self.nbins, range=self.maze_range)
            firing_map = firing_map.T/self.O/self.dt
            firing_map[np.isnan(firing_map)] = 0
            firing_map[np.isinf(firing_map)] = 0
            firing_map_smoothed[neuron_id] = signal.convolve2d(firing_map, self.gkern(self.kernlen, self.kernstd), boundary='symm', mode='same')
            firing_map_smoothed[firing_map_smoothed==0] = 1e-25
        return firing_map_smoothed        


    def get_field(self, spk_time_dict, neuron_id, start=None, end=None):
        '''
        f, ax = plt.subplots(1,2,figsize=(20,9))
        ax[0].plot(self.pos[:,0], self.pos[:,1])
        ax[0].plot(self.firing_pos[:,0], self.firing_pos[:,1], 'mo', alpha=0.5)
        # ax[0].pcolormesh(self.X, self.Y, self.FR, cmap=cm.hot)
        pc = ax[1].pcolormesh(X, Y, FR_GAU, cmap=cm.hot)
        colorbar(pc, ax=ax[1], label='Hz') 
        '''
        spk_times = spk_time_dict[neuron_id]
        ### for cross-validation and field stability check
        ### calculate representation from `start` to `end`
        if start is not None and end is not None:
            spk_times = spk_times[np.logical_and(start<=spk_times, spk_times<end)]
        self._get_field(spk_times)


    def plot_field(self, trajectory=False, cmap='gray', marker=True, alpha=0.5, markersize=5):
        f, ax = plt.subplots(1,1,figsize=(13,10));
        pcm = ax.pcolormesh(self.X, self.Y, self.FR_smoothed, cmap=cmap);
        plt.colorbar(pcm, ax=ax, label='Hz');
        if trajectory:
            ax.plot(self.pos[:,0], self.pos[:,1], alpha=0.8);
            ax.plot(self.pos[0,0], self.pos[0,1], 'ro');
            ax.plot(self.pos[-1,0],self.pos[-1,1], 'ko');
        if marker:
            ax.plot(self.firing_pos[:,0], self.firing_pos[:,1], 'mo', alpha=alpha, markersize=markersize);
        return f,ax


    def get_fields(self, spk_time_dict, start=None, end=None, rank=True):
        '''
        spk_time_dict is dictionary start from 1: {1: ... 2: ... 3: ...}
        '''
        self.n_fields = len(spk_time_dict.keys())
        self.n_units  = self.n_fields
        self.fields = {}
        self.fields_matrix = np.zeros((self.n_fields, self.O.shape[0], self.O.shape[1]))

        self.metric = {}
        self.metric['spatial_bit_spike'] = np.zeros((self.n_fields,))
        self.metric['spatial_bit_smoothed_spike'] = np.zeros((self.n_fields,))
        self.metric['spatial_sparcity'] = np.zeros((self.n_fields,))

        self.firing_poshd = {}

        for i in spk_time_dict.keys():
            ### get place fields from neuron i
            self.get_field(spk_time_dict, i, start, end)
            self.fields[i] = self.FR_smoothed
            self.firing_poshd[i] = self.firing_pos
            self.fields_matrix[i] = self.FR_smoothed
            ### metrics for place fields
            self.metric['spatial_bit_spike'][i] = info_bits(self.FR, self.P) 
            self.metric['spatial_bit_smoothed_spike'][i] = info_bits(self.FR_smoothed, self.P)
            self.metric['spatial_sparcity'][i] = info_sparcity(self.FR, self.P)

        self.fields_matrix[self.fields_matrix==0] = 1e-25

        if rank is True:
            self.rank_fields('spatial_bit_smoothed_spike')


    def rank_fields(self, metric):
        '''
        metric: spatial_bit_spike, spatial_bit_smoothed_spike, spatial_sparcity
        '''
        self.sorted_fields_id = np.argsort(self.metric[metric])[::-1]


    def plot_fields(self, N=12, size=3, cmap='hot', marker=False, markersize=1, alpha=0.8, order=True):
        '''
        order: if True will plot with ranked fields according to the metric 
        '''
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
            if marker:
                ax.plot(self.firing_poshd[field_id][:,0], self.firing_poshd[field_id][:,1], 
                                                          'mo', markersize=markersize, alpha=alpha)

        plt.grid('off')
        plt.show();
        return fig


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
