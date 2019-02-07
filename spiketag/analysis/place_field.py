from playground.base import logger
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.interpolate import interp1d



def info_bits(Fr, P):
    Fr[Fr==0] = 1e-25
    MFr = sum(P.ravel()*Fr.ravel())
    return sum(P.ravel()*(Fr.ravel()/MFr)*np.log2(Fr.ravel()/MFr))


def info_sparcity(Fr, P):
    Fr[Fr==0] = 1e-25
    MFr = sum(P.ravel()*Fr.ravel())
    return sum(P.ravel()*Fr.ravel()**2/MFr**2)


class place_field(object):
    """getting the place fields from subspaces"""
    def __init__(self, logfile=None, session_id=0, v_cutoff=5, maze_range=[[-100,100], [-100,100]], bin_size=4, sync=True):
        super(place_field, self).__init__()
        # default mode:
        # 1: ts, pos, dt 
        # 2: v_smoothed, v_cutoff
        # 3: maze_range, bin_size 
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
        self.get_speed(smooth_window=60, std=15, v_cutoff=5) 
        self.occupation_map(bin_size)


    def interp_pos(self, t, pos, N=1):
        '''
        convert irregularly sampled pos into regularly sampled pos
        N is the dilution sampling factor. N=2 means half of the resampled pos
        '''
        dt = np.mean(np.diff(t))
        x, y = interp1d(t, pos[:,0], fill_value="extrapolate"), interp1d(t, pos[:,1], fill_value="extrapolate")
        new_t = np.arange(0.0, dt*len(t), dt*N)
        new_pos = np.hstack((x(new_t).reshape(-1,1), y(new_t).reshape(-1,1)))
        return new_t, new_pos 
        

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
        self.v_still_idx = np.where(self.v_smoothed < self.v_cutoff)[0]
        '''
        # check speed:
        f, ax = plt.subplots(1,1, figsize=(18,8))
        offset=20000
        plot(ts[offset:1000+offset], v[offset:1000+offset])
        plot(ts[offset:1000+offset], v_smoothed[offset:1000+offset])
        ax.axhline(5, c='m', ls='-.')
        '''
        # return v_smoothed, v
        

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


    @staticmethod
    def gkern(kernlen=21, std=2):
        """Returns a 2D Gaussian kernel array."""
        gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
        gkern2d = np.outer(gkern1d, gkern1d)
        gkern2d /= gkern2d.sum()
        return gkern2d


    def _get_field(self, spk_times):
        spk_tw = np.searchsorted(self.ts, spk_times) - 1
        idx = np.setdiff1d(spk_tw, self.v_still_idx)
        self.firing_ts  = self.ts[spk_tw] #[:,1]
        self.firing_pos = self.pos[idx]        
        self.firing_map, x_edges, y_edges = np.histogram2d(x=self.firing_pos[:,0], y=self.firing_pos[:,1], 
                                                           bins=self.nbins, range=self.maze_range)
        self.firing_map = self.firing_map.T
        np.seterr(divide='ignore', invalid='ignore')
        self.FR = self.firing_map/self.O/self.dt
        self.FR = np.nan_to_num(self.FR)
        self.FR_smoothed = signal.convolve2d(self.FR, self.gkern(self.kernlen, self.kernstd), boundary='symm', mode='same')
        return self.FR_smoothed


    def get_field(self, spk_time_dict, neuron_id):
        '''
        f, ax = plt.subplots(1,2,figsize=(20,9))
        ax[0].plot(self.pos[:,0], self.pos[:,1])
        ax[0].plot(self.firing_pos[:,0], self.firing_pos[:,1], 'mo', alpha=0.5)
        # ax[0].pcolormesh(self.X, self.Y, self.FR, cmap=cm.hot)
        pc = ax[1].pcolormesh(X, Y, FR_GAU, cmap=cm.hot)
        colorbar(pc, ax=ax[1], label='Hz') 
        '''
        self._get_field(spk_times=spk_time_dict[neuron_id])

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



    def get_fields(self, spk_time_dict):
        '''
        spk_time_dict is dictionary start from 1: {1: ... 2: ... 3: ...}
        '''
        self.n_fields = len(spk_time_dict.keys())
        self.fields = {}
        self.fields_matrix = np.zeros((self.n_fields, self.O.shape[0], self.O.shape[1]))

        self.metric = {}
        self.metric['spatial_bit_spike'] = np.zeros((self.n_fields,))
        self.metric['spatial_bit_smoothed_spike'] = np.zeros((self.n_fields,))
        self.metric['spatial_sparcity'] = np.zeros((self.n_fields,))

        self.firing_poshd = {}

        for i in spk_time_dict.keys():
            ### get place fields from neuron i
            self.get_field(spk_time_dict, i)
            self.fields[i] = self.FR_smoothed
            self.firing_poshd[i] = self.firing_pos
            self.fields_matrix[i] = self.FR_smoothed
            ### metrics for place fields
            self.metric['spatial_bit_spike'][i] = info_bits(self.FR, self.P) 
            self.metric['spatial_bit_smoothed_spike'][i] = info_bits(self.FR_smoothed, self.P)
            self.metric['spatial_sparcity'][i] = info_sparcity(self.FR, self.P)

        self.fields_matrix[self.fields_matrix==0] = 1e-25


    def rank_fields(self, metric):
        '''
        metric: spatial_bit_spike, spatial_bit_smoothed_spike, spatial_sparcity
        '''
        self.sorted_fields_id = np.argsort(self.metric[metric])[::-1]


    def plot_fields(self, N, size=1.8, cmap='gray', marker=True, markersize=1, alpha=0.5, order=True):
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




