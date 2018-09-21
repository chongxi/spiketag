from playground.base import logger
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


class place_field(object):
    """getting the place fields from subspaces"""
    def __init__(self, logfile, session_id=0, v_cutoff=5, maze_range=[[-100,100], [-100,100]], bin_size=4, sync=True):
        super(place_field, self).__init__()
        self.logfile = logfile
        self.log = logger(self.logfile, sync=sync)
        self.ts, self.pos = self.log.to_trajectory(session_id)
        self.dt = self.ts[1] - self.ts[0]
        self.v_smoothed, self.v = self.log.get_speed(self.ts, self.pos, smooth_window=60, std=15)
        self.v_cutoff = v_cutoff
        self.v_still_idx = np.where(self.v_smoothed < self.v_cutoff)[0]
        self.occupation_map(maze_range, bin_size)
        

    def occupation_map(self, maze_range=[[-100,100], [-100,100]], bin_size=4, time_cutoff=None):
        '''
        f, ax = plt.subplots(1,2,figsize=(20,9))
        ax[0].plot(self.pos[:,0], self.pos[:,1])
        ax[0].plot(self.pos[0,0], self.pos[0,1], 'ro')
        ax[0].plot(self.pos[-1,0], self.pos[-1,1], 'ko')
        ax[0].pcolormesh(self.X, self.Y, self.O, cmap=cm.hot_r)
        sns.heatmap(self.O[::-1]*self.dt, annot=False, cbar=False, ax=ax[1])
        '''
        self.maze_range = maze_range
        self.maze_size = np.array([maze_range[0][1]-maze_range[0][0], maze_range[1][1]-maze_range[1][0]])
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


    def plot_occupation_map(self):
        f, ax = plt.subplots(1,2,figsize=(20,9))
        ax[0].plot(self.pos[:,0], self.pos[:,1])
        ax[0].plot(self.pos[0,0], self.pos[0,1], 'ro')
        ax[0].plot(self.pos[-1,0], self.pos[-1,1], 'go')
        ax[0].pcolormesh(self.X, self.Y, self.O, cmap=cm.hot)
        # sns.heatmap(self.O[::-1]*self.dt, annot=False, cbar=False, ax=ax[1])
        ax[1].pcolormesh(self.X, self.Y, self.O, cmap=cm.hot)


    @staticmethod
    def gkern(kernlen=21, std=2):
        """Returns a 2D Gaussian kernel array."""
        gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
        gkern2d = np.outer(gkern1d, gkern1d)
        gkern2d /= gkern2d.sum()
        return gkern2d


    def get_field(self, spk_time, neuron_id, kernlen=21, std=3):
        '''
        f, ax = plt.subplots(1,2,figsize=(20,9))
        ax[0].plot(self.pos[:,0], self.pos[:,1])
        ax[0].plot(self.firing_pos[:,0], self.firing_pos[:,1], 'mo', alpha=0.5)
        # ax[0].pcolormesh(self.X, self.Y, self.FR, cmap=cm.hot)
        pc = ax[1].pcolormesh(X, Y, FR_GAU, cmap=cm.hot)
        colorbar(pc, ax=ax[1], label='Hz') 
        '''
        spk_tw = np.searchsorted(self.ts, spk_time[neuron_id]) - 1
        # spk_tw = np.vstack((spk_tw-1, spk_tw)).T
        # idx = np.delete(spk_tw, self.v_still_idx)
        idx = np.setdiff1d(spk_tw, self.v_still_idx)
        self.firing_ts  = self.ts[spk_tw] #[:,1]
        self.firing_pos = self.pos[idx]
        self.firing_map, x_edges, y_edges = np.histogram2d(x=self.firing_pos[:,0], y=self.firing_pos[:,1], 
                                                           bins=self.nbins, range=self.maze_range)
        self.firing_map = self.firing_map.T
        # if self.firing_map.sum() == 0:
        #     print('no firing when animal move under 5cm/s')
        np.seterr(divide='ignore', invalid='ignore')
        self.FR = self.firing_map/self.O/self.dt
        self.FR = np.nan_to_num(self.FR)
        self.FR_smoothed = signal.convolve2d(self.FR, self.gkern(kernlen, std), boundary='symm', mode='same')
        # self.FR_smoothed = self.FR_smoothed[::-1]


    def plot_field(self, trajectory=False):
        f, ax = plt.subplots(1,1,figsize=(13,10));
        ax.plot(self.pos[:,0], self.pos[:,1], alpha=0.8);
        ax.plot(self.pos[0,0], self.pos[0,1], 'ro');
        ax.plot(self.pos[-1,0],self.pos[-1,1], 'ko');
        ax.plot(self.firing_pos[:,0], self.firing_pos[:,1], 'mo', alpha=0.5);
        pcm = ax.pcolormesh(self.X, self.Y, self.FR_smoothed, cmap=cm.hot);
        plt.colorbar(pcm, ax=ax, label='Hz');


    def get_fields(self, spk_time, kernlen=21, std=3):
        self.fields = {}
        for i in spk_time.keys():
            self.get_field(spk_time, i, kernlen, std)
            self.fields[i] = self.FR_smoothed

    def plot_fields(self, N, size=1.8):
        cluNo = self.fields.keys()[-1]
        nrow = cluNo/N+1
        ncol = N
        fig = plt.figure(figsize=(ncol*size, nrow*size));
        plt.tight_layout();
        plt.subplots_adjust(wspace=None, hspace=None);
        for i in self.fields.keys():
            if i != 0:
                ax = fig.add_subplot(nrow, ncol, i);
                pcm = ax.pcolormesh(self.X, self.Y, self.fields[i], cmap=cm.hot);
        plt.show();
