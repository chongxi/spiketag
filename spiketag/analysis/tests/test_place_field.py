import numpy as np
from scipy.io import loadmat
from spiketag.analysis import place_field

pos_data = loadmat('./rec02-whl_fix.mat')['whlDataLfp']
pos = pos_data[:,-2:]

pc = place_field()
fs = 1250.
ts = np.arange(0, pos.shape[0]/fs, 1/fs)
new_fs = 60.
pc.ts, pc.pos = pc.interp_pos(ts, pos, N=fs/new_fs)

ephys_start_time = 1.461
ephys_end_time = 2462
pc.ts = pc.ts + ephys_start_time
pc.ts, pc.pos = pc.ts[pc.ts<ephys_end_time], pc.pos[pc.ts<ephys_end_time]

bin_size = 5
v_cutoff = 15
pc.initialize(bin_size=bin_size, v_cutoff=v_cutoff)  #, maze_range=np.array([[100,500], [100,500]])
# pc.plot_occupation_map()


spk_time_full = np.load('./dusty_spk_time_full.npy').item()
pc.get_fields(spk_time_full, kernlen=14, std=3)
pc.rank_fields('spatial_bit_smoothed_spike')

pc.plot_fields(N=16, size=2, cmap='hot', 
                 marker=False, markersize=1, alpha=0.9, order=True);

