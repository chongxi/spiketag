from sklearn.covariance import log_likelihood
from .core import softmax, licomb_Matrix, bayesian_decoding, argmax_2d_tensor, smooth
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from ..utils import plot_err_2d
from ..analysis import sliding_window_to_feature, FA
from .manifold import spike_noise_bernoulli, spike_noise_gaussian
import copy
import torch
from torch import nn
import torch.nn.functional as F

def gaussian_inhibition_field(coord, size_x=32, size_y=32):
    '''
    Genearte a gassian inhibition field with center at coord and size of size_x by size_y
    for two-step bayesian decoding: https://journals.physiology.org/doi/epdf/10.1152/jn.1998.79.2.1017 (eq. 43)
    - ||x_last - x||^2 
    '''
    xy  = np.stack(np.meshgrid(np.arange(0,size_x), np.arange(0,size_y)), axis=-1)
    coord_xy = np.ones((size_x,size_y,2)) * np.array(coord)
    constraint_field = coord_xy - xy
    return -np.linalg.norm(constraint_field, axis=-1)**2

def mua_count_cut_off(X, y=None, minimum_spikes=1):
    '''
    temporary solution to cut the frame that too few spikes happen

    X is the spike count vector(scv), (B_bins, N_neurons), the count in each bin is result from (t_window, t_step)
    minimum_spikes is the minimum number of spikes that allow the `bins`(rows) enter into the decoder
    '''
    for i in range(100):  # each iteration some low rate bins is removed
        mua_count = X.sum(axis=1) # sum over all neuron to get mua
        idx = np.where(mua_count<=minimum_spikes)[0]
        X[idx] = X[idx-1]
        if y is not None:
            y[idx] = y[idx-1]
    return X, y


def load_decoder(filename):
    # step 1: load the decoder from file
    dec = torch.load(filename)

    # step 2: assign place field to the place decoder
    # call pc.get_fields() first to update pc.fields and then transfer to dec.fields
    dec.pc.get_fields() # !critical to update pc.fields using this method first
    dec.fields = dec.pc.fields[1:]  # remove the first field, which is the 'noise'

    # step 3: store some reusable values in the decoder for fast online computing
    # cached value for real-time decoding on incoming bin (N units by B bins) from BMI
    dec.spatial_bin_size, dec.spatial_origin = dec.pc.bin_size, dec.pc.maze_original
    dec.poisson_matrix = dec.t_window*dec.fields.sum(axis=0) # one matrix reused in bayesian decoding
    dec.log_fr = np.log(dec.fields)  # log fields, make sure Fr[Fr==0] = 1e-12

    dec.partition(training_range=[0.0, 1.0], valid_range=[0.0, 1.0], testing_range=[0.0, 1.0], 
                  low_speed_cutoff={'training': True, 'testing': True})
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = dec.get_data(minimum_spikes=2)
    predicted_y = dec.predict(X_test)
    smooth_factor  = int(2/dec.pc.t_step) # 2 second by default
    sm_predicted_y = smooth(predicted_y, smooth_factor)
    score = dec.r2_score(sm_predicted_y, y_test)
    print(f'decoder uses {dec.fields.shape[0]} neurons, R2 score: {score}')
    return dec


class Decoder(object):
    """Base class for the decoders for place prediction"""
    def __init__(self, t_window, t_step=None, verbose=True):
        '''
        t_window is the bin_size
        t_step   is the step_size (if None then use pc.ts as natrual sliding window)
        https://github.com/chongxi/spiketag/issues/47 
        
        For Non-RNN decoder, large bin size in a single bin are required
        For RNN decoder,   small bin size but multiple bins are required

        During certain neural state, such as MUA burst (ripple), a small step size is required 
        (e.g. t_window:20ms, t_step:5ms is used by Pfeiffer and Foster 2013 for trajectory events) 

        dec.partition(training_range, valid_range, testing_range, low_speed_cutoff) 
        serves the cross-validation
        https://github.com/chongxi/spiketag/issues/50
        '''
        self.t_window = t_window
        self.t_step   = t_step
        self.verbose  = verbose

    @property
    def B_bins(self):
        self._b_bins = int(np.round(self.t_window / self.t_step))
        return self._b_bins

    def connect_to(self, pc):
        '''
        This decoder is specialized for position decoding
        Connect to a place-cells object that contains behavior, neural data and co-analysis
        '''
        # self.pc = pc
        self.pc = copy.deepcopy(pc)
        self.pc.rank_fields('spatial_bit_spike') # rerank the field
        self.fields = self.pc.fields
        if self.t_step is not None:
            print('Link the decoder with the place cell object (pc):\r\n resample the pc according to current decoder input sampling rate {0:.4f} Hz'.format(1/self.t_step))
            self.pc(t_step=self.t_step)

    def drop_neuron(self, _disable_neuron_idx):
        if type(_disable_neuron_idx) == int:
            _disable_neuron_idx = [_disable_neuron_idx]
        self._disable_neuron_idx = _disable_neuron_idx
        if self._disable_neuron_idx is not None:
            self.neuron_idx = np.array([_ for _ in range(self.fields.shape[0]) if _ not in self._disable_neuron_idx])

    def resample(self, t_step=None, t_window=None):
        if t_window is None:
            t_window = self.binner.bin_size*self.binner.B
        elif t_window != self.t_window:
            self.t_window = t_window
        if t_step is None:
            t_step = self.binner.bin_size
        elif t_step != self.t_step:
            self.t_step   = t_step
            self.connect_to(self.pc)

    def _percent_to_time(self, percent):
        len_frame = len(self.pc.ts)
        totime = int(np.round((percent * len_frame)))
        if totime < 0: 
            totime = 0
        elif totime > len_frame - 1:
            totime = len_frame - 1
        return totime

    def partition(self, training_range=[0.0, 0.5], valid_range=[0.5, 0.6], testing_range=[0.5, 1.0],
                        low_speed_cutoff={'training': True, 'testing': False}, v_cutoff=None):

        self.train_range = training_range
        self.valid_range = valid_range
        self.test_range  = testing_range
        self.low_speed_cutoff = low_speed_cutoff

        if v_cutoff is None:
            self.v_cutoff = self.pc.v_cutoff
        else:
            self.v_cutoff = v_cutoff

        self.train_time = [self.pc.ts[self._percent_to_time(training_range[0])], 
                           self.pc.ts[self._percent_to_time(training_range[1])]]
        self.valid_time = [self.pc.ts[self._percent_to_time(valid_range[0])], 
                           self.pc.ts[self._percent_to_time(valid_range[1])]]
        self.test_time  = [self.pc.ts[self._percent_to_time(testing_range[0])], 
                           self.pc.ts[self._percent_to_time(testing_range[1])]]

        self.train_idx = np.arange(self._percent_to_time(training_range[0]),
                                   self._percent_to_time(training_range[1]))
        self.valid_idx = np.arange(self._percent_to_time(valid_range[0]),
                                   self._percent_to_time(valid_range[1]))
        self.test_idx  = np.arange(self._percent_to_time(testing_range[0]),
                                   self._percent_to_time(testing_range[1]))

        if low_speed_cutoff['training'] is True:
            self.train_idx = self.train_idx[self.pc.v_smoothed[self.train_idx]>self.v_cutoff]
            self.valid_idx = self.valid_idx[self.pc.v_smoothed[self.valid_idx]>self.v_cutoff]

        if low_speed_cutoff['testing'] is True:
            self.test_idx = self.test_idx[self.pc.v_smoothed[self.test_idx]>self.v_cutoff]

        if self.verbose:
            print('{0} training samples\n{1} validation samples\n{2} testing samples'.format(self.train_idx.shape[0],
                                                                                             self.valid_idx.shape[0],
                                                                                             self.test_idx.shape[0]))

    def save(self, filename):
        torch.save(self, filename)

    def get_data(self, minimum_spikes=2, remove_first_unit=False):
        '''
        Connect to pc first and then set the partition parameter. After these two we can get data
        The data strucutre is different for RNN and non-RNN decoder
        Therefore each decoder subclass has its own get_partitioned_data method
        In low_speed periods, data should be removed from train and valid:
        '''
        assert(self.pc.ts.shape[0] == self.pc.pos.shape[0])

        self.pc.get_scv(self.t_window); # t_step is None unless specified, using pc.ts
        self.pc.output_variables = ['scv', 'pos']
        X, y = self.pc[:]
        assert(X.shape[0]==y.shape[0])

        self.train_X, self.train_y = X[self.train_idx], y[self.train_idx]
        self.valid_X, self.valid_y = X[self.valid_idx], y[self.valid_idx]
        self.test_X,  self.test_y  = X[self.test_idx], y[self.test_idx]

        if minimum_spikes>0:
            self.train_X, self.train_y = mua_count_cut_off(self.train_X, self.train_y, minimum_spikes)
            self.valid_X, self.valid_y = mua_count_cut_off(self.valid_X, self.valid_y, minimum_spikes)
            self.test_X,  self.test_y  = mua_count_cut_off(self.test_X,  self.test_y,  minimum_spikes)

        if remove_first_unit:
            self.train_X = self.train_X[:,1:]
            self.valid_X = self.valid_X[:,1:]
            self.test_X  = self.test_X[:,1:]

        return (self.train_X, self.train_y), (self.valid_X, self.valid_y), (self.test_X, self.test_y) 

    def r2_score(self, y_true, y_predict, multioutput=True):
        '''
        use sklearn.metrics.r2_score(y_true, y_pred, multioutput=True)
        Note: r2_score is not symmetric, r2(y_true, y_pred) != r2(y_pred, y_true)
        '''
        if multioutput is True:
            score = r2_score(y_true, y_predict, multioutput='raw_values')
        else:
            score = r2_score(y_true, y_predict)
        if self.verbose:
            print('r2 score: {}\n'.format(score))
        return score

    def auto_pipeline(self, t_smooth=2, remove_first_unit=False, firing_rate_modulation=True):
        '''
        example for evaluate the funciton of acc[partition]:
        >>> dec = NaiveBayes(t_window=500e-3, t_step=60e-3)
        >>> dec.connect_to(pc)
        >>> r_scores = []
        >>> partition_range = np.arange(0.1, 1, 0.05)
        >>> for i in partition_range:
        >>>     dec.partition(training_range=[0, i], valid_range=[0.5, 0.6], testing_range=[i, 1],
        >>>                   low_speed_cutoff={'training': True, 'testing': True})
        >>>     r_scores.append(dec.auto_pipeline(2))
        '''
        (X_train, y_train), (X_valid, y_valid), (self.X_test, self.y_test) = self.get_data(minimum_spikes=2, 
                                                                                           remove_first_unit=remove_first_unit)
        self.fit(X_train, y_train, remove_first_unit=remove_first_unit)
        self.predicted_y = self.predict(self.X_test, 
                                        firing_rate_modulation=firing_rate_modulation, 
                                        two_steps=False)
        self.smooth_factor  = int(t_smooth/self.pc.t_step) # 2 second by default
        self.sm_predicted_y = smooth(self.predicted_y, self.smooth_factor)
        score = self.r2_score(self.y_test, self.sm_predicted_y) # ! r2 score is not symmetric, needs to be (true, prediction)
        return score

    def score(self, t_smooth=2, remove_first_unit=False, firing_rate_modulation=True):
        '''
        dec.score will first automatically train the decoder (fit) and then test it (predict). 
        The training set and test set are also automatically saved in dec.X_train and dec.X_test
        The training and test label are saved in dec.y_train and dec.y_test
        '''
        return self.auto_pipeline(t_smooth=t_smooth, 
                                  remove_first_unit=remove_first_unit,
                                  firing_rate_modulation=firing_rate_modulation)

    def plot_decoding_err(self, real_pos, dec_pos, err_percentile=90, N=None, err_max=None):
        err = abs(real_pos - dec_pos)
        dt = self.t_step
        if N is None:
            N = err.shape[0]
        return plot_err_2d(real_pos, dec_pos, err, dt, N, err_percentile, err_max)



class NaiveBayes(Decoder):
    """
    NaiveBayes Decoder for position prediction (input X, output y) 
    where X is the spike bin matrix (B_bins, N_neurons)
    where y is the 2D position (x,y)

    Examples:
    -------------------------------------------------------------
    from spiketag.analysis import NaiveBayes, smooth

    dec = NaiveBayes(t_window=250e-3, t_step=50e-3)
    dec.connect_to(pc)

    dec.partition(training_range=[0.0, .7], valid_range=[0.5, 0.6], testing_range=[0.6, 1.0], 
                  low_speed_cutoff={'training': True, 'testing': True})
    (train_X, train_y), (valid_X, valid_y), (test_X, test_y) = dec.get_data(minimum_spikes=0)
    dec.fit(train_X, train_y)

    predicted_y = dec.predict(test_X)
    dec_pos  = smooth(predicted_y, 60)
    real_pos = test_y
    
    score = dec.r2_score(real_pos, dec_pos)

    # optional (cost time):
    # dec.plot_decoding_err(real_pos, dec_pos);

    To get scv matrix to hack (check the size):
    -------------------------------------------------------------
    _scv = dec.pc.scv
    _scv = dec.pc.scv[dec.train_idx]
    _scv = dec.pc.scv[dec.test_idx]
    _y = dec.predict(_scv)
    post2d = dec.post_2d

    Test real-time prediction:
    -------------------------------------------------------------
    _y = dec.predict_rt(_scv[8])

    """
    def __init__(self, t_window, t_step=None):
        super(NaiveBayes, self).__init__(t_window, t_step)
        self.name = 'NaiveBayes'
        self.rt_post_2d, self.binned_pos = None, None  # these two variables can be used for real-time visualization in the playground
        self._disable_neuron_idx = None  # mask out neuron
        self.last_max_bin = None  # real-time prediction binned position (keep last_max_bin for two-step prediction)
        
        
    def fit(self, X=None, y=None, remove_first_unit=False):
        '''
        Naive Bayes place decoder fitting use precise spike timing to compute the representation 
        (Rather than using binned spike count vector in t_window)
        Therefore the X and y is None for the consistency of the decoder API
        '''
        if remove_first_unit: # remove the first neuron (the one classified as noise)
            self.pc.spk_time_dict = {i: self.pc.spk_time_dict[i+1] for i in range(len(self.pc.spk_time_dict.keys())-1)}
        self.pc.get_fields(self.pc.spk_time_dict, self.train_time[0], self.train_time[1], v_cutoff=self.v_cutoff, rank=False)
        # self.pc.get_fields()
        self.fields = self.pc.fields
        self.spatial_bin_size, self.spatial_origin = self.pc.bin_size, self.pc.maze_original

        # for real-time decoding on incoming bin from BMI   
        self.poisson_matrix = self.t_window*self.fields.sum(axis=0)
        self.log_fr = np.log(self.fields) # make sure Fr[Fr==0] = 1e-12
        self.neuron_idx = np.arange(self.fields.shape[0])

        if self._disable_neuron_idx is not None:
            self.neuron_idx = np.array([_ for _ in range(self.fields.shape[0]) if _ not in self._disable_neuron_idx])

    @property
    def mean_firing_rate(self):
        return np.mean(self.train_X[:, self.neuron_idx])

    def predict(self, X, firing_rate_modulation=True, two_steps=False):
        '''
        # TODO: #2 Add two_steps decoding method to cope with erratic jumps 
        zhang et al., 1998 (https://journals.physiology.org/doi/full/10.1152/jn.1998.79.2.1017)
        '''
        X_arr = X.copy()

        if len(X_arr.shape) == 1:
            X_arr = X_arr.reshape(1,-1)

        if self._disable_neuron_idx is not None:
            firing_bins = X_arr[:, self.neuron_idx]
            place_fields = self.fields[self.neuron_idx]
        else:
            firing_bins = X_arr
            place_fields = self.fields

        if firing_rate_modulation:
            self.post_2d = bayesian_decoding(place_fields, firing_bins, 
                                            t_window=self.t_window, 
                                            mean_firing_rate=self.mean_firing_rate)
        else:
            self.post_2d = bayesian_decoding(place_fields, firing_bins, t_window=self.t_window)
        binned_pos = argmax_2d_tensor(self.post_2d)
        y = binned_pos*self.spatial_bin_size + self.spatial_origin
        return y

    def predict_rt(self, X, two_steps=False, gamma=0.06, mean_firing_rate=None):
        # Ponential update (performance improved when using with 3 seconds moving average window)
        if X.ndim == 1:
            X = X.reshape(1,-1)
        elif X.ndim>1 and X.shape[0]>1:
            X = np.sum(X, axis=0)  # X is (B_bins, N_neurons) spike count matrix, we need to sum up B bins to decode the full window

        X = X.ravel()

        if self._disable_neuron_idx is not None:
            firing_bins = X[self.neuron_idx]
            place_fields = self.fields[self.neuron_idx]
        else:
            firing_bins = X
            place_fields = self.fields
        
        if mean_firing_rate is not None:
            firing_rate_ratio = firing_bins.mean()/mean_firing_rate # eq.44 (Zhang et al., 1998) change speed to firing rate
        else:
            firing_rate_ratio = 1

        suv_weighted_log_fr = licomb_Matrix(firing_bins, np.log(place_fields))

        if self.last_max_bin is not None and two_steps == True:
            self.two_step_constraint_field = gaussian_inhibition_field(coord=self.last_max_bin, 
                                                              size_x=suv_weighted_log_fr.shape[0], 
                                                              size_y=suv_weighted_log_fr.shape[1])
        else:
            self.two_step_constraint_field = np.zeros_like(suv_weighted_log_fr)

        ## unormalized log likelihood, check eq.36,41,43,47 (Zhang et al., 1998): 
        ## firing_rate_ratio is the modulation factor of the firing rate (m(t) in e.q. 47)
        self.log_likelihood = suv_weighted_log_fr - \
                              firing_rate_ratio*self.t_window*place_fields.sum(axis=0) + \
                              gamma * self.two_step_constraint_field / (firing_rate_ratio**2)
    
        self.rt_post_2d = np.exp(self.log_likelihood)
        self.rt_post_2d /= self.rt_post_2d.sum()
        self.rt_pred_binned_pos = argmax_2d_tensor(self.rt_post_2d)
        self.last_max_bin = self.rt_pred_binned_pos
        y = self.rt_pred_binned_pos*self.spatial_bin_size + self.spatial_origin
        return y, self.rt_post_2d


class Olayer(nn.Module):
    def __init__(self, hidden_dim=[128,64]):
        super(Olayer, self).__init__()
        self.fc1l = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc1r = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc1m = nn.Linear(hidden_dim[0], hidden_dim[1])
        
    def forward(self, xp, freq=torch.pi):
        xl = self.fc1l(xp)      # hidden_dim[0] -> hidden_dim[1]
        xr = self.fc1r(xp)
#         xm = self.fc1m(xp)
        xg = torch.cos(freq*xl) + torch.cos(freq*xr) # *torch.cos(freq*xm)
        return xg

class SineDec(nn.Module):

    def __init__(self, input_dim, hidden_dim=[128,64], output_dim=2, bn=False, LSTM=True):
        super(SineDec, self).__init__()
        self.LSTM = LSTM
        self.bn = bn
        self.encoder = nn.Linear(input_dim, hidden_dim[0])
        self.ln1 = nn.LayerNorm(hidden_dim[0])
        self.bn1 = nn.BatchNorm1d(hidden_dim[0])
        # self.bn1 = nn.InstanceNorm1d(hidden_dim[0])
        self.fc1  = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc1xm = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc2g = nn.Linear(hidden_dim[1], output_dim, bias=True)
        self.fc2p = nn.Linear(hidden_dim[1], output_dim, bias=True)
        self.fc2  = nn.Linear(hidden_dim[1], output_dim, bias=True)
        # self.olayer1 = Olayer(hidden_dim)
        # self.olayer2 = Olayer(hidden_dim)
        # self.olayer3 = Olayer(hidden_dim)
        # self.olayer4 = Olayer(hidden_dim)
        # self.olayer5 = Olayer(hidden_dim)
        # self.olayer6 = Olayer(hidden_dim)
        # self.olayer7 = Olayer(hidden_dim)
        # self.olayer8 = Olayer(hidden_dim)
        self.olayer = nn.ModuleList([Olayer(hidden_dim) for i in range(11)])
        self.dropout = nn.Dropout(0.1)
        # alternative (not used yet)
        self.lstm = nn.LSTM(hidden_dim[0], hidden_dim[0], batch_first=True)   #input: (batch, seq, feature)
        self.ln_lstm = nn.LayerNorm(hidden_dim[0])
        # speed related
        self.fcx2v  = nn.Linear(hidden_dim[1], output_dim, bias=True)
        self.v2x_lstm = nn.LSTM(output_dim, hidden_dim[1], bias=True)
        self.ln2 = nn.LayerNorm(hidden_dim[1])
        self.fcv2x = nn.Linear(output_dim, hidden_dim[1], bias=True)
        
    def forward(self, X):
        x = self.encoder(X)  # input_dim -> hidden_dim[0]
        # x = self.ln1(x)
        if self.bn:
            x = self.bn1(x)      # 

        x = self.dropout(x)

        # x = self.dropout(x)
#         x = torch.sin(x)
        # x = self.fc1(x)    # to both place and speed prediction 
        x = F.softmax(F.relu(torch.sin(x) + torch.sin(self.fc1(x)))) * x
        # x = F.softmax(F.relu(self.fc1(x))) * x


        if self.LSTM:
            x, h = self.lstm(x.view(len(x), 1, -1))
            x = self.ln_lstm(x)
            x.squeeze_()

        xg = self.olayer[0](x) + x
        xv = self.olayer[1](x) + x
        
        xv = self.olayer[2](xv) + xv
#         xv = F.relu(xv)
#         xv = torch.sin(xv)/(xv+1e-15)
        xv = self.dropout(xv)
        v = self.fcx2v(xv)
#         v2x, _ = self.v2x_lstm(v.view(len(v), 1, -1))
#         v2x = self.ln2(v2x)
#         v2x.squeeze_()
#         xg = self.dropout(v2x) # heavy dropout for grid emergence
#         xg = v2x
        xg = self.olayer[3](xg) + xg
        xg = self.olayer[4](xg) + xg
        xg = self.olayer[5](xg) + xg
        xg = self.olayer[6](xg) + xg
        # xg = self.olayer[7](xg) + xg
        # xg = self.olayer[8](xg) + xg
        # xg = self.olayer[9](xg) + xg
        # xg = self.olayer[10](xg) + xg
        # xg = self.dropout(xg)
        # prediction
        y = self.fc2p(xg)      # hidden_dim[1] -> 2

#         xg = self.olayer7(xg)
#         xg = self.olayer8(xg)

#         xp = self.fc2p(xp)
        return x, xg, y, v


class DeepOSC(Decoder):
    """
    DeepOSC Decoder for position prediction (input X, output y) 
    where X is the spike bin matrix (T_step, N_neurons*B_bins) # ! this is different from naive bayes decoder 
    where y is the 2D position (x,y)

    Examples:
    -------------------------------------------------------------
    from spiketag.analysis import DeepOSC, smooth

    dec = DeepOSC(t_window=250e-3, t_step=50e-3)
    dec.connect_to(pc)

    dec.partition(training_range=[0.0, .7], valid_range=[0.5, 0.6], testing_range=[0.6, 1.0], 
                  low_speed_cutoff={'training': True, 'testing': True})
    (train_X, train_y), (valid_X, valid_y), (test_X, test_y) = dec.get_data(minimum_spikes=0)
    dec.fit(train_X, train_y)

    predicted_y = dec.predict(test_X)
    dec_pos  = smooth(predicted_y, 60)
    real_pos = test_y
    
    score = dec.r2_score(real_pos, dec_pos)

    # optional (cost time):
    # dec.plot_decoding_err(real_pos, dec_pos);

    To get scv matrix to hack (check the size):
    -------------------------------------------------------------
    # ! 1. data
    pc.output_variables = ['scv', 'pos']
    N = int(len(pc)*0.5)
    X,y = pc[:N]
    X_test, y_test = pc[N:]
    # ! 2. training
    dec.train(X,y) 
    # ! 3. predict   
    _y = dec.predict(X_test)
    y_decoded  = smooth(_y, 60)

    # ! 4. test
    score = dec.r2_score(y_test, y_decoded)

    Test real-time prediction:
    -------------------------------------------------------------
    _y = dec.predict_rt()
    """

    # TODO 
    pass

    def __init__(self, input_dim, hidden_dim=[128, 128], output_dim=2, bn=False, LSTM=False, t_window=3, t_step=0.1):
        super(DeepOSC, self).__init__(t_window, t_step)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = SineDec(input_dim=self.input_dim, 
                             hidden_dim=self.hidden_dim, 
                             output_dim=self.output_dim, 
                             bn=bn,
                             LSTM=LSTM)
        self.test_r2 = []
        self.losses = []

    def unroll(self, scv, n):
        '''
        unroll scv such that it has (T_step, B_bins, N_neurons) structure
        '''
        pass

    def fit(self, X, y, X_test, y_test, max_epoch=5000, smooth_factor=60, max_noise=1, 
                                        early_stop_r2=0.82, lr=3e-4, weight_decay=0.01, cuda=True):
        '''
        training the deep neural network, using GPU if `cuda` == True
        '''
        # optmizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        
        if type(X) == np.ndarray:
            X = torch.from_numpy(X).float()
        if type(y) == np.ndarray:
            y = torch.from_numpy(y).float()
        if type(X_test) == np.ndarray:
            X_test = torch.from_numpy(X_test).float()
        if type(y_test) == np.ndarray:
            y_test = torch.from_numpy(y_test).float()

        if cuda:
            X = X.cuda()
            y = y.cuda()
            X_test = X_test.cuda()
            y_test = y_test.cuda()
            self.model.cuda();

        for epoch in range(max_epoch):
            self.model.train()
            self.optimizer.zero_grad()
            gain = 1 + 0.1*torch.randn(1, device='cuda')
            std = 0.05 + np.abs(np.sin(epoch/500*3.14)) * max_noise
            # gausian noise with sqrt spike count
        #     noise_X = spike_noise_gaussian(X, noise_level=0,
        #                                   mean=0, std=std, gain=gain,
        #                                   IID=True, cuda=True)
            # bernoulli noise with spike count
            noise_X = spike_noise_bernoulli(
                X, noise_level=std, p=0.5, gain=gain, cuda=True, IID=True)
            h, grid, _y, _v = self.model(noise_X)
            now_location = (y + 0.5*torch.rand_like(y, device='cuda')) 
            loss = F.mse_loss(now_location, _y) 
        #         loss = F.mse_loss(y[:-1] + _v[:-1], y[1:]) * 5
        #     loss += F.mse_loss(y[:-1] + _v[:-1], y[1:])
        #     loss += F.mse_loss(_v, v)
        #     loss_g = F.mse_loss(now_location, h)
            norm_val = torch.norm(grid, p=1, dim=1).sum() * 1e-6 + \
                       torch.norm(grid, p=1, dim=0).sum() * 1e-6
            norm_h = torch.norm(h, p=1) * 1e-6
            loss = loss + norm_val  # norm_h # + norm_val # + loss_g # + norm_val # + norm_h
        #     grid = nn.Dropout(0.6)(grid)
        #     grid_pos = self.model.fc2(grid)
        #     grid_pos_loss = F.mse_loss(grid_pos, now_location)
        #     loss += grid_pos_loss

            loss.backward()
            self.optimizer.step()
        #     lrtim.step()
            if epoch > 1000:
                self.set_learning_rate(lr/4)
            if epoch > 2000:
                self.set_learning_rate(lr/8)

            if epoch % 10 == 0:
                with torch.inference_mode():
                    _yo = self.predict(X_test)
                dec_y = smooth(_yo, smooth_factor)
                r2_p = r2_score(y_test.cpu().numpy(), dec_y)
                self.test_r2.append(r2_p)
                self.losses.append(loss.item())
                print(f'[{epoch}] loss: {loss.item():.3f}, test r2: {r2_p:.3f}, std:{std:.3f}, norm_h:{norm_h:.3f}, norm_output:{norm_val:.3f}', end='\r')

    def set_learning_rate(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def plot_loss(self):
        fig, ax1 = plt.subplots(figsize=(8,5))
        color = 'C0'
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('test R2', color=color)
        ax1.plot(np.arange(len(self.test_r2))*10, self.test_r2, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'C1'
        ax2.set_ylabel('training losses', color=color)  # we already handled the x-label with ax1
        ax2.plot(np.arange(len(self.losses))*10, self.losses, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        return fig

    def predict(self, X, cuda=True, mode='eval', bn_momentum=0.1):
        if type(X) == np.ndarray:
            X = torch.from_numpy(X).float()
        if X.ndim == 1:
            X = X.view(1,-1)
        if cuda:
            X = X.cuda()
        if mode=='eval':
            self.model.eval()
        elif mode=='train':
            self.model.train()
        self.model.bn1.momentum = bn_momentum

        with torch.inference_mode():
            _, _, _yo, _vo = self.model(X)
            _yo = _yo.cpu().detach().numpy()
            # _vo = _vo.cpu().detach().numpy()
        return np.nan_to_num(_yo)

    def predict_rt(self, X, cuda=True, mode='train', bn_momentum=0.1):
        y = self.predict(X, cuda, mode, bn_momentum)
        self.rt_post_2d = self.pc.real_pos_2_soft_pos(y)
        return y, self.rt_post_2d
