from .core import bayesian_decoding, argmax_2d_tensor, smooth
import numpy as np
from sklearn.metrics import r2_score



class Decoder(object):
    """Base class for the decoders for place prediction"""
    def __init__(self, t_window, t_step=None):
        '''
        t_window is the bin_size
        t_step   is the step_size (if None then use pc.ts as natrual sliding window)
        https://github.com/chongxi/spiketag/issues/47 
        
        For Non-RNN decoder, large bin size in a single bin are required
        For RNN decoder,   small bin size but multiple bins are required

        During certain neural state, such as MUA burst (ripple), a small step size is required 
        (e.g. t_window:20ms, t_step:5ms is used by Pfeiffer and Foster 2013 for trajectory events) 
        '''
        self.t_window = t_window
        self.t_step   = t_step

    def connect_to(self, pc):
        '''
        This decoder is specialized for position decoding
        Connect to a place-cells object that contains behavior, neural data and co-analysis
        '''
        self.pc = pc
        if self.t_step is not None:
            self.pc(t_step=self.t_step)

    def resample(self, t_window, t_step):
        self.t_window = t_window
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

    def partition(self, training_range=[0.0, 0.5], valid_range=[0.5, 0.6], testing_range=[0.6, 1.0]):
        
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

    def evaluate(self, y_predict, y_true, multioutput=True):
        if multioutput is True:
            score = r2_score(y_true, y_predict, multioutput='raw_values')
        else:
            score = r2_score(y_true, y_predict)
        return score




class NaiveBayes(Decoder):
    """NaiveBayes Decoder for place prediction
    >>> nbdec = NaiveBayes(pc)
    >>> nbdec(t_window=delta_t, t_step=np.diff(pc.ts).mean())
    >>> nbdec.partition(training_range=[0.0, .5], valid_range=[0.5, 0.6], testing_range=[0.6, 1.0])
    >>> (train_X, train_y), (valid_X, valid_y), (test_X, test_y) = nbdec.get_partitioned_data()
    >>> nbdec.fit()
    >>> predicted_y = nbdec.predict(test_X[:, pc.v_smoothed>25])
    """
    def __init__(self, t_window, t_step=None):
        super(NaiveBayes, self).__init__(t_window, t_step)


    def get_partitioned_data(self, low_speed_cutoff={'training': True, 'testing': False}, v_cutoff=5):
        '''
        The data strucutre is different for RNN and non-RNN decoder
        Therefore each decoder subclass has its own get_partitioned_data method
        In low_speed periods, data should be removed from train and valid:
        '''
        assert(pc.ts.shape[0] == pc.pos.shape[0])
        X = self.pc.get_scv(self.t_window) # t_step is None unless specified
        y = self.pc.pos

        if low_speed_cutoff['training'] is True:
            train_X = X[:, np.where(self.pc.v_smoothed[self.train_idx]>v_cutoff)[0]]
            train_y = y[np.where(self.pc.v_smoothed[self.train_idx]>v_cutoff)[0]]
            valid_X = X[:, np.where(self.pc.v_smoothed[self.valid_idx]>v_cutoff)[0]]
            valid_y = y[np.where(self.pc.v_smoothed[self.valid_idx]>v_cutoff)[0]]
        else:
            train_X, train_y = X[:, self.train_idx], y[self.train_idx]
            valid_X, valid_y = X[:, self.valid_idx], y[self.valid_idx]

        if low_speed_cutoff['testing'] is True:
            test_X = X[:, np.where(self.pc.v_smoothed[self.test_idx]>v_cutoff)[0]]
            test_y = y[np.where(self.pc.v_smoothed[self.test_idx]>v_cutoff)[0]]
        else:
            test_X, test_y = X[:, self.test_idx], y[self.test_idx]
        return (train_X, train_y), (valid_X, valid_y), (test_X, test_y) 

        
    def fit(self, X=None, y=None):
        '''
        Naive Bayes place decoder fitting use precise spike timing to compute the representation 
        (Rather than using binned spike count vector in t_window)
        Therefore the X and y is None for the consistency of the decoder API
        '''
        self.pc.get_fields(self.pc.spk_time_dict, self.train_time[0], self.train_time[1], rank=False)
        self.fr = self.pc.fields_matrix


    def predict(self, X):
        post_2d = bayesian_decoding(self.fr, X, delta_t=self.t_window)
        binned_pos = argmax_2d_tensor(post_2d)
        y = smooth(self.pc.binned_pos_2_real_pos(binned_pos), 5)
        return y
