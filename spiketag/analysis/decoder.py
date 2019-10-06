from .core import bayesian_decoding, argmax_2d_tensor
import numpy as np


class Decoder(object):
    """Base class for the decoders for place prediction"""
    def __init__(self, pc):
        self.pc = pc

    def __call__(self, t_window, t_step=None):
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
        
        # self.train_set=np.arange(int(training_range[0]*num_examples)+bins_before,int(training_range[1]*num_examples)-bins_after)
        # self.valid_set=np.arange(int(valid_range[0]*num_examples)+bins_before,int(valid_range[1]*num_examples)-bins_after)
        # self.test_set=np.arange(int(testing_range[0]*num_examples)+bins_before,int(testing_range[1]*num_examples)-bins_after)

    def get_scv(self):
        self.scv, self.ts, self.pos = self.pc.get_scv(t_window=self.t_window, t_step=self.t_step)


class NaiveBayes(Decoder):
    """NaiveBayes Decoder for place prediction
    >>> nbdec = NaiveBayes(pc)
    >>> nbdec(t_window=delta_t, t_step=np.diff(pc.ts).mean())
    >>> nbdec.partition(training_range=[0.0, .5], valid_range=[0.5, 0.6], testing_range=[0.6, 1.0])
    >>> (train_X, train_y), (valid_X, valid_y), (test_X, test_y) = nbdec.get_partitioned_data()
    >>> nbdec.fit()
    >>> predicted_y = nbdec.predict(test_X[:, pc.v_smoothed>25])
    """
    def __init__(self, pc):
        super(NaiveBayes, self).__init__(pc)


    def get_partitioned_data(self):
        '''
        The data strucutre is different for RNN and non-RNN decoder
        Therefore each decoder subclass has its own get_partitioned_data method
        '''
        X = self.pc.get_scv(self.t_window, self.t_step) # t_step is None unless specified
        y = self.pc.binned_pos
        train_X, train_y = X[:, self.train_idx], y[self.train_idx]
        valid_X, valid_y = X[:, self.valid_idx], y[self.valid_idx]
        test_X, test_y   = X[:, self.test_idx],  y[self.test_idx]
        return (train_X, train_y), (valid_X, valid_y), (test_X, test_y) 

        
    def fit(self, X=None):
        '''
        Naive Bayes place decoder fitting use precise spike timing to compute the representation 
        (Rather than using binned spike count vector in t_window)
        '''
        self.pc.get_fields(self.pc.spk_time_dict, self.train_time[0], self.train_time[1], rank=False)
        self.fr = self.pc.fields_matrix


    def predict(self, X):
        post_2d = bayesian_decoding(self.fr, X, delta_t=self.t_window)
        y = argmax_2d_tensor(post_2d)
        return y
