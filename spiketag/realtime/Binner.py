from ..utils.utils import EventEmitter, Timer
import numpy as np


class Binner(EventEmitter):
    """
    The binner takes real-time spike input from the BMI and output when B time bins are ready for the decoder
    To decode the output in a given time bin, we used the spike count of N neurons in B time bins output from the binner

    binner = Binner(bin_size=33.33, n_id=N, n_bin=B)    # binner initialization (space and time)
    binner.input(bmi_output, type='individual_spike')   # internal state update triggered

    Parameters:

    N N neurons give rise to N spike count in each bin
    B B bins
    bin_size The time span (ms) to compute the spike count of each bin
    Internal States:

    count_vec (+1 on with the input spike_id)
    nbins (+1 when the input timestamps goes to the next bin, its number is the current bin)
    output (emitted variable to the decoder, N neuron's spike count in previous B bins)

    https://github.com/chongxi/spiketag/issues/47
    """
    def __init__(self, bin_size, n_id, n_bin, sampling_rate=25000):
        super(Binner, self).__init__()
        self.bin_size = bin_size
        self.N = n_id
        self.B = n_bin
        self.count_vec = self.new_empty_bin
        self.nbins = 1 # self.nbins-1 is the index of the last bin
        self.fs = sampling_rate
        self.dt = 1/self.fs*1e3   # each frame is 0.04ms, which is the resolution of timestamp

    def input(self, bmi_output, type='individual_spike'):
        '''
        each bmi_output is a spike with its timestamp and spike id
        each time a bmi_output arrive, this function is triggered
        when nbins grows, the binner emits the `decode` event with its `_output`
        '''
        current_bin = (bmi_output.timestamp*self.dt)//(self.bin_size) # devicded by [bin_size] 
        if current_bin == self.nbins-1:   # state integrate
            self.count_vec[bmi_output.spk_id, self.nbins-1] += 1
        elif current_bin > self.nbins-1:   # first condition for the output to decoder
            self.nbins += 1
            self.count_vec = np.hstack((self.count_vec, self.new_empty_bin))
            self.count_vec[bmi_output.spk_id, self.nbins-1] += 1
            # second condition for the output to decoder
            if self.count_vec.shape[1]>self.B:
                self.emit('decode', X=self.output)

    @property
    def output(self):
        # first row is the noise
        # last column is the just added, output the last three before the last column
        self._output = self.count_vec[1:, -self.B-1:-1] 
        return self._output

    @property
    def new_empty_bin(self):
        return np.zeros((self.N,1))