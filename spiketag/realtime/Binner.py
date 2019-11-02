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

    count_vec: shape = (B,N) (entry +1 on with the input spike_id)
    nbins (+1 when the input timestamps goes to the next bin, its number is the current bin)
    output (emitted variable to the decoder, N neuron's spike count in previous B bins)

    https://github.com/chongxi/spiketag/issues/47
    """
    def __init__(self, bin_size, n_id, n_bin, sampling_rate=25000):
        super(Binner, self).__init__()
        self.bin_size = bin_size
        self.N = n_id
        self.B = n_bin
        self.count_vec = np.zeros((self.B, self.N))
        self.nbins = 1 # self.nbins-1 is the index of the last bin
        self.fs = sampling_rate
        self.dt = 1/self.fs   # each frame is 1/25000:40us, which is the resolution of timestamp
        self.last_bin = 0

    def input(self, bmi_output, type='individual_spike'):
        '''
        each bmi_output is a spike with its timestamp and spike id
        each time a bmi_output arrive, this function is triggered
        when nbins grows, the binner emits the `decode` event with its `_output`
        '''
        self.current_time = bmi_output.timestamp*self.dt
        self.current_bin = int(self.current_time//self.bin_size) # devided by [bin_size], current_bin is abosolute bin

        if self.current_bin < self.B:                                                 # within B, no new bin
            self.count_vec[self.current_bin, bmi_output.spk_id] += 1                  # update according to current_bin
        elif self.current_bin >= self.B and self.current_bin==self.last_bin:          # current_bin 
            self.count_vec[-1, bmi_output.spk_id] += 1
        elif self.current_bin >= self.B and self.current_bin>self.last_bin:           # key: current_bin>last_bin means a input to decoder is completed 
            self.emit('decode', X=self.output)                                        # output count_vec for decoding
            self.count_vec = np.vstack((self.count_vec[1:], np.zeros((1, self.N))))   # roll and append new bin (last row)
            self.count_vec[-1, bmi_output.spk_id] += 1                                # update the newly appended bin (last row)

        # print(self.count_vec.shape, self.current_bin, self.last_bin)
        self.last_bin = self.current_bin

    @property
    def output(self):
        # first column (unit) is the noise
        self._output = self.count_vec[:, 1:] 
        return self._output
