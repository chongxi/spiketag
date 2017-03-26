import sys
sys.path.append('../../../')
import unittest
import numpy as np
from spiketag.core import correlate

class TestCorrelate(unittest.TestCase):

    def setUp(self):
        self.spike_time = np.array([1,3,5,10,11])
        self.membership = np.array([1,0,1,0,1])
        self.ids = np.array([0,1])

    '''
        TestCase
    '''
    def test_correlate1(self):
        window_size = 2
        bin_size = 1
        fs = 1e3
       
        expected_ccgs = np.array(([[0,0],[0,1]],[[1,0],[0,0]]))
        real_ccgs = correlate(self.spike_time, self.membership, self.ids,  window_bins = window_size, bin_size = bin_size, fs = fs)
        self.assertTrue(np.array_equal(expected_ccgs,real_ccgs)) 


    def test_correlate2(self):
        window_size = 2 
        bin_size = 2
        fs = 1e3

        expected_ccgs = np.array(([[0,0],[1,2]],[[2,1],[0,0]]))
        real_ccgs = correlate(self.spike_time, self.membership, self.ids,  window_bins = window_size, bin_size = bin_size, fs = fs)
        self.assertTrue(np.array_equal(expected_ccgs,real_ccgs)) 



