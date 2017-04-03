import sys
sys.path.append('../../../')
import unittest
import numpy as np
from spiketag.base import ProbeFactory

class TestProbe(unittest.TestCase):

    def test_linear_probe(self):
        
        linear_probe = ProbeFactory.genLinearProbe(25e3, 32)
    
        expected_type = 'linear'
        expected_n_group = 32
        expected_n_ch = 32
        expected_len_group = 3

        self.assertEqual(expected_type, linear_probe.type)
        self.assertEqual(expected_n_group, linear_probe.n_group)
        self.assertEqual(expected_n_ch, linear_probe.n_ch)
        self.assertEqual(expected_len_group, linear_probe.len_group)

        expected_near_ch_1 = [4, 5, 6]
        near_ch_1 = linear_probe.get_group_ch(5)
        self.assertListEqual(list(near_ch_1), expected_near_ch_1)

        expected_near_ch_2 = [-1, 0, 1]
        near_ch_2 = linear_probe.get_group_ch(0)
        self.assertListEqual(list(near_ch_2), expected_near_ch_2)

        expected_near_ch_3 = [30, 31, -1]
        near_ch_3 = linear_probe.get_group_ch(31)
        self.assertListEqual(list(near_ch_3), expected_near_ch_3)
   
    def test_tetrode_probe(self):
        
        tetrode_probe = ProbeFactory.genTetrodeProbe(25e3, 100)
    
        expected_type = 'tetrode'
        expected_n_group = 25
        expected_n_ch = 100
        expected_len_group = 4

        self.assertEqual(expected_type, tetrode_probe.type)
        self.assertEqual(expected_n_group, tetrode_probe.n_group)
        self.assertEqual(expected_n_ch, tetrode_probe.n_ch)
        self.assertEqual(expected_len_group, tetrode_probe._len_group)

        expected_near_ch_1 = [0, 1, 2, 3]
        near_ch_1 = tetrode_probe.get_group_ch(0)
        self.assertListEqual(list(near_ch_1), expected_near_ch_1)
        near_ch_1 = tetrode_probe.get_group_ch(3)
        self.assertListEqual(list(near_ch_1), expected_near_ch_1)

        expected_near_ch_2 = [4, 5, 6, 7]
        near_ch_2 = tetrode_probe.get_group_ch(4)
        self.assertListEqual(list(near_ch_2), expected_near_ch_2)

        expected_near_ch_3 = [96, 97, 98, 99]
        near_ch_3 = tetrode_probe.get_group_ch(96)
        self.assertListEqual(list(near_ch_3), expected_near_ch_3)
        near_ch_3 = tetrode_probe.get_group_ch(99)
        self.assertListEqual(list(near_ch_3), expected_near_ch_3)

