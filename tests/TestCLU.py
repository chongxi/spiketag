import sys
sys.path.append('.')
import unittest
import numpy as np
from spiketag.base import CLU
import numpy as np

class TestCLU(unittest.TestCase):
    
    def setUp(self):
        membership = np.array([0,1,2,
                        2,1,0,
                        0,2,1,
                        1,2,0])
        self.clu = CLU(membership)

    '''
       Test Cases
    '''
    def test_index(self):
        expected = {0:[0,5,6,11],
                      1:[1,4,8,9],
                      2:[2,3,7,10]}
        self.assertDictEqual(self._array2list(self.clu.index),expected)

    def test_global2local(self):
        globals = np.array([1,5,6,9])
        expected = {0:[1,2],1:[0,3]}
        result = self.clu.global2local(globals)
        self.assertDictEqual(self._array2list(result),expected)


    def test_local2global(self):
        locals = {0:[1,2],1:[0,3]}
        expected = [1,5,6,9]
        result = self.clu.local2global(locals)
        self.assertListEqual(list(result),expected)

    def test_move_multi2one_exclude_same_clu(self):
        '''
            clu.index = {0:[0,5,6,11],1:[1,4,8,9],2:[2,3,7,10]}
            move {0:[0,2],1:[1,3]} to 2
        '''
        clus_from = {0:np.array([0,2]),1:np.array([1,3])}
        clu_to = 2
        expected_result = [0,3,4,6]
        expected_index = {0:[5,11],1:[1,8],2:[0,2,3,4,6,7,9,10]}
        result = self.clu.move(clus_from,clu_to)
        self.assertListEqual(list(result),expected_result)
        self.assertDictEqual(self._array2list(self.clu.index),expected_index)
        
        self.setUp()

    def test_move_multi2one_include_same_clu(self):
        '''
            clu.index = {0:[0,5,6,11],1:[1,4,8,9],2:[2,3,7,10]}
            move {0:[0,2],1:[1,3],2:[0,1]} to 2
        '''
        clus_from = {0:np.array([0,2]),1:np.array([1,3]),2:np.array([0,1])}
        clu_to = 2
        expected_result = [0,1,2,3,4,6]
        expected_index = {0:[5,11],1:[1,8],2:[0,2,3,4,6,7,9,10]}
        result = self.clu.move(clus_from,clu_to)
        self.assertListEqual(list(result),expected_result)
        self.assertDictEqual(self._array2list(self.clu.index),expected_index)
        
        self.setUp()

    def test_move_one2one_not_same_clu(self):
        '''
            clu.index = {0:[0,5,6,11],1:[1,4,8,9],2:[2,3,7,10]}
            move {0:[0,2]} to 2
        '''
        clus_from = {0:np.array([0,2])}
        clu_to = 2
        expected_result = [0,3]
        expected_index = {0:[5,11],1:[1,4,8,9],2:[0,2,3,6,7,10]}
        result = self.clu.move(clus_from,clu_to)
        self.assertListEqual(list(result),expected_result)
        self.assertDictEqual(self._array2list(self.clu.index),expected_index)
        
        self.setUp()

    def test_move_one2one_is_same_clu(self):
        '''
            clu.index = {0:[0,5,6,11],1:[1,4,8,9],2:[2,3,7,10]}
            move {0:[0,2]} to 0
        '''
        clus_from = {0:np.array([0,2])}
        clu_to = 0
        expected_result = [0,2]
        expected_index = {0:[0,5,6,11],1:[1,4,8,9],2:[2,3,7,10]}
        result = self.clu.move(clus_from,clu_to)
        self.assertListEqual(list(result),expected_result)
        self.assertDictEqual(self._array2list(self.clu.index),expected_index)
        
        self.setUp()

      
    '''
        Private methond
    '''
    def _array2list(self,dict):
        '''
            because assert method can not assert a dict containing numpy array, so need a method converting list from numpy array
        '''
        for k,v in dict.iteritems():
            dict[k] = list(v)
        return dict


if __name__ == "__main__":
    unittest.main()

