import numpy as np
import json
from collections import OrderedDict
from spiketag.view import probe_view
import pandas as pd
from ..utils import EventEmitter


#------------------------------
#        Channel Group
#------------------------------
class ch_group(object):
    """atomic class for probe
       contains channel numbers(_channels) and positions(_pos)

       tetrode = ch_group([0,5,9,12])
       tetrode.pos = np.array([[0,0], [0,10], [10,0],[10,10]])
       tetrode.rank()

    """
    def __init__(self, channels):
        self.channels = np.array(channels)
        self.pos = np.zeros((self.channels.shape[0], 2)) 
        self.__shank__ = 0
        self.__group__ = 0

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, _channels):
        self._channels = np.array(_channels)

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, _pos):
        self._pos = np.array(_pos)    

    def rank(self):
        idx = np.argsort(self._channels)
        self.channels = self.channels[idx]
        self.pos = self.pos[idx]

    def __getitem__(self, i):
        return zip(self._channels, self._pos)[i]

    def __setitem__(self, idx, args):
        ch = args[0]
        pos = args[1]
        self._channels[idx] = ch
        self._pos[idx] = pos

    def __repr__(self):
        return str(zip(self._channels, self._pos))



#------------------------------
#        shank
#------------------------------
class shank(object):
    def __init__(self, shank_id=0):
        self.__shank__ = shank_id
        self.ch_group = {}
        self.mapping = OrderedDict()
        self.m = []  # middle side
        self.l = []  # left side
        self.r = []  # right side
        self.xl = 0   # shank left side x position (tip)
        self.yl = 0   # shank left side y position (tip)
        self.xr = 0   # shank right side x position
        self.yr = 0   # shank right side y position

    def __getitem__(self, group_id):
        return self.ch_group[group_id]

    def __setitem__(self, group_id, ch_grp):
        self.ch_group[group_id] = ch_grp

    def __repr__(self):
        return 'shank_{}: (left:{}, middle:{}, right:{})'.format(str(self.__shank__), str(self.l), str(self.m), str(self.r))




#------------------------------
#        probe
#------------------------------


class BaseProbe(EventEmitter):
    ''' the base class of probe
    '''
    def __init__(self):
        super(BaseProbe, self).__init__()
        self.shanks = {}
        self._grp_dict = {}
        self.mapping = OrderedDict()
        self.ch2g = {}
        self._bad_chs = np.array([], dtype=int)
        self.fpga_connected = False
        self.reorder_by_chip = False

    @property
    def fs(self):
        return self._fs

    @fs.setter
    def fs(self, _fs):
        self._fs = _fs
    
    @property
    def n_ch(self):
        return self._n_ch

    @n_ch.setter
    def n_ch(self, _n_ch):
        self._n_ch = _n_ch
        mask_chs = np.setdiff1d(np.arange(_n_ch), self.chs)
        mask_chs = np.union1d(mask_chs, self.bad_chs)
        self._mask_chs = mask_chs

    @property
    def mask_chs(self):
        return self._mask_chs

    @property
    def n_group(self):
        '''
          number of groups, group can be treated as basic unit to statistic data
        '''
        self._n_group = len(self.grp_dict.keys())
        return self._n_group
    
    @property
    def groups(self):
        '''
            return all group numbers
        '''
        for i in range(self._n_group):
            yield i

    @property
    def group_len(self):
        '''
            number of channles in a group
        '''
        return self._group_len


    @property
    def grp_dict(self):
        return self._grp_dict

    @grp_dict.setter
    def grp_dict(self, grp_dict_in):
        self._grp_dict = grp_dict_in
        for g, chs in self._grp_dict.items():
            chs.sort()
            self._update_chs2group(chs, g)

    @property
    def chs(self):
        return np.hstack(self.grp_dict.values())

    @property
    def bad_chs(self):
        return self._bad_chs

    @bad_chs.setter
    def bad_chs(self, v):
        self._bad_chs = v.astype(int)

    def _update_chs2group(self, chs, g):
       for ch in chs:
            self.ch2g[ch] = g

    def __getitem__(self, group):
        assert group >= 0 and group < self._n_group, 'invalid group value'
        return self.grp_dict[group]

    def __setitem__(self, group, chs):
        assert group >= 0 and group < self._n_group, 'invalid group value'
        # assert len(chs) == 4, 'invalid amount of chs in group (so far only support 4)'
        assert isinstance(chs, np.ndarray), 'chs should numpy array, make sure numba works'
        self.grp_dict[group] = np.sort(chs)
        self._update_chs2group(chs, group)  

    def ch_hash(self, ch):
        if ch in self.chs:
            return self.grp_dict[self.ch2g[ch]]
        elif ch in self.mask_chs:
            mask_grp = self.mask_chs.reshape(-1, self.group_len)
            return mask_grp[np.where(mask_grp == ch)[0]][0]
        else:
            print('ch not in range, check prb.chs and prb.mask_chs')

    def __str__(self):
        return '\n'.join(['{}:{}'.format(key, val) for key, val in self.grp_dict.items()]) 

    __repr__ = __str__



class probe(BaseProbe):
    '''
        shk = shank(shank_id=0)
        shk[1] = ch_group([3,4,10,1])
        prb = probe()
        prb[0] = shk

        prb.shanks[0][1]
        prb[0][1]
        
        returns:
        [(3, array([0., 0.])), (4, array([0., 0.])), (10, array([0., 0.])), (1, array([0., 0.]))]
        
        shank 0, group 1: ([ch, pos])
        prb.shanks[0].ch_group

    '''
    def __init__(self, fs=25000., nch=160, group_len=4, prb_type=None, grp_No=None, shank_no=None):
        super(probe, self).__init__()
        if shank_no is not None:
            self.shanks = {}
            for shank_id in range(shank_no):
                self.shanks[shank_id] = shank(shank_id)
        else:
            #TOD: load prb file
            pass

        self.type = prb_type
        self._n_ch = nch
        self._fs = fs
        self._group_len = group_len
        if grp_No is None:
            self._n_group = int(self._n_ch / self._group_len)
        else:
            self._n_group = grp_No
        self.sorting_status = np.zeros((self._n_group,)).astype(np.int)

    def auto_pos(self):
        if self.type == 'bow_tie':
            delta_y = 10
            for shank_id, shank in self.shanks.items():
                # print shank_id, shank
                # left side
                x, y = shank.xl, shank.yl
                for ch in shank.l:
                    shank.mapping[ch] = np.array([x,y])
                    y += delta_y
                # right side
                x, y = shank.xr, shank.yr
                for ch in shank.r:
                    shank.mapping[ch] = np.array([x,y])
                    y += delta_y
                # print shank.mapping
                self.mapping.update(shank.mapping)

        #shinsuke added	
        if self.type == 'neuronexus':
            delta_x = 3
            delta_y = 10
            for shank_id, shank in self.shanks.items():
                # print shank_id, shank
                # left side
                x, y = shank.xl, shank.yl
                for ch in shank.l:
                    shank.mapping[ch] = np.array([x,y])
                    x -= delta_x
                    y += delta_y
                # right side
                x, y = shank.xr, shank.yr
                for ch in shank.r:
                    shank.mapping[ch] = np.array([x,y])
                    x += delta_x
                    y += delta_y
                # print shank.mapping
                self.mapping.update(shank.mapping)


    def show(self, font_size=23):
        self.prb_view = probe_view()
        self.prb_view.set_data(self, font_size=font_size)
        self.prb_view.run()


    def save(self, filename):

        # for open-ephys gui and regular use
        self.n_ch = self._n_ch
        ch_list = np.hstack((self.chs, self.mask_chs))
        ch_dict = {}
        ch_dict['refs'] = {"channels": [-2, -1, -1, -1]}
        ch_dict['recording'] = {}
        ch_dict['recording']['channels'] = 175*[False]
        ch_dict['0'] = {}
        ch_dict['0']['enabled'] = 175*[True]
        ch_dict['0']['mapping'] = [int(_) for _ in (ch_list+1)]
        ch_dict['0']['reference'] = 175*[0]

        # for spiketag loading probe position mapping
        ch_dict['shank_no'] = [int(_) for _ in self.shanks.keys()]
        ch_dict['pos'] = {}
        for key, value in self.mapping.items():
            ch_dict['pos'][int(key)] = [int(_) for _ in value]
        self._ch_dict = ch_dict

        with open(filename, 'w') as fp:
            json.dump(ch_dict, fp, indent=4, separators=(',', ': '))


    def load(self, filename):
        with open(filename) as ff:
            prb_json = json.load(ff)
            for i in prb_json['pos'].keys():
                self.mapping[int(i)] = prb_json['pos'][i] 
            for i, chs in enumerate(np.array(prb_json['0']['mapping']).reshape(-1,4)):
                self.__setitem__(i, chs - 1)


    def _to_txt(self, filename):
        with open(filename,'w') as f:
            for ch, pos in self.mapping.items():
                _str = str(ch)+": " + str(pos) + '\n'
                f.write(_str)
            for grpNo in range(self.n_group):
                _str = "groupNo_" + str(grpNo) + ":" + str(self[grpNo]) + '\n'
                f.write(_str)


if __name__ == '__main__':
    from spiketag.base import ch_group, shank, probe
    # shk = shank()
    # shk[1] = ch_group([3, 4, 10, 1])
    prb = probe(shank_no=3)
    prb[0].l = [59,60,10,58,12,11,57,56]
    prb[0].r = [5,52,3,54,53,4,13,2,55]
    prb[0].x = -100.
    prb[0].y = 10

    prb[1].l = [15,63,48,47,0,61,9,14,62,6]
    prb[1].r = [8, 1,51,50,18,34,31,25,33,17,22]
    prb[1].x = -50.
    prb[1].y = 5

    prb[2].l = [39,38,20,45,44,24,7,32,16,23,46,30]
    prb[2].r = [19,37,21,35,36,26,29,40,27,42,41,28,43]
    prb[2].x = 0.
    prb[2].y = 0
    # print prb[0]
    prb.auto_pos()
    print(prb.mapping)
    # print prb[0][1]
    # print prb.shanks


# --------------------------------------------------------------------------------------------------------------------------------------------------
# 
# --------------------------------------------------------------------------------------------------------------------------------------------------
# 
# --------------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------
#        Probe Factory 
#------------------------------

# class ProbeFactory(object):
#     ''' probe factory: 
#             generate the probe by spicifying the type of probe.

#         type
#         ------
#         LinearProbe:
#                 fs : float
#                     sample rate
#                 n_ch : int
#                     number of channel
#                 ch_span : int
#                     cross up and down ch_span channel respectly 
#                 gmaps : dist
#                     the map from goup to channel, i.e: {1:[1,2,3]}. if gmaps is None, defaut sequence maps will be applied.   
#                     for linear probe, the ch == -1 means ch not exist the ch == -1 means ch not exists.
        
#         TetrodeProbe:
#                 fs : float 
#                     sample rate
#                 n_ch : int
#                     number of channel
#                 gmaps : dist
#                     the map from goup to channel, i.e: {1:[1,2,3,4]}. if gmaps is None, defaut sequence maps will be applied.   
#     '''
#     @staticmethod
#     def genLinearProbe(fs, n_ch, ch_span=1, gmaps=None):
#         assert n_ch > 0, 'amount of chs should be positive'
#         assert ch_span > 0, 'span between chs should be positive'
#         if not gmaps:
#             gmaps = {}
#             chmax = n_ch - 1
#             for center_ch in range(n_ch):
#                 chs = np.arange(center_ch - ch_span, center_ch + ch_span + 1, 1)
#                 chs[chs>chmax] = -1 
#                 chs[chs<0] = -1 
#                 gmaps[center_ch] = chs
#         return LinearProbe(fs, n_ch, ch_span, gmaps)

#     @staticmethod 
#     def genTetrodeProbe(fs, n_ch, gmaps=None):
#         assert n_ch > 0, 'amount of chs should be positive'
#         if not gmaps:
#             gmaps = {}
#             for g in range(n_ch/4):
#                gmaps[g] = np.arange(g*4, g*4+4)
#         return TetrodeProbe(fs, n_ch, gmaps)

# # -------------------------------
# #         Type of Probes
# # -------------------------------

# class Probe(object):
#     ''' the base class of probe
#     '''
#     def __init__(self, type):
#         assert type is not None
#         self._type = type

#     @property
#     def type(self):
#         '''
#           the string of type, some place need to distinguish the type 
#         '''
#         return self._type

#     @property
#     def fs(self):
#         return self._fs
    
#     @property
#     def n_ch(self):
#         return self._n_ch

#     @property
#     def n_group(self):
#         '''
#           number of groups, group can be treated as basic unit to statistic data
#         '''
#         return self._n_group
    
#     @property
#     def groups(self):
#         '''
#             return all group numbers
#         '''
#         for i in range(self._n_group):
#             yield i

#     @property
#     def len_group(self):
#         '''
#             number of channles in a group
#         '''
#         return self._len_group

#     def __str__(self):
#         return '\n'.join(['{}:{}'.format(key, val) for key, val in self._g2chs.items()]) 

#     __repr__ = __str__






# class LinearProbe(Probe):
#     ''' linear probe
#     '''
#     def __init__(self, fs, n_ch, ch_span, gmaps):
#         super(LinearProbe, self).__init__('linear')
        
#         assert fs > 0 
#         assert n_ch > 0
#         assert ch_span >0 and ch_span < n_ch
#         assert gmaps

#         self._fs = fs
#         self._n_ch = n_ch
#         self._ch_span = ch_span
#         self._n_group = self._n_ch
#         self._len_group = 2 * ch_span + 1
#         self._g2chs = gmaps
#         self._ch2g = {}
#         for g, chs in self._g2chs.items():
#             self._update_chs2group(chs, g)
        
   
#     @property
#     def ch_span(self):
#         return self._ch_span

#     # -------------------------------
#     #        public method 
#     # -------------------------------

#     def belong_group(self, ch):
#         '''
#             get the group number which ch belong
#         '''
#         assert ch >= 0 and ch < self._n_ch
#         return self._ch2g[ch] 

#     def get_chs(self, group):
#         '''
#             get the chs which group has
#         '''
#         assert group >= 0 and group < self._n_group
#         return self._g2chs[group]

#     def fetch_pivotal_chs(self, group):
#         '''
#             get the most important chs within group
#         '''
#         assert group >= 0 and group < self._n_group

#         chs = self._g2chs[group]
#         return np.asarray([chs[len(chs)/2]])

#     def _update_chs2group(self, chs, g):
#        for ch in chs:
#             self._ch2g[ch] = g

#     def __getitem__(self, group):
#         assert group >=0  and group < self._n_group, 'invalid group value'
#         return self._g2chs[group]

#     def __setitem__(self, group, chs):
#         assert group >=0  and group < self._n_group, 'invalid grup value'
#         assert isinstance(chs, np.ndarray), 'chs should numpy array, make sure numba works'
#         self._g2chs[group] = chs
#         self._update_chs2group(chs, group)





# class TetrodeProbe(Probe):
#     ''' tetrode probe
#     '''
#     def __init__(self, fs, n_ch, g2chs):
#         super(TetrodeProbe, self).__init__('tetrode')

#         assert fs > 0
#         assert n_ch > 0 and n_ch % 4 == 0
#         assert g2chs

#         self._fs = fs
#         self._n_ch = n_ch
#         self._n_group = int(self._n_ch / 4)
#         self._len_group = 4
#         self._g2chs = g2chs
#         self._ch2g = {}
#         for g, chs in self._g2chs.items():
#             self._update_chs2group(chs, g)
            
#     # -------------------------------
#     #        public method 
#     # -------------------------------

#     def get_chs(self, group):
#         '''
#             get the chs which group has
#         '''
#         assert group >= 0 and group < self._n_group
#         return self._g2chs[group] 

#     def belong_group(self, ch):
#         '''
#             return the group number which ch is belonged
#         '''
#         assert ch >= 0 and ch < self._n_ch
#         return self._ch2g[ch]

#     def fetch_pivotal_chs(self, group):
#         '''
#             get the most important chs within group
#         '''
#         assert group >= 0 and group < self._n_group
#         return self._g2chs[group]
    
#     def _update_chs2group(self, chs, g):
#        for ch in chs:
#             self._ch2g[ch] = g

#     def __getitem__(self, group):
#         assert group >= 0 and group < self._n_group, 'invalid group value'
#         return self._g2chs[group]

#     def __setitem__(self, group, chs):
#         assert group >= 0 and group < self._n_group, 'invalid group value'
#         assert len(chs) == 4, 'invalid amount of chs in tetrode'
#         assert isinstance(chs, np.ndarray), 'chs should numpy array, make sure numba works'
#         self._g2chs[group] = chs
#         self._update_chs2group(chs, group)

#     def fromfile(self, file, filetype='json'):
#         if filetype == 'json':
#             with open(file) as mapping_file:    
#                 data = json.load(mapping_file)
#         else:
#             print 'the file needs to be json'
#         tetrode_ch_hash = np.array(data['0']['mapping'])[:self._n_ch].reshape(-1,4) - 1
#         for i, _ch_hash in enumerate(tetrode_ch_hash):
#                 self[i] = _ch_hash

#     def ch_hash(self, ch):
#         return self.get_chs(self.belong_group(ch))
