import numpy as np
from ..utils.utils import EventEmitter
from ..utils.utils import Timer


class CLU(EventEmitter):
    """docstring for Clu"""
    def __init__(self, clu):
        super(CLU, self).__init__()
        self.membership = clu.copy()
        while min(self.membership) < 0:
            self.membership += 1
        self._stack_init_()
        self.__construct__()

    def _stack_init_(self):
        self._membership_undo_stack = []
        self._membership_undo_stack.append(self.membership.copy())

        @self.connect
        def on_reverse(*args, **kwargs):
            if kwargs['action'] != 'undo':
                self._membership_undo_stack.append(self.membership.copy())
            else: #TODO: redo
                pass
        #       for key, value in kwargs.items():
        #               print "{}: {}".format(key, value)

    def __construct__(self):
        '''
        construct the base properties
        nclu:        number of clusters
        index:       dict of {id: index}
        index_count: dict of {id: counts}
        index_id:    array of id
        '''
        self.index       = {}
        self.index_count = {}
        self.index_id    = np.unique(self.membership)
        self.index_id.sort()
        self.make_id_continuous()

        self.nclu        = len(self.index_id)
        _counts_per_clu = [0,]
        for cluNo in self.index_id:
                self.index[cluNo] = np.where(self.membership==cluNo)[0]
                self.index_count[cluNo] = len(self.index[cluNo])
                _counts_per_clu.append(self.index_count[cluNo])
        self._clu_cumsum = np.cumsum(np.asarray(_counts_per_clu))

    def make_id_continuous(self):
        if self._is_id_discontineous():
            for i in range(1, self.index_id.size):
                self.membership[self.membership==self.index_id[i]] = i
            self.index_id    = np.unique(self.membership)
            self.index_id.sort()
    
    def global2local(self,global_idx):
        '''
            get the local id and clu_no from the global idx, the result is dist,eg:
            {0:[1,2,3],1:[1,2,3]}
        '''
        local_idx = {}

        if isinstance(global_idx,int):
            global_idx = np.array([global_idx])

        if len(global_idx):
            clus_nos = np.unique(self.membership[global_idx])
        
            for clus_no in clus_nos:
                clus_no = int(clus_no)
                local_global_idx = np.intersect1d(self.index[clus_no],global_idx,assume_unique=True)
                sub_local_idx = np.searchsorted(self.index[clus_no], local_global_idx)
                sub_local_idx.sort()
                local_idx[clus_no] = sub_local_idx
        
        return local_idx
 
        
    def local2global(self,local_idx):
        '''
            get the global id from the clus, the clus is a dict, including the clu_no and sub_idex,eg:
            {0:[1,2,3],1:[1,2,3]}
        '''
        global_idx = np.array([],dtype='int')

        for clu_no, local_idx in local_idx.iteritems():
            global_idx = np.append(global_idx,self._glo_id(clu_no, local_idx))
        global_idx.sort()

        return global_idx
            

    def _is_id_discontineous(self):
        if any(np.unique(self.index_id[1:]-self.index_id[:-1]) > 1):
            return True
        else:
            return False

    def __getitem__(self, i):
        if i in self.index_id:
            return self.index[i]
        else:
            print('cluster id should be in {}'.format(self.index_id))

    def __str__(self):
        for cluNo, index in self.index.items():
            print(cluNo, index)
        return str(self.index_count)

    def _glo_id(self, selected_clu, sub_idx):
        '''
        get the global id of a subset in selected_clu
        '''
        if type(sub_idx) is not list:
            sub_idx = list(sub_idx)
        glo_idx = self.index[selected_clu][sub_idx]
        glo_idx.sort()
        return glo_idx                

    def _sub_id(self, global_idx):
        '''
        get the local id and sub_clu from the global idx
        '''
        sub_clu = np.unique(self.membership[global_idx])
        if len(sub_clu) == 1:
            sub_clu = int(sub_clu)
            sub_idx = np.searchsorted(self.index[sub_clu], global_idx)
            sub_idx.sort()
            return sub_clu, sub_idx
        else:
            print('goes to more than one cluster')


    def select(self, selectlist):
        '''
            select global_idx of spikes
        '''
        self.selectlist = selectlist # update selectlist
        self.emit('select', action='select')
    
    def select_clu(self, selected_clu_list):
        '''
            select clu_id
        '''
        self.select_clu = selected_clu_list
        self.emit('select', action='select_clu')

    def merge(self, mergelist):
        '''
        merge two or more clusters, target cluNo is the lowest id
        merge([0,3]): merge clu0 and clu3 to clu0
        merge([2,6,4]): merge clu2,4,6, to clu2
        '''
        clu_to = min(mergelist)
        for cluNo in mergelist:
            self.membership[self.membership==cluNo] = clu_to
        self.__construct__()
        self.emit('cluster', action='merge')

    def move(self,clus_from,clu_to):
        '''
          move subsets from clus_from to clu_to, the clus_from is dict which including at least one clu and the subset in this clu,eg:
          clus_from = {1:[1,2,3,4],2:[2,3,4,5]}, move these all spk to the clu_to 1
        '''
        selected_global_idx = self.local2global(clus_from)

        self.membership[selected_global_idx] = clu_to
        self.__construct__()
        
        self.emit('cluster', action = 'move')
        
        return self.global2local(selected_global_idx)[clu_to]
    

    def split(self, clus_from):
        clu_to = self.index_id.max()+1
        self.move(clus_from, clu_to)


    def undo(self):
        if len(self._membership_undo_stack)>1:
            self._membership_undo_stack.pop()
            self.membership = self._membership_undo_stack[-1]
            self.__construct__()
            self.emit('reverse', action = 'undo')
        else:
            print('out of stack')


    def redo(self):
        # TODO: add redo stack
        pass






# class CLU(Clustering):
#     def __init__(self, clu):
#         super(CLU, self).__init__(clu)
#         self.__init()
        
#     def __init(self):
#         self.spike_id     = {}
#         self.spike_counts = {}
#         self.update()
#         @self.connect
#         def on_cluster(up):
#             self.update()
    

#     def update(self):
#         self.spike_counts = {}
#         self.spike_id     = {}
#         for cluNo in self.cluster_ids:
#             self.spike_id[cluNo]     = self.spikes_in_clusters((cluNo,))
#             self.spike_counts[cluNo] = self.spikes_in_clusters((cluNo,)).shape[0]


#     def __getitem__(self, cluNo):
#         if cluNo in self.cluster_ids:
#             return self.spikes_in_clusters((cluNo,))
#         else:
#             print "out of possible index\ncluster_ids: {}".format(self.cluster_ids)

#     def raster(self, method='vispy', toi=None, color='k'):
#         """
#         Creates a raster plot
#         Parameters
#         ----------
#         toi:    [t0, t1]
#                 time of interest 
#         color : string
#                 color of vlines 
#         Returns
#         -------
#         ax : an axis containing the raster plot
#         """
#         if method == 'matplotlib':
#             fig = plt.figure()
#             ax = plt.gca()
#             for ith, trial in self.spike_id.item():
#                 ax.vlines(trial, ith + .5, ith + 1.5, color=color)
#             ax.set_ylim(.5, len(self.spktime) + .5)
#             ax.set_xlim(toi)
#             ax.set_xlabel('time')
#             ax.set_ylabel('cell')
#             fig.show()
#             return fig, ax
    
#         elif method == 'vispy':
#             rview = raster_view()
#             rview.set_data(timelist=self.spktime, color=(1,1,1,1))
# #             rview.show()
#             return rview

