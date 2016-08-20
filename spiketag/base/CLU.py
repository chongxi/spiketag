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

    def _subset_idx(self, selected_clu, subset, targeted_clu=None):
        '''
        get the global id of a subset in selected_clu if targeted_clu is None
        or
        get the local  id of a subset in selected_clu in targeted_clu
        '''
        if targeted_clu == None:
            '''
            get the global id of a subset in selected cluster
            '''
            return self._glo_id(selected_clu, subset)

        else:
            '''
            get the local id in targeted cluster of a subset in selected cluster
            The solved problem is:
            subset in selected cluster, if such subset is in targeted cluster, what is
            the idx?
            '''
            targeted_idx = np.where(self.membership==targeted_clu)[0]
            selected_idx = np.where(self.membership==selected_clu)[0][subset]
            subset_idx  = np.searchsorted(targeted_idx, selected_idx)
            subset_idx.sort()
            if selected_clu != targeted_clu:
                subset_idx += np.arange(len(subset_idx))
            return subset_idx


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


    def move(self, clu_from, subset, clu_to):
        '''
        move subset from clu_from to clu_to
        move(2, [2,3,4,5,6], 1): move the subset [2,3,4,5,6]
                                 from clu2
                                 to clu1
        '''
        with Timer('get_subset', verbose=False):
            subset_idx_from   = list(subset)
            subset_idx_global = self._subset_idx(clu_from, subset_idx_from)
            subset_idx_to     = self._subset_idx(clu_from, subset_idx_from, clu_to)
        with Timer('reassign membership', verbose=False):
            self.membership[subset_idx_global] = clu_to
        with Timer('reconstruct', verbose=False):
            self.__construct__()
        with Timer('emit move signal', verbose=False):
            self.emit('cluster', action = 'move')
        return subset_idx_to


    def split(self, clu_from, subset):
        clu_to = self.index_id.max()+1
        self.move(clu_from, subset, clu_to)


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

