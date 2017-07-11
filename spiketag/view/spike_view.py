import numpy as np
import numexpr as ne
from collections import OrderedDict
from phy.plot import View, base, visuals
from vispy.util.event import Event
from .color_scheme import palette
from ..utils.utils import Timer
from ..utils import conf 
from ..utils.conf import error, warning
from ..base.CLU import CLU
from ._core import _get_array, _accumulate
from ._core import _spkNo2maskNo_numba, _cache_out, _cache_in_vector, _cache_in_scalar, _representsInt 


class spike_view(View):
   
    def __init__(self, interactive=True):
        super(spike_view, self).__init__('grid')
        self.palette = palette
        self._transparency = 0.2
        self._highlight_color = np.array([1,0,0,1]).astype('float32')
        self._data_bound = (-2, -1, 2, 1)
        self.interactive = interactive
        self._selected = {}
        self._view_lock = True
        self.events.add(model_modified=Event)
        
    def attach(self, gui):
        gui.add_view(self)

    def _affine_transform(self, x):
        # important!, this map the spk to view space through a affine transformation: y = ax + b
        # self.y = self._affine_transform(self.spk.transpose(2,0,1).ravel())
        # that gives the affine transformation y = ax + b
        a,b = self._get_affine_params(x.min(), x.max())
        y = a*x + b
        return y

    def _get_affine_params(self, min, max):
        '''
            get the affine param which transfer [min, max] to [-1, 1]
        '''
        a = np.ones((2,2), dtype=np.int32)
        a[0,0], a[1,0] = min, max
        b = np.array([-1,1])
        r = np.linalg.solve(a,b)
        return r[0], r[1]

    def _get_data(self, data_bound):
        new_data_list=[]
        y=[]
        for chNo in range(self.n_ch):
            for cluNo in self.clu.index_id:
                # x           = [self._xsig for i in y]
                # y           = list(self.spk[self.clu[cluNo],:,chNo].squeeze())
                s           = self.spk[self.clu.index[cluNo],:,chNo].squeeze()
                if s.ndim == 1:
                    s = s[np.newaxis,:]
                n = s.shape[0]
                y.append(np.asarray(s))
                transparency= np.tile(self._transparency, (n,1))
                color       = np.hstack((np.asarray([self.palette[cluNo] for i in s]), transparency)).astype(np.float32)
                # depth       = np.zeros((n,1))
                # data_bounds = np.tile((data_bound), (n,1))
                box_index   = np.tile((chNo, cluNo), ((n*self.n_samples,1))).astype(np.float32)
                new_data_list.append({# 'y': y,
                                      # 'x': x,
                                      'color': color,
                                      # 'depth': depth,
                                      # 'data_bounds': data_bounds,
                                      'box_index': box_index})
        self.data = _accumulate(new_data_list)
        self._xsig = np.linspace(-0.5, 0.5, self.n_samples)
        self.x = np.tile(self._xsig, self.n_ch*self.n_signals)
        self.y = np.vstack(y).ravel()

    def _build(self):
        self.grid.shape = (self.n_ch, self.clu.nclu)
        data = self.data
        self.box_index = data.pop('box_index')
        self.depth = np.c_[self.x, self.y, np.zeros(*self.x.shape)].astype(np.float32)
        self.color = np.repeat(data['color'], self.n_samples, axis=0)
        self._cache_depth = self.depth.copy()
        self._cache_color = self.color.copy()
        self._cache_mask_ = np.array([])

    def set_data(self, spk, clu=None):
        #################################
        # this init block take about 1ms
        self.spk = self._affine_transform(spk)
        if clu is None:
            self.clu = CLU(np.zeros(spk.shape[0]).astype('int'))
        else:
            self.clu = clu
        self.n_signals = spk.shape[0]
        self.n_samples = spk.shape[1]
        self.n_ch      = spk.shape[2]
        #################################
        self.clear()

        ###################################
        # render take 400ms for 5 million points
        self.render()
        
        @self.clu.connect
        def on_cluster(*args, **kwargs):
            with Timer('rerender', verbose=conf.ENABLE_PROFILER):
                self._selected = {}
                self.rerender()
        
        @self.clu.connect
        def on_select(*args, **kwargs):
            self._selected = self.clu.global2local(self.clu.selectlist)
            self.highlight(self._selected, external=True)

    def render(self, update=False):

        visual = visuals.PlotVisual()
        self.add_visual(visual)
        
        with Timer('render - step 0: get data', verbose=conf.ENABLE_PROFILER):
            self._get_data(self._data_bound)

        with Timer('render - step 1: set data', verbose=conf.ENABLE_PROFILER):
            self._build()
            self.signal_index = np.repeat(np.arange(len(self.y)/len(self._xsig)), 
                                              len(self._xsig)).astype(np.float32)


        with Timer('render - step 2: gsgl update', verbose=conf.ENABLE_PROFILER):
            visual.program['a_position'] = self.depth
            visual.program['a_color'] = self.color
            visual.program['a_signal_index'] = self.signal_index
            visual.program['a_box_index'] = self.box_index


    def rerender(self, clu=None, data_bound=None):
        '''
        called when you want to rerender the new clustering 
        '''
        if clu is not None:
            self.clu = clu
        nclu = self.clu.nclu
        if data_bound is None:
            data_bound = (-1,-1,1,1)

        with Timer('rerender - step 0: get data', verbose=conf.ENABLE_PROFILER):
            self._get_data(data_bound)
        
        with Timer('rerender - step 1: set data', verbose=conf.ENABLE_PROFILER):
            self._build()

        with Timer('rerender - step 2: gsgl update', verbose=conf.ENABLE_PROFILER):
            # [self._a_pos, self._a_color, self._a_index] = _pv_set_data(self.visuals[0], **data)
            self.visuals[0].program['a_position'] = self.depth
            self.visuals[0].program['a_color'] = self.color
            self.visuals[0].program['a_box_index'] = self.box_index


    def clear(self):
        """Reset the view."""
        self._items = OrderedDict()
        self.visuals = []
        self.update()


    def clear_virtual(self):
        """Reset the virtual layer"""
        if len(self.visuals) > 1:
            self._items = OrderedDict()
            self.visuals.pop()
            self.update()
        else:
            pass


    def plot_virtual(self, cluNo):
        '''
        plot self.selected_spk in the column where mouse is hovering on:
        1. selected_spk is global_idx of the selected spks
        2. cluNo is the column where the mouse is hovering on
        '''
        self.clear_virtual()
        l = len(self.selected_spk)
        if l > 100:
            subsample_id = np.random.choice(l, 100, replace=False)
            virtual_spkNo = self.selected_spk[subsample_id]
        else:
            virtual_spkNo = self.selected_spk
        for chNo in range(self.n_ch):
            self[chNo, cluNo].plot(y = self.spk[virtual_spkNo,:,chNo].squeeze(),
                                   color = self._highlight_color,
                                   data_bounds=self._data_bound )
        self.build()

    @property
    def transparency(self):
        return self._transparency

    @transparency.setter
    def transparency(self, v):
        self._transparency = v
        if self._transparency >= 0.9:
            self._transparency = 0.9
        elif self._transparency <= 0.001:
            self._transparency = 0.001
        self.color[:,-1] = self._transparency
        self._cache_color[:,-1] = self._transparency
        self.visuals[0].program['a_color'] = self.color
        self.update()


    # ---------------------------------------------
    def _spkNo2maskNo(self, spkNolist, cluNo):
        '''
        only for highlight, calculate view mask
        accelerated by numba
        '''
        spkNolist = np.array(list(spkNolist)).astype('int32')
        n_signals=self.n_signals
        n_samples=self.n_samples
        n_ch     =self.n_ch
        clu_offset = self.clu._clu_cumsum
        N = n_samples*n_ch*len(spkNolist)
        mask = np.zeros((N,)).astype('int64')
        # below is a numba func
        _spkNo2maskNo_numba(n_signals, n_samples, n_ch, clu_offset,
                            cluNo, spkNolist, mask)
        if any(mask) > 0:
            return mask
        else:
            return np.array([])

    # ---------------------------------------------
    def highlight(self, selected, external=False, refresh=True):
        """
        highlight the selected spikes:
        the selected is dist, eg: {cluNo:[spikelist]}, the spike list is local idx in the clu, and the num of clu could be one or more
        """
        with Timer('is external', verbose=conf.ENABLE_PROFILER):
            if external:
                self.clear_virtual()

        with Timer('is refresh', verbose=conf.ENABLE_PROFILER):
            if refresh:
                self._clear_highlight()

        with Timer('do highlight', verbose=conf.ENABLE_PROFILER):
            for k,v in selected.iteritems():
                self._highlight(v,k,refresh=False)

    def _highlight(self, spkNolist, cluNo, refresh=True):
        """
        highlight the nth spike in cluNo, nth is local id
        only local ids and cluNo is needed
        now is optimized enough few hundreds spikes take 8ms
        if refresh is True then previous highlight is removed 
        if refresh is False then previous highlight would be preserved
        accelerated by numba
        """
        try:
            with Timer('get_view_mask', verbose=conf.ENABLE_PROFILER):
                view_mask = self._spkNo2maskNo(spkNolist=spkNolist, cluNo=cluNo)
                n_view_mask = len(view_mask)
                n_cache_mask = len(self._cache_mask_)

            with Timer('render mask', verbose=conf.ENABLE_PROFILER):
                if refresh and n_cache_mask>0:
                    _cache_out(self._cache_mask_, self._cache_color, self.color)
                    _cache_out(self._cache_mask_, self._cache_depth, self.depth)
                    self._cache_mask_ = np.array([])

            with Timer('update view', verbose=conf.ENABLE_PROFILER):
                if n_view_mask>0:
                    # selected_color = np.hstack((self._highlight_color[:3], self.transparency))
                    _cache_in_vector(view_mask, self._highlight_color, self.color)
                    _cache_in_scalar(view_mask, -1, self.depth[:,2].reshape(-1,1))
                    self._cache_mask_ = np.hstack((self._cache_mask_, view_mask)).astype('int64')
                    # self.color[view_mask,-1] = 1
                    self.visuals[0].program['a_color']    = self.color
                    self.visuals[0].program['a_position'] = self.depth  # pos include depth
                    self.update()

        except Exception, e:
            pass

    def _clear_highlight(self):
          if len(self._cache_mask_) > 0:
                _cache_out(self._cache_mask_, self._cache_color, self.color)
                _cache_out(self._cache_mask_, self._cache_depth, self.depth)
                self._cache_mask_ = np.array([])
                self.update()


    def _data_in_box(self, box):
        ch_No = box[0]
        clu_No = box[1]
        data_in_box = self.spk[self.clu[clu_No], :, ch_No].squeeze()
        return data_in_box

    def _get_closest_spk(self, box, pos):
        data = self._data_in_box(box)
        nearest_x = abs(pos[0] - self._xsig).argmin()
        ref  = pos[1]
        pool = data[:, nearest_x]
        nearest_spkNo = abs(pos[1] - data[:, nearest_x]).argmin()
        return nearest_spkNo

    def _get_close_spks(self, box, pos, lpos=None):
        data = self._data_in_box(box)
        nearest_x = abs(pos[0] - self._xsig).argmin()
        ref  = pos[1]
        pool = data[:, nearest_x]
        if lpos is not None:
            top, bottom = (ref, lpos[1]) if ref > lpos[1] else (lpos[1], ref)
            expr = "(pool < top) & (pool > bottom)" 
        else:
            expr = "abs(ref - pool) < 0.005"
        close_spkNolist = np.where(ne.evaluate(expr))[0]
        return close_spkNolist

    @property
    def selected_clus(self):
        return self._selected.keys()

    @property
    def selected_spk(self):
        selected_spk = self.clu.local2global(self._selected)
        return selected_spk
    
    @property
    def is_single_mode(self):
        '''
            if only have one cluster selected, return True, otherwise return False 
        '''
        return len(self._selected) == 1

    @property
    def selected_cluster(self):
        '''
            if on single mode, return the only cluster no we have
        '''
        if self.is_single_mode:
            return self._selected.keys()[0]

    @property
    def selected_whole_cluster(self):
        '''
            if at least one cluster whole selected, return True, else return False 
        '''
        for k,v in self._selected.iteritems():
            if len(v) == self.clu.index_count[k]:
                return True
        return False

    @property
    def is_spk_empty(self):
        if len(self._selected) > 0:
            for k,v in self._selected.iteritems():
                if len(v) > 0:
                    return False
        return True

    def _reset_cluster(self):
        for k,v in self._selected.iteritems():
            self._selected[k] = np.array([])

    def _move_spikes(self, target_clu_no):
        if not self.is_spk_empty and self.selected_cluster != target_clu_no:
            # cluster takes 100 ms + on_cluster event handler take 700ms
            if self.selected_whole_cluster is False:  # move
                try:
                    with Timer('move', verbose=conf.ENABLE_PROFILER):
                        target_local_idx = self.clu.move(self._selected,
                                                        target_clu_no)
                        self._selected = {target_clu_no:target_local_idx}
                except IndexError, e:
                    error("Move spikes failure: {}".format(e))
                    error("Move selected spikes {} to target clu no {}. ".format(self._selected, target_clu_no))
                    error("Current cluster index is {}.".format(self.clu.index))
                    return 

            if self.selected_whole_cluster is True:   # merge
                try:
                    global_idx = self.clu.local2global(self._selected)
                    with Timer('merge', verbose=conf.ENABLE_PROFILER):
                        self.clu.merge(np.append(self._selected.keys(),target_clu_no))
                    self._selected = self.clu.global2local(global_idx)
                except IndexError, e:
                    error("Merge spikes failure: {}".format(e))
                    error("Merge selected spikes {} to target clu no {}".format(self._selected, target_clu_no))
                    error("Current cluster index is {}.".format(self.clu.index))

            with Timer('highlight', verbose=conf.ENABLE_PROFILER):
                self.highlight(selected=self._selected) 

    def on_mouse_press(self, e):
        if e.button == 1:
            ndc = self.panzoom.get_mouse_pos(e.pos)
            box = self.interact.get_closest_box(ndc)

            if not self.is_spk_empty:
                if self.view_lock is True:
                    target_clu_No = box[1]
                    self._move_spikes(target_clu_No)  

        if e.button == 3:
            self.clear_virtual()
            self._reset_cluster()

    def on_mouse_move(self, e):
        '''
        selected cluster 
        selected spikes 
        for highlight 
        '''
        try:
            self.set_current()
            if self.interactive is True:
                ndc = self.panzoom.get_mouse_pos(e.pos)
                box = self.interact.get_closest_box(ndc)
                tpos = self.get_pos_from_mouse(e.pos, box)[0]
                
                modifiers = e.modifiers
                if modifiers is () and not isinstance(e.button, int):
                    if self.view_lock is False:
                        self.clear_virtual()
                        spkNo = self._get_closest_spk(box, tpos) # one number
                        self._selected = {box[1]:(spkNo,)}
                        self.highlight(selected=self._selected)
                        self.clu.select(self.selected_spk, caller=self.__module__)
                    else:
                        if len(self._selected) > 0: # there are spikes are selected
                            # self._spkNolist = set()
                            self.plot_virtual(cluNo=box[1])


                if modifiers is not ():
                    selected_cluster = box[1]
                    if len(modifiers) ==2 and modifiers[0].name == 'Shift' and modifiers[1].name == 'Control':
                        '''
                        spikes selection mode
                        '''
                        self.clear_virtual()
                        
                        last_ndc = self.panzoom.get_mouse_pos(e.last_event.pos)
                        last_box = self.interact.get_closest_box(last_ndc)
                        if last_box == box: 
                            last_tpos = self.get_pos_from_mouse(e.last_event.pos, last_box)[0]
                            close_spkNolist = self._get_close_spks(box, tpos, last_tpos)
                        else:
                            close_spkNolist = self._get_close_spks(box, tpos)
                        self.highlight(selected={selected_cluster:close_spkNolist},refresh=False)
                        self._selected[selected_cluster] = np.union1d(self._selected.get(selected_cluster, np.array([], dtype=np.int64)), close_spkNolist)
                        #  self._selected[selected_cluster] = set(self._selected.get(selected_cluster,np.array([]))).union(set(close_spkNolist))
                        # self.on_select()
                        self.clu.select(self.selected_spk, caller=self.__module__)
                    
                    elif len(modifiers) ==1 and modifiers[0].name == 'Control':
                        '''
                        spikes observation mode
                        '''
                        self.clear_virtual()
                        self._selected = {selected_cluster:self._get_close_spks(box,tpos)}
                        self.highlight(selected=self._selected)
                        self.clu.select(self.selected_spk, caller=self.__module__)
                    
                    elif len(modifiers) ==1 and modifiers[0].name == 'Shift':
                        '''
                        spikes trim mode
                        '''
                        self.clear_virtual()
                        close_spkNolist = self._get_close_spks(box, tpos)
                        intersect_spks_idx = np.where(np.in1d(self._selected[selected_cluster], close_spkNolist))[0]
                        if len(intersect_spks_idx)>0:
                            self._selected[selected_cluster] = np.delete(self._selected[selected_cluster], intersect_spks_idx)
                        self.highlight(selected=self._selected) 
                        # self.on_select()
                        self.clu.select(self.selected_spk, caller=self.__module__)

        except Exception, e:
            pass


    @property
    def view_lock(self):
        return self._view_lock

    @view_lock.setter
    def view_lock(self, v):
        #  self._reset_cluster()
        self._view_lock = v


    def on_mouse_wheel(self, e):
        modifiers = e.modifiers
        if modifiers is not ():
            if modifiers[0].name == 'Control':
                self.transparency *= np.exp(e.delta[1]/4)



    def on_key_press(self, e):

        if e.text == 'c':
            self.view_lock = not self.view_lock
            if self.is_single_mode and not self.is_spk_empty:
                self.highlight(selected=self._selected) 

        if e.text == 'z':
            self.clu.undo()

        if e.text == 'a':
            if self.is_single_mode:
                all_spkNolist = np.arange(self.clu[self.selected_cluster].size)
                self._selected[self.selected_cluster] = all_spkNolist
                self.highlight(selected=self._selected)   
                self.clu.select(self.selected_spk, caller=self.__module__)

        if e.text == 's':
            if not self.is_spk_empty:
                target_clu_No = max(self.clu.index_id) + 1
                self._move_spikes(target_clu_No)
        
        if e.text == 'd':
            if len(self.selected_spk) > 0:
                with Timer('Delete spks from spk view.', verbose=conf.ENABLE_PROFILER): 
                    self.events.model_modified(Event('delete'))
                    self._selected = {}

        if _representsInt(e.text):
            ### assign selected spikes to cluster number ###
            if not self.is_spk_empty:
                target_clu_No = int(e.text)
                self._move_spikes(target_clu_No)

        if e.modifiers is not () and e.modifiers[0].name == 'Control' and e.key == 'Left':
            if self.is_single_mode and (not self.is_spk_empty):           
                target_clu_No = self.selected_cluster-1 if self.selected_cluster>=1 else 0
                self._move_spikes(target_clu_No)

        elif e.modifiers is not () and e.modifiers[0].name == 'Control' and e.key == 'Right':
            if self.is_single_mode and (not self.is_spk_empty):           
                target_clu_No = self.selected_cluster+1 if self.selected_cluster<max(self.clu.index_id)+1 else max(self.clu.index_id)+1
                self._move_spikes(target_clu_No)


