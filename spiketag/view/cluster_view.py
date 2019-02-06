from vispy import app, scene, visuals
from vispy.util import keys
import numpy as np
from ..utils import EventEmitter, key_buffer



class cluster_view(scene.SceneCanvas):
    def __init__(self):
        scene.SceneCanvas.__init__(self, keys=None, title='clusters overview')
        self.unfreeze()
        self.view = self.central_widget.add_view()
        self.view.camera = 'panzoom'    
        # every group (grp_marker) has a clustering result each cluster will has a (clu_marker)
        # every group has its status: finish, unfinished
        # every clu has its status: spike counts, high quality, low quality, information bits
        self.grp_marker  = scene.visuals.Markers(parent=self.view.scene)
        self.nclu_text = scene.visuals.Text(parent=self.view.scene)
        self.event = EventEmitter() 


        # keybuf
        self.mode = ''
        self.key_buf = key_buffer()


    def set_data(self, clu_manager, selected_group_id=0, size=25):
        '''
        group_No is a scala number #grp
        nclu_list is a list with length = group_No
        '''
        
        self.clu_manager = clu_manager
        
        self.group_No = self.clu_manager.ngroup
        self.nclu_list = np.array(self.clu_manager.nclu_list)
        self.sorting_status = np.array(self.clu_manager.state_list)
        self.nspks_list = None
        self._size = size

        self.xmin = -0.02
        self.xmax =  0.04
        grp_x_pos = np.zeros((self.group_No,))
        grp_y_pos = np.arange(self.group_No)
        self.grp_pos = np.vstack((grp_x_pos, grp_y_pos)).T
        self.nclu_text_pos = np.vstack((grp_x_pos+0.02, grp_y_pos)).T
        
        self.current_group, self.selected_group_id = selected_group_id, selected_group_id

        self.color = self.generate_color(self.sorting_status, self.nspks_list, self.selected_group_id) 
        self.ecolor = np.zeros_like(self.color)
        self.ecolor[self.current_group] = np.array([0,1,0,1])

        self.grp_marker.set_data(self.grp_pos, symbol='square', 
                                 face_color=self.color, edge_color=self.ecolor, edge_width=4, size=size)
        
        self.nclu_text.text = [str(i) for i in self.nclu_list]
        self.nclu_text.pos  = self.nclu_text_pos
        self.nclu_text.color = 'g'
        self.nclu_text.font_size = size*0.50

        self.view.camera.set_range(x=[self.xmin, self.xmax])
        # self.view.camera.interactive = False
        
        if self.clu_manager._event_reg_enable:
            self.event_register()
        
        
    def event_register(self):
        @self.clu_manager.connect
        def on_update(state, nclu):
            self.refresh()
        #     print(state)
        #     print(nclu)
        self.clu_manager._event_reg_enable = not self.clu_manager._event_reg_enable


    def generate_color(self, sorting_status, nspks_list, selected_group_id):
        self.color = np.ones((self.group_No, 4)) * 0.5
        self.color[sorting_status==0] = np.array([1,1,1, .3]) # IDLE
        self.color[sorting_status==1] = np.array([1,0,0, 1.]) # BUSY
        self.color[sorting_status==2] = np.array([0,1,0, .7]) # READY
        self.color[sorting_status==3] = np.array([1,1,0, .8]) # DONE
        if nspks_list is not None:
            self.transparency = np.array(nspks_list)/np.array(nspks_list).max()
            self.color[:, -1] = self.transparency
        return self.color 


    def on_key_press(self, e):
        if e.text == 'r':
            self.view.camera.set_range(x=[self.xmin, self.xmax])
        if e.text == 'k':
            self.moveto(self.next_group)
        if e.text == 'j':
            self.moveto(self.previous_group)
        if e.text == 'd':
            self.set_cluster_done(self.current_group)
        if e.text == 'u':
            self.set_cluster_undone(self.current_group)
        if e.text == 'o':
            self.select(self.current_group)

        ## key for backend clustering ##
        if e.text == 'b':
            self.mode = 'backend'

        if e.text.isdigit():
            self.key_buf.push(e.text)

        if e.text == 'g':
            if self.mode == '':
                selected_group = int(self.key_buf.pop()) 
                print('moveto {}'.format(selected_group))
                if selected_group in range(self.clu_manager.ngroup):
                    self.moveto(selected_group)
                    self.select(selected_group)
            if self.mode == 'backend':
                n_comp=int(self.key_buf.pop())
                if 1<n_comp<40:
                    self.clu_manager.emit(self.mode, 
                                          method='dpgmm', n_comp=n_comp)
                else:
                    print('the number you type has to be between (1, 20)')
                self.mode = ''

        if e.text == 'h':
            if self.mode == 'backend':
                min_cluster_size=int(self.key_buf.pop())
                if 1<min_cluster_size<100:
                    self.clu_manager.emit(self.mode, 
                                          method='hdbscan', min_cluster_size=min_cluster_size)
                else:
                    print('the number you type has to be between (1, 100)')
                self.mode = ''


    @property
    def cpu_ready_list(self):
        return np.where(self.sorting_status==1)[0]

    def set_cluster_ready(self, grp_id):
        self.sorting_status[grp_id] = 1
        self.refresh()

    def set_cluster_done(self, grp_id):
        self.clu_manager[grp_id] = 'DONE'
        self.refresh()

    def set_cluster_undone(self, grp_id):
        self.clu_manager[grp_id] = 'READY'
        self.refresh()

    def refresh(self):
        self.set_data(clu_manager=self.clu_manager, selected_group_id=self.current_group, size=self._size)


    @property
    def previous_group(self):
        if self.current_group>0:
            self._previous_group = self.current_group - 1
            return self._previous_group
        else:
            self._previous_group = 0
            return self._previous_group 


    @property
    def next_group(self):
        if self.current_group<self.group_No-1:
            self._next_group = self.current_group + 1
            return self._next_group
        else:
            self._next_group = self.group_No-1
            return self._next_group 


    def moveto(self, group_id):
        self.current_group = group_id
        self.set_data(self.clu_manager, self.current_group, self._size) 


    def select(self, group_id):
        self.clu_manager.emit('select', group_id=self.current_group)


    def run(self):
        self.show()
        self.app.run()
