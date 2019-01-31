from vispy import app, scene, visuals
from vispy.util import keys
import numpy as np
from ..utils import EventEmitter


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


    def set_data(self, group_No, nclu_list, sorting_status=None, selected_group_id=None, nspks_list=None, size=25):
        '''
        group_No is a scala number #grp
        nclu_list is a list with length = group_No
        '''
        self.group_No = group_No
        self.nclu_list = nclu_list
        self.sorting_status = sorting_status
        self.nspks_list = nspks_list
        self._size = size

        self.xmin = -0.02
        self.xmax =  0.04
        grp_x_pos = np.zeros((group_No,))
        grp_y_pos = np.arange(group_No)
        self.grp_pos = np.vstack((grp_x_pos, grp_y_pos)).T
        self.nclu_text_pos = np.vstack((grp_x_pos+0.02, grp_y_pos)).T

        if selected_group_id is None and group_No>1:
            selected_group_id = np.min(np.where(self.sorting_status==1)[0])
            self.current_group = selected_group_id
        elif group_No == 1:
            self.current_group = 0
            self._previous_group = 0
            self._next_group = 0
        else:
            self.current_group = selected_group_id

        self.color = self.generate_color(sorting_status, nspks_list, selected_group_id) 

        self.grp_marker.set_data(self.grp_pos, symbol='square', face_color=self.color, size=size)
        self.nclu_text.text = [str(i) for i in nclu_list]
        self.nclu_text.pos  = self.nclu_text_pos
        self.nclu_text.color = 'g'
        self.nclu_text.font_size = size*0.50

        self.view.camera.set_range(x=[self.xmin, self.xmax])
        # self.view.camera.interactive = False


    def generate_color(self, sorting_status, nspks_list, selected_group_id):
        self.color = np.ones((self.group_No, 4)) * 0.5
        self.color[sorting_status==0] = np.array([1,1,1, .2]) # cpu busy at automatic sorting
        self.color[sorting_status==1] = np.array([0,1,1, .3]) # cpu ready for manual sorting
        self.color[sorting_status==2] = np.array([1,0,1, .3]) # manual sorting is done
        # self.color[selected_group_id] = np.array([1,1,1,  1]) # selected group id (current_group)
        self.color[selected_group_id, -1] = 1
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
            self.moveto(self.next_group)
        if e.text == 'o':
            self.select(self.current_group)

    @property
    def cpu_ready_list(self):
        return np.where(self.sorting_status==1)[0]

    def set_cluster_ready(self, grp_id):
        self.sorting_status[grp_id] = 1
        self.refresh()

    def set_cluster_done(self, grp_id):
        self.sorting_status[grp_id] = 2
        self.refresh()

    def refresh(self):
        self.set_data(self.group_No, self.nclu_list, self.sorting_status, self.current_group, self.nspks_list, self._size)


    @property
    def previous_group(self):
        if self.current_group>0:
            self._previous_group = self.current_group - 1
            return self._previous_group
        else:
            return self._previous_group 


    @property
    def next_group(self):
        if self.current_group<self.group_No-1:
            self._next_group = self.current_group + 1
            return self._next_group
        else:
            return self._next_group 


    def moveto(self, group_id):
        self.current_group = group_id
        self.set_data(self.group_No, self.nclu_list, self.sorting_status, self.current_group, self.nspks_list, self._size) 


    def select(self, group_id):
        # if self.sorting_status[group_id] != 0:
        self.event.emit('select', group_id=self.current_group)
        # else:
            # print('unable to select busy cpu {}'.format(self.current_group)) 


    def run(self):
        self.show()
        self.app.run()

