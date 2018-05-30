import numpy as np
from vispy import app, scene, visuals
from vispy.util import keys
from vispy.color import Color
from collections import OrderedDict
from sklearn.neighbors import KDTree
from itertools import combinations as comb
from ..utils.utils import Picker


white = Color("#ecf0f1")
gray = Color("#121212")
red = Color("#e74c3c")
blue = Color("#2980b9")
orange = Color("#e88834")


def star(inner=0.5, outer=1.0, n=5):
    R = np.array([inner, outer] * n)
    T = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    P = np.zeros((2 * n, 3))
    P[:, 0] = R * np.cos(T)
    P[:, 1] = R * np.sin(T)
    return P


def rec(left=-15, right=15, bottom=-25, top=25):
    P = np.zeros((4, 3))
    R = np.array([[left,  bottom],
                  [right, bottom],
                  [right, top   ],
                  [left,  top   ]])
    P[:, :2] = R
    return P



class shank(object):
    def __init__(self, pos):
        self.pos = pos

class probe_geometry(object):
    """docstring for probe_geometry"""
    def __init__(self, shanks):
        super(probe_geometry, self).__init__()
        self.shanks = shanks


class probe_view(scene.SceneCanvas):
    '''probe view
    '''
    def __init__(self):
        scene.SceneCanvas.__init__(self, keys=None, title='probe view')
        self.unfreeze()
        self.view = self.central_widget.add_view()
        self.view.camera = 'panzoom'
        self.electrode_pads = scene.visuals.Markers(parent=self.view.scene)
        self.electrode_text = scene.visuals.Text(parent=self.view.scene)
        # self.electrode_edge = scene.visuals.Line(antialias=False, method='gl', color=(1, 1, 1, 0.2), parent=self.view.scene)
        self.electrode_poly = [] 

        # Picker
        self._picker = Picker(self.scene, self.electrode_pads.node_transform(self.view))
        self.key_option = '2' 
        

    def set_data(self, prb, font_size=17):
        '''
        both mapping and grp_dict are dictionary
        mapping is electrode_id to position
        grp_dict is group_id to electrode_id
        mapping = {0: np.array([0., 0.])...}
        grp_dict = {0: np.array([0,1,2,3])}
        '''
        self.prb = prb
        mapping = prb.mapping

        self.electrode_id = np.array(mapping.keys())
        self.electrode_pos = np.vstack(mapping.values())
        self.electrode_pos_KD = KDTree(self.electrode_pos, leaf_size=30, metric='euclidean')
        self.electrode_pads.set_data(self.electrode_pos, symbol='square', size=font_size)
        self.electrode_text.text = [str(i) for i in self.electrode_id]
        self.electrode_text.pos  = self.electrode_pos
        self.electrode_text.font_size = font_size - 15

        if hasattr(prb, 'grp_dict'):
            self.edges, self.grp_idx = self.grp_2_edges(prb.grp_dict)
            # print edges
            self.edeges_color = np.ones((self.electrode_pos.shape[0], 4))*0.5
            # color[grp_idx[1],:] = np.array([1,0,0,0.5])
            self.electrode_edge = scene.visuals.Line(pos=self.electrode_pos, connect=self.edges, antialias=False, method='gl',
                                                     color=self.edeges_color, parent=self.view.scene)

        self.view.camera.set_range([-150,150])


    def grp_2_edges(self, grp_dict):
        _grp_idx = []
        for grp_id, grp in grp_dict.items():
            _grp = np.array([np.where(self.electrode_id==i)[0] for i in grp]).squeeze()
            _grp_idx.append(_grp)
            # print _grp
            if grp_id == 0:
                edges = np.array([i for i in comb(_grp, 2)])
            else:
                edges = np.vstack((edges, np.array([i for i in comb(_grp, 2)])))
        # print edges
        return edges, _grp_idx


    def imap_2_group_id(self, mouse_pos):
        tr = self.view.scene.transform
        pos = tr.imap(mouse_pos)[:2]
        pos = pos.reshape(1, -1)
        _, nn = self.electrode_pos_KD.query(pos, k=1)
        nn_electrode = self.electrode_id[nn[0]][0]
        #TODO: from nn_electrode to group number (_ch2grp) which should be the function of prb
        return nn_electrode


    def imap(self, mouse_pos):
        tr = self.view.scene.transform
        Point = tr.imap(mouse_pos)[:2]
        return Point


    def select(self, group_id):
        if hasattr(self.prb, 'grp_dict'):
            self.electrode_edge.color[:] = np.ones((self.electrode_pos.shape[0], 4))*0.5
            self.electrode_edge.color[self.grp_idx[group_id],:] = np.array([1,0,0,0.5])
            self.update()


    def on_key_press(self, e):
        if e.text == 'r':
            self.view.camera.set_range([-150,150])


    def on_mouse_press(self, e):
        if keys.CONTROL in e.modifiers:
            if self.key_option in ['1','2']:
                self._picker.origin_point(e.pos)


    def on_mouse_move(self, e):
        if keys.CONTROL in e.modifiers and e.is_dragging:
            if self.key_option == '1':
                self._picker.cast_net(e.pos,ptype='rectangle')
            if self.key_option == '2':
                self._picker.cast_net(e.pos,ptype='lasso')


    def on_mouse_release(self,e):
        if keys.CONTROL in e.modifiers and e.is_dragging:
            if self.key_option in ['1','2']:
                mask = self.electrode_id[self._picker.pick(self.electrode_pos)]
                print mask
                # self.highlight(mask)

        else:
            if e.button == 1:
                nn_electrode = self.imap_2_group_id(e.pos)
                try:
                    self.selected_group = self.prb.ch2g[nn_electrode]
                    self.select(self.selected_group)
                    print('{}:{} selected'.format(self.selected_group, self.prb[self.selected_group]))
                except:
                    pass


    def run(self):
        self.show()
        self.app.run()
