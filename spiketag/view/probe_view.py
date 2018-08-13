import numpy as np
from vispy import app, scene, visuals
from vispy.util import keys
from vispy.color import Color
from collections import OrderedDict
from sklearn.neighbors import KDTree
from itertools import combinations as comb
from ..utils import Picker, key_buffer
from matplotlib import path


# white = Color("#ecf0f1")
# gray = Color("#121212")
# red = Color("#e74c3c")
# blue = Color("#2980b9")
# orange = Color("#e88834")


# def star(inner=0.5, outer=1.0, n=5):
#     R = np.array([inner, outer] * n)
#     T = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
#     P = np.zeros((2 * n, 3))
#     P[:, 0] = R * np.cos(T)
#     P[:, 1] = R * np.sin(T)
#     return P


# def rec(left=-15, right=15, bottom=-25, top=25):
#     P = np.zeros((4, 3))
#     R = np.array([[left,  bottom],
#                   [right, bottom],
#                   [right, top   ],
#                   [left,  top   ]])
#     P[:, :2] = R
#     return P



# class shank(object):
#     def __init__(self, pos):
#         self.pos = pos

# class probe_geometry(object):
#     """docstring for probe_geometry"""
#     def __init__(self, shanks):
#         super(probe_geometry, self).__init__()
#         self.shanks = shanks


SCV_Color = np.repeat(np.array([1.,1.,1.]).reshape(1,-1), 800, axis=0)
# SCV_Color[:,1] = np.linspace(1, 0, 800)
# SCV_Color[:,2] = np.linspace(1, 0, 800)
# sns.palplot(SCV_Color[::10])


# def from_scv_to_color(scv):


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

        # keybuf
        self.key_buf = key_buffer()
        

    def set_data(self, prb, font_size=30):
        '''
        both mapping and grp_dict are dictionary
        mapping is electrode_id to position
        grp_dict is group_id to electrode_id
        mapping = {0: np.array([0., 0.])...}
        grp_dict = {0: np.array([0,1,2,3])}
        '''
        self.prb = prb
        self.n_grp = len(prb.grp_dict.keys())
        mapping = prb.mapping
        pos = np.vstack((prb.mapping.values()))
        self.xmin = pos[:,0].min() - 50
        self.xmax = pos[:,0].max() + 50
        self.ymin = pos[:,1].min() - 25
        self.ymax = pos[:,1].max() + 25
        self.font_size = font_size


        self.electrode_id = np.array(mapping.keys())
        self.electrode_pos = np.vstack(mapping.values())
        self.electrode_pos_KD = KDTree(self.electrode_pos, leaf_size=30, metric='euclidean')
        self.electrode_pads_color = np.repeat(np.array([1., 1., 1., 1.]).reshape(1,-1), self.electrode_pos.shape[0], axis=0)
        self.electrode_pads.set_data(self.electrode_pos, symbol='square', face_color=self.electrode_pads_color, size=self.font_size)
        self.electrode_text.text = [str(i) for i in self.electrode_id]
        self.electrode_text.pos  = self.electrode_pos
        self.electrode_text.font_size = int(self.font_size * 0.40)


        if hasattr(prb, 'grp_dict'):
            self.edges, self.grp_idx = self.grp_2_edges(prb.grp_dict)
            # print edges
            self.edeges_color = np.ones((self.electrode_pos.shape[0], 4))*0.5
            # color[grp_idx[1],:] = np.array([1,0,0,0.5])
            self.electrode_edge = scene.visuals.Line(pos=self.electrode_pos, connect=self.edges, antialias=False, method='gl',
                                                     color=self.edeges_color, parent=self.view.scene)

        self.view.camera.set_range([self.xmin, self.xmax])


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

    def get_range_of_electrodes(self, selected_group_id):
        # get the selected electrodes pos from group_id
        group_id = selected_group_id
        selected_electrodes_pos = self.electrode_pos[self.grp_idx[group_id]]
        # get the range of those electrodes
        xmin, xmax = selected_electrodes_pos[:,0].min(), selected_electrodes_pos[:,0].max()
        ymin, ymax = selected_electrodes_pos[:,1].min(), selected_electrodes_pos[:,1].max()
        return np.array([xmin, xmax, ymin, ymax])


    def imap_2_group_id(self, mouse_pos):
        # get mouse position
        tr = self.view.scene.transform
        pos = tr.imap(mouse_pos)[:2]
        pos = pos.reshape(1, -1)
        # get the 1nn electrode number
        _, nn = self.electrode_pos_KD.query(pos, k=1)
        nn_electrode = self.electrode_id[nn[0]][0]
        # get the group_id of that electrode using prb.ch2g (channel_id to group_id mapping)
        try:
            self.selected_group = self.prb.ch2g[nn_electrode]
            # get the range of those electrodes in that group_id
            electrodes_range = self.get_range_of_electrodes(self.selected_group) 
            # print electrodes_range, pos
            # p = path.Path(self.electrode_pos[self.grp_idx[self.selected_group]]) 
            # if p.contains_points(pos):
            #     return self.selected_group
            if pos[0][0] < electrodes_range[1] and pos[0][0] > electrodes_range[0] and pos[0][1] < electrodes_range[3] and pos[0][1] > electrodes_range[2]:
                return self.selected_group
            else:
                return None
        except:
            return None


    def imap(self, mouse_pos):
        tr = self.view.scene.transform
        Point = tr.imap(mouse_pos)[:2]
        return Point


    def select(self, group_id):
        if group_id is not None:
            if hasattr(self.prb, 'grp_dict'):
                self.electrode_edge.color[:] = np.ones((self.electrode_pos.shape[0], 4))*0.5
                self.electrode_edge.color[self.grp_idx[group_id],:] = np.array([0,1,1,0.8])
                self.update()
                self.prb.emit('select', group_id=group_id, chs=self.prb[group_id])


    def set_scv(self, scv, upper_bound_sc=1000):
        '''
        spike count vector (scv): every group has spike counts over a time bin (30 or 100 ms etc)
        function: modulates color in electrode_pads
        '''
        self.scv = np.clip(scv, 0, upper_bound_sc)
        if scv.shape[0] != self.n_grp:
            print('spike count vector length mismatch: there are {} electrodes to update spike count'.format(self.electrode_pads_color.shape[0]))
        # the nearer scv to upper_bound_sc, the redder it would be
        else:
            red_score = (upper_bound_sc-scv)/upper_bound_sc
            red_score = np.clip(red_score, 0, 1)
            for group_id in range(self.n_grp):
                self.electrode_pads_color[self.grp_idx[group_id]] = np.repeat(np.array([1, red_score[group_id], red_score[group_id], 1]).reshape(1,-1) ,4, axis=0)
            self.electrode_pads.set_data(self.electrode_pos, symbol='square', face_color=self.electrode_pads_color, size=self.font_size)


    def on_key_press(self, e):
        if e.text == 'r':
            self.view.camera.set_range([self.xmin, self.xmax])
        if e.text == 'g':
            try:
                self.selected_group = int(self.key_buf.pop())
                if self.selected_group in self.prb.grp_dict.keys():
                    self.select(self.selected_group)
            except:
                self.key_buf.pop()
                pass
        elif e.text.isdigit():
            self.key_buf.push(e.text)


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
                self.selected_group = self.imap_2_group_id(e.pos)
                self.select(self.selected_group)


    def run(self):
        self.show()
        self.app.run()
