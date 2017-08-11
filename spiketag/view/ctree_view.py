import numpy as np
from vispy import app, gloo, visuals, scene
from ..utils.utils import Timer
from ..utils import conf
from ..utils.conf import debug
from .color_scheme import palette
from matplotlib import path

class ctree_view(scene.SceneCanvas):
    
    def __init__(self, show=False):
        scene.SceneCanvas.__init__(self, keys=None)

        self.unfreeze()
        self._view = self.central_widget.add_view()
        self._view.camera = 'panzoom'
        # transform will change dynamiclly by view changed
        self._transform = self._view.camera.transform

        self._mesh = scene.visuals.Mesh()
        self._mesh.mode = 'triangles' 
	self._line = scene.visuals.Line(connect='segments', method='gl')
        self._view.add(self._mesh) 
	self._view.add(self._line)

        self._clu_x_coords = []
        self._clu_y_ccords = []
        self._vertices = []
        self._face = []
        self._face_colors = []
        self._line_pos = []
        self._bar_coords = []
        self._bar_index = []
        self._cluster_bounds = {}

        self._last_highlight = None
       
        if show is True:
            self.show()

    def set_data(self, clu):
	assert clu._extra_info is not None	
        
        self._clu = clu
        self._whole_tree = clu._extra_info['condensed_tree']
        self._clu_tree = self._whole_tree[self._whole_tree['child_size'] > 1]
        #  self._select_clusters = np.array(clu._extra_info['default_select_clusters'], dtype=np.int64)
        self._select_clusters = clu.select_clusters
        self._build_data(self._whole_tree)
	self._render()
        self._set_range()		

    def on_key_press(self, e):
        if e.text == 'r':
            self._view.camera.reset()
            self._view.camera.set_range()

    def on_mouse_move(self, e):
        with Timer("[View] Ctreeview -- get_cluster_by_pos.", conf.ENABLE_PROFILER):
            current_cluster = self._get_cluster_by_pos(e.pos)
        if not current_cluster:
            self._last_highlight = None
        elif current_cluster == self._last_highlight:
            return
        else:
            with Timer("[View] Ctreeview -- select cluster {}.".format(current_cluster), conf.ENABLE_PROFILER):
                self._select(current_cluster)
            self._last_highlight = current_cluster
            return
    
    def on_mouse_double_click(self, e):
        if e.button == 1:
            current_cluster = self._get_cluster_by_pos(e.pos)
            if current_cluster:
                # expand cluster
                if current_cluster in self._select_clusters:
                    with Timer("[VIEW] Ctreeview -- expand cluster {}.".format(current_cluster), conf.ENABLE_PROFILER):
                        debug('ctree view expand clustier {} here'.format(current_cluster))
                        children = self._get_children(self._clu_tree, current_cluster)
                        if len(children) == 0:
                            return 

                        parent = self._get_parent(self._clu_tree, current_cluster)
                        if parent in self._select_clusters:
                            return

                        # delete self and add children to select cluster 
                        sc = self._select_clusters
                        sc = np.append(np.delete(sc, np.where(sc == current_cluster)[0]), children)
                        self._select_clusters = np.sort(sc)
                        self._clu.select_clusters = self._select_clusters
                        group_of_idx = []
                        group_of_clu = []
                        if (self._select_clusters[-2:] == children).all():
                            group_of_idx.append(self._get_leaves_from_whole_tree(current_cluster))                        
                            group_of_clu.append(np.array([0]))
                            for child in children:
                                global_idx = self._get_leaves_from_whole_tree(child)
                                clu = np.where(self._select_clusters == child)[0] + self._clu.max_clu_id  #plus one since always ignore the grey one
                                group_of_idx.append(global_idx)
                                group_of_clu.append(clu)

                            self._clu.fill(group_of_idx, group_of_clu)
                        else:
                            debug('Not support expand here now!')
                    
                        self._update()

                # collapse cluster
                else:
                    debug('ctree view collapse cluster {} here'.format(current_cluster))
                    children = self._get_children(self._clu_tree, current_cluster)
                    for child in children:
                        if child not in self._select_clusters:
                            return 
                    sc = self._select_clusters
                    group_of_idx = [].append(np.where(sc == children[0])[0])
                    group_of_clu = [np.array([], dtype=np.int64)]
                    for child in children:
                        sc = np.delete(sc, np.where(sc == child)[0])
                    sc = np.append(sc, current_cluster)
                    self._select_clusters = np.sort(sc)
                    self._clu.select_clusters = self._select_clusters   
                    for child in children:
                        group_of_clu[0] = np.append(group_of_clu[0], self._get_leaves_from_whole_tree(child))
                    self._clu.fill(group_of_idx, group_of_clu) 

                    

    def _get_cluster_by_pos(self, pos):
        for cluster, vertices in self._cluster_bounds.iteritems():
            p = path.Path(self._transform.map(vertices)[:,:2], closed=True)
            selected = p.contains_point(pos)
            if selected:
                return cluster
        return None


    def _build_data(self, tree):
        leaves = self._get_leaves_from_clu_tree()
	last_leaf, root = tree['parent'].max(), tree['parent'].min()

	self._clu_x_coords, self._clu_y_coords = self._get_x_y_coords(tree, root, last_leaf, leaves)
        
        self._vertices, self._faces, self._face_colors= self._get_mesh_data(tree, root, last_leaf, self._clu_x_coords, self._clu_y_coords)
        self._line_pos = self._get_line_coords(tree, root, self._clu_x_coords, self._clu_y_coords)

        self._invert_y(self._vertices)
        self._invert_y(self._line_pos)

    def _get_mesh_data(self, tree, root, last_leaf, clu_x_coords, clu_y_coords):
	self._bar_coords, self._bar_index, self._cluster_bounds = self._get_bar_coords(tree, root, last_leaf, clu_x_coords, clu_y_coords)
	
	vertices = self._bar_coords.ravel().reshape(-1, 2)
	
	#scaling
        #  vertices = np.column_stack((vertices[:,0] / vertices[:,0].max(), vertices[:,1] / vertices[:,1].max()))

	face_base = np.array([0, 1, 2, 0, 3, 2])    
        faces = np.tile(face_base, vertices.shape[0]/4)
        faces = faces + np.repeat(np.arange(vertices.shape[0]/4) * 4, 6)
	faces = faces.reshape(-1, 3)

        face_colors = np.full([faces.shape[0], 4], (1, 1, 0, 1), dtype=np.float64)
        for idx, val in enumerate(self._select_clusters):
            color = np.hstack((np.asarray(palette[idx+1]), 1))
            for child in self._get_descendants(self._clu_tree, val):
                face_idx = np.hstack((self._bar_index[child] * 2, self._bar_index[child] * 2 + 1))
                face_colors[face_idx] = color
	return vertices, faces, face_colors 

    def _select(self, cluster_id):
        with Timer("[View] Ctreeview -- get_leaves.", conf.ENABLE_PROFILER):
            global_idx = self._get_leaves_from_whole_tree(cluster_id)
        with Timer("[View] Ctreeview -- clu.select", conf.ENABLE_PROFILER):
            self._clu.select(global_idx)

    def _get_leaves_from_clu_tree(self, root=None):
        '''because the reason of performance, so seperate the method of get leaves from clu_tree or whole tree.
        '''
        if not root:
            root = self._clu_tree['parent'].min()

        nodes = []
        def __get_leaves(tree, root):
            children = tree[tree['parent'] == root]['child']
            if len(children) == 0:
                nodes.append(root)
            else:
                for child in children: __get_leaves(tree, child)
        __get_leaves(self._clu_tree, root)
        return np.array(nodes)

    def _get_leaves_from_whole_tree(self, root=None):
        '''because the reason of performance, so seperate the method of get leaves from clu_tree or whole tree.
        '''

        if not root:
            root = self._clu_tree['parent'].min()

        sub_tree = self._get_descendants(self._clu_tree, root)
        leaves = np.array([], dtype=np.int64)
        
        for node in sub_tree:
            children = self._whole_tree[self._whole_tree['parent'] == node]
            leaves = np.append(leaves, children[children['child_size'] == 1]['child'])
        return leaves

    def _get_descendants(self, tree, root):
        '''get all descendants of tree and given root node
        '''
        nodes = []
        def __get_descendants(root):
            nodes.append(root)
            for child in tree[tree['parent'] == root]['child']:
                __get_descendants(child)
        __get_descendants(root)

        return np.array(nodes)

    def _get_parent(self, tree, child):
        '''get parent of node.
        '''
        return tree[tree['child'] == child]['parent']

    def _get_children(self, tree, parent):
        '''get all children of node.
        '''
        return tree[tree['parent'] == parent]['child']


    def _get_x_y_coords(self, tree, root, last_leaf, leaves):
 	clu_x_coords = dict(zip(leaves, [x for x in range(len(leaves))]))
	clu_y_coords = {root: 0.0}

	for clu in range(last_leaf, root - 1, -1):
	    split = tree[['child', 'lambda_val']]
	    split = split[(tree['parent'] == clu) &
			  (tree['child_size'] > 1)]
	    if len(split['child']) > 1:
		left_child, right_child = split['child'] # here means only allowed two children
		clu_x_coords[clu] = np.mean([clu_x_coords[left_child], clu_x_coords[right_child]])
		clu_y_coords[left_child] = split['lambda_val'][0]
		clu_y_coords[right_child] = split['lambda_val'][1]

	return clu_x_coords, clu_y_coords

    def _get_bar_coords(self, tree, root, last_leaf, clu_x_coords, clu_y_coords):
	bar_centers = [] 
	bar_tops = []
	bar_bottoms = []
	bar_widths = []	
        bar_index = {}

	scaling = np.sum(tree[tree['parent'] == root]['child_size'])
	scaling = np.log(scaling)

        CURSOR = 0

        cluster_bounds = {}

	for c in range(last_leaf, root - 1, -1):
	    c_children = tree[tree['parent'] == c]
	    current_size = np.sum(c_children['child_size'])
	    current_lambda = clu_y_coords[c]
	    
	    current_size = np.log(current_size)
            idx = []
                         
            cb_vertices = self._get_rectangle_vertices(clu_x_coords[c] * scaling - (current_size / 2.0),
                                                  clu_x_coords[c] * scaling + (current_size / 2.0),
                                                  np.max(c_children['lambda_val']),
                                                  clu_y_coords[c])
            self._invert_y(cb_vertices)
            cluster_bounds[c] = cb_vertices 
            
	    for i in np.argsort(c_children['lambda_val']):
		row = c_children[i]
		if row['lambda_val'] != current_lambda:
                    bar_centers.append(clu_x_coords[c] * scaling)
                    bar_tops.append(row['lambda_val'] - current_lambda)
                    bar_bottoms.append(current_lambda)
                    bar_widths.append(current_size)

                    #  idx  = np.append(idx, CURSOR)
                    idx.append(CURSOR)
                    CURSOR += 1
                    
                exp_size = np.exp(current_size) - row['child_size']
		if exp_size > 0.01:
                    current_size = np.log(np.exp(current_size) - row['child_size'])
		else:
		    current_size = 0.0
		current_lambda = row['lambda_val']

            bar_index[c] = np.array(idx)

        bar_centers = np.asarray(bar_centers)
        bar_widths = np.asarray(bar_widths)
        bar_bottoms = np.asarray(bar_bottoms)
        bar_tops = np.asarray(bar_tops)

	up_left = np.column_stack((bar_centers - bar_widths / 2, bar_bottoms))
	bottom_left = np.column_stack((bar_centers - bar_widths / 2, bar_bottoms + bar_tops))
	up_right = np.column_stack((bar_centers + bar_widths / 2, bar_bottoms))
	bottom_right = np.column_stack((bar_centers + bar_widths / 2, bar_bottoms + bar_tops))

	coords = np.column_stack((up_left, up_right, bottom_right, bottom_left)).reshape(-1,4,2)
	
	return coords, bar_index, cluster_bounds

    def _get_line_coords(self, tree, root, clu_x_coords, clu_y_coords):

	line_pos = np.array([])
	scaling = np.sum(tree[tree['parent'] == root]['child_size'])
	scaling = np.log(scaling)

	for row in tree[tree['child_size'] > 1]: 
            parent = row['parent']
	    child = row['child']
	    child_size = row['child_size']

	    child_size = np.log(child_size)
	    sign = np.sign(clu_x_coords[child] - clu_x_coords[parent])

	    line_pos = np.append(line_pos, [clu_x_coords[parent] * scaling, clu_y_coords[child]])
	    line_pos = np.append(line_pos, [clu_x_coords[child] * scaling + sign * (child_size / 2.0), clu_y_coords[child]])

        return line_pos.reshape(-1, 2)

    def _invert_y(self, pos):
        assert pos.shape[1] == 2
        pos[:,1] = (-1) * pos[:,1]

    def _set_range(self):
        x_bound = (self._vertices[:,0].min(), self._vertices[:,0].max())
        y_bound = (self._vertices[:,1].min(), self._vertices[:,1].max())
        self._view.camera.set_range(x=x_bound, y=y_bound)
   
    def _render(self):
        self._mesh.set_data(vertices=self._vertices, faces=self._faces, face_colors=self._face_colors);
        self._line.set_data(pos=self._line_pos, color=(1,1,0,1))

    def _update(self):
        self._build_data(self._whole_tree)
        self._render()
        self._set_range()		

    def _get_rectangle_vertices(self, left, right, top, bottom):
        center = (np.mean([left, right]), np.mean([bottom, top]))
        width = abs(right - left)
        height = abs(top - bottom)
        up_left = (center[0] - width/2, center[1] + height/2)
        bottom_left = (center[0] - width/2, center[1] - height/2)
        up_right = (center[0] + width/2, center[1] + height/2)
        bottom_right = (center[0] + width/2, center[1] - height/2)
        return np.array([up_left, bottom_left, bottom_right, up_right, up_left])

