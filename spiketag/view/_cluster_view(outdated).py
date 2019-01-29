import  numpy as np
from phy.gui.widgets import Table

class cluster_view(Table):
    '''
        For now, just show the id of cluster and number of spikes for each cluster
    '''
    def __init__(self):
        super(cluster_view, self).__init__()

    ### ----------------------------------------------
    ###              public method 
    ### ----------------------------------------------

    def set_data(self, clu):
        self._clu = clu
       
        @self.add_column
        def spikes(id):
            '''
                add a column named 'spikes' here
            '''
            return self._clu.index_count[id]

        @self.connect_
        def on_select(ids):
            '''
                listener the element selected event from view, and emit clu select event.  
            '''
            if len(ids) > 0:
                self._clu.select_clu(np.array(ids))

        @self._clu.connect
        def on_cluster(*args, **kwargs):
            self._render()

        # !!Attention!! 
        # set content of column must after add_column and connect_
        self._render()


    ### ----------------------------------------------
    ###              private  method 
    ### ----------------------------------------------

    def _render(self):
        '''
            set the column of ids, the first column of table
        '''
        cluster_ids = self._clu.index_id.tolist()
        self.set_rows(cluster_ids)

