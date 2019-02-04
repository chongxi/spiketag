from spiketag.base import CLU, status_manager
from spiketag.view import cluster_view
import numpy as np


# Setup clus and their manager
clu_manager = status_manager()

for i in range(40):
    clu = CLU(np.arange(i+1))
    clu._id = i
    clu_manager.append(clu)

@clu_manager.connect
def on_update(**kwargs):
    print(kwargs)

print(clu_manager)



# Setup the view that the manager controls
cluview = cluster_view()
cluview.set_data(clu_manager, 0)


if __name__ == '__main__':

    # 1. First way to update
    clu_manager[1].emit('report', state='BUSY')


    # 2. Second way to update
    # clu_manager[3] = 'BUSY'
    # clu_manager[8] = 'READY'
    # clu_manager.emit('update', state=clu_manager.state_list, nclu=clu_manager.nclu_list)

    cluview.run()



