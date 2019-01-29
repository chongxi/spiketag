from spiketag.view import cluster_view
import numpy as np


if __name__ == '__main__':
    cluview = cluster_view()
    group_No = 20
    sorting_status = np.random.randint(low=0, high=3, size=group_No) # 0: not ready;   1: ready;   2: done
    print(sorting_status)
    cluview.set_data(group_No=group_No, sorting_status=sorting_status, nclu_list=group_No*[8])
    # cluview.select(8)
    print(cluview.cpu_ready_list)

    @cluview.event.connect
    def on_select(group_id):
        print('group {} is selected'.format(group_id))

    cluview.run()
    