import ipyparallel as ipp
import numpy as np
from vispy import app

dpgmm_def = '''
def dpgmm(max_cluster_n=10):
    from sklearn.mixture import BayesianGaussianMixture as DPGMM
    dpgmm = DPGMM(
        n_components=max_cluster_n, covariance_type='full', weight_concentration_prior=1e-3,
        weight_concentration_prior_type='dirichlet_process', init_params="kmeans",
        max_iter=300, random_state=0, verbose=0, verbose_interval=10) # init can be "kmeans" or "random"
    dpgmm.fit(data)
    label = dpgmm.predict(data)
    return label
'''

dpgmm_exec = 'label = dpgmm({})'


class multicore(object):
    """docstring for cpu"""
    def __init__(self, cpu_No):
        super(multicore, self).__init__()
        self.cpu_No = cpu_No
        self.cpu = ipp.Client()
        self.dview = self.cpu[:]
        self.dview.block = False
        self.cpu_ready = {}

    def set_data(self, cpu_id, dict):
        self.cpu[cpu_id].push(dict)

    def _import(self, cpu_id, func_def):
        self.cpu[cpu_id].execute(func_def)

    def _exec(self, cpu_id, func_exe, *args):
        cmd = func_exe.format(args[0])
        self.cpu_ready[cpu_id] = self.cpu[cpu_id].execute(cmd)

    def test_cpu_ready(self, cpu_id):
        return self.cpu_ready[cpu_id].ready()

    @property
    def cpu_ready_list(self):
        try:
            return [self.test_cpu_ready(cpu_id) for cpu_id in range(self.cpu_No)]
        except:
            return [True]*self.cpu_No

    def get(self, cpu_id, var):
        return self.cpu[cpu_id].pull(var).get()

    def __getitem__(self, cpu_id):
        return self.cpu[cpu_id]


class DPGMM_IPY(multicore):
    """docstring for DPGMM_multicore"""
    def __init__(self, cpu_No):
        super(DPGMM_IPY, self).__init__(cpu_No)
        for i in range(cpu_No):
            self._import(i, dpgmm_def)

    def __call__(self, fet):
        for i in range(self.cpu_No):
            self.set_data(i, {'data': fet[i]})
        
    def run(self, group_id):
        self._exec(group_id, dpgmm_exec, 4)

    def run_all(self):
        for i in range(self.cpu_No):
            self._exec(i, dpgmm_exec, 4)

        
def report_cpu_status(ev):
    group_id = dpgmm[0]['group_id']
    print('cpu ready list:', dpgmm.cpu_ready_list)
    if dpgmm.cpu_ready_list[0]:
        dpgmm[0]['data'] = fet[group_id]
        dpgmm[0].execute('label = dpgmm(4)')
        labels = dpgmm[0]['label']
        print(group_id, labels)
    dpgmm[0]['group_id'] += 1
    if dpgmm[0]['group_id'] == group_No:
        timer.stop()
        app.quit()


if __name__ == '__main__':
    # 1. Test base class multicore
    # backend = multicore(cpu_No=8)
    # for i in range(8):
    #     backend.set_data(i, {'data': np.random.randn(10000,4)})
    #     backend._import(i, dpgmm_def)
    #     backend._exec(i, dpgmm_exec, 4)

    # for i in range(8):
    #     clu = backend.get(i, 'label')
    #     ar  = backend.test_cpu_ready(i)
    #     print('cpu'+str(i), clu)
    #     print('cpu'+str(i), ar)


    # 2. Test sequential background execution
    # cpu_No = 1
    # group_No = 8
    # fet = np.random.randn(group_No, 10000, 4)
    # dpgmm = DPGMM_IPY(cpu_No=cpu_No)
    # dpgmm(fet)
    # dpgmm[0]['group_id'] = 0
    # timer = app.Timer(connect=report_cpu_status, interval=0.5)
    # timer.start()
    # app.run()
    

    # 3. Test parallel background execution
    cpu_No = 8
    group_No = 8
    fet = np.random.randn(group_No, 10000, 4)
    dpgmm = DPGMM_IPY(cpu_No=cpu_No)
    dpgmm(fet)
    dpgmm.run_all()
    print('cpu ready list:', dpgmm.cpu_ready_list)
    for i in range(dpgmm.cpu_No):
        print(dpgmm[i]['label'])
    print('cpu ready list:', dpgmm.cpu_ready_list)