import ipyparallel as ipp


class bg_cluster():
    def __init__(self):
        self.client = ipp.Client()
        self.cpu = self.client.load_balanced_view()
    
    def fit(self, fet, clu, n_comp=8, max_iter=300):
        self.fet = fet
        self.clu = clu
        ar = self.cpu.apply_async(self.cluster, fet=fet, n_comp=n_comp, max_iter=max_iter)
        def get_result(ar):
            self.clu.fill(ar.get())  # clu is a Troy, by `fill`
        ar.add_done_callback(get_result)
    
    @staticmethod
    def cluster(fet, n_comp, max_iter):
        from sklearn.mixture import BayesianGaussianMixture as DPGMM
        dpgmm = DPGMM(
            n_components=n_comp, covariance_type='full', weight_concentration_prior=1e-3,
            weight_concentration_prior_type='dirichlet_process', init_params="kmeans",
            max_iter=100, random_state=0, verbose=0, verbose_interval=10) # init can be "kmeans" or "random"
        dpgmm.fit(fet)
        label = dpgmm.predict(fet)
        return label

