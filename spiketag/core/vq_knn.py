import torch
import numpy as np


def get_n_vq(y, N=500, max_noise_N=200, method='proportional'):
    '''
    make sure the label y vector has more than 1 labels
    '''
    nclu = y.long().bincount().shape[0]
    if method == 'proportional':
        dist = y.long().bincount()/float(y.shape[0])
    elif method == 'equal':
        dist = torch.ones((nclu,)) / nclu
    n_vq = torch.round(dist*(N)).long()
    nvq_from_noise = (n_vq[0] - max_noise_N)/(nclu-1)
    n_vq[1:] += nvq_from_noise
    n_vq[0]  = max_noise_N
    err = n_vq.sum() - N
    n_vq[0] -= err
    assert(n_vq.sum()==N)
    return n_vq
  

def VQ(X, y, N=500, max_noise_N=200, method='proportional', n_vq=None):
    n_vq = get_n_vq(y, N, max_noise_N, method)
    vq = []
    nclu = y.long().bincount().shape[0]
    from sklearn.cluster import MiniBatchKMeans
    for _clu_id in range(nclu):
        km = MiniBatchKMeans(n_vq[_clu_id])
        km.fit(X[y==_clu_id].numpy())
        vq.append(km.cluster_centers_)
    vq = np.vstack(vq)
    vq_labels = torch.repeat_interleave(torch.arange(nclu), n_vq)
    return vq, vq_labels
  

class VQ_KNN(object):
    '''
    model = VQ_KNN(X,y,N=6,k=1)
    model.predict(X_test)
    '''
    def __init__(self, X, y, N=500, k=1, method='proportional', max_noise_N=200, n_vq=None):
        if type(X) == np.ndarray:
            X = torch.from_numpy(X)
        if type(y) == np.ndarray:
            y = torch.from_numpy(y)
        self.vq, self.vq_label = VQ(X=X, y=y, N=N, max_noise_N=max_noise_N, method=method)
        from sklearn.neighbors import KNeighborsClassifier as KNN
        self.knn = KNN(n_neighbors=k)
        self.knn.fit(self.vq, self.vq_label)
    
    def predict(self, X_test):
        return self.knn.predict(X_test)
