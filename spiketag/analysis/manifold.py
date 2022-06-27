import sklearn
import numpy as np
import torch
import torch.nn.functional as F


class FA(sklearn.decomposition.FactorAnalysis):
    '''
    Factor analysis extended from sklearn
    `reconstruct` input to 
        1. remove independent noise (i.e., off-manifold activity) across neurons
        2. keep only the shared variability (on-manifold activity) across neurons
        The result is not integer.
    `sample_manifold` input to 
        acquire resampled on-manifold spike count (integer)

    Parameters
        X: array-like of shape (n_samples, n_features)

    Example
        fa = FA(n_components=30)
        fa.fit(scv) #scv (spike count vector): (n_samples, n_units)
        reconstructed_scv = fa.reconstruct(scv) 
        resampled_scv = fa.sample_manifold(scv) 
    '''
    def reconstruct(self, X):
        factors = self.transform(X)
        reconstructed_X = np.clip(factors@self.components_ + self.mean_, 0, 100)
        return reconstructed_X

    def sample_manifold(self, X, sampling_method=np.random.poisson):
        reconstructed_X = self.reconstruct(X)
        sampled_X = sampling_method(reconstructed_X)
        return sampled_X
