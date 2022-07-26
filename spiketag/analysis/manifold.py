import sklearn
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import FactorAnalysis

def spike_noise_bernoulli(X, noise_level=1, p=0.5, gain=1, cuda=True, IID=True):
    '''
    Add IID noise to data (spike vector count) to train network to ignore off-manifold activity

    each neuron will add a noise to each time bin: 
    noise = uniform(-noise_level, noise_level) * bernoulli(p)
    '''
    if cuda:
        noise = torch.ones_like(X).uniform_(-noise_level, noise_level)*torch.bernoulli(torch.ones_like(X)*p).cuda()
        X = X.cuda()
    else:
        noise = torch.ones_like(X).uniform_(-noise_level, noise_level)*torch.bernoulli(torch.ones_like(X)*p)
    if IID:
        X = torch.relu(gain*(X + noise))
    else:
        X = torch.relu(gain*(X + noise*X.mean(axis=0)))
    return X


def spike_noise_gaussian(X, noise_level=1, mean=0.0, std=3.0, gain=1, cuda=True, IID=True):
    '''
    Add IID noise to data (spike vector count) to train network to ignore off-manifold activity

    each neuron will add a noise to each time bin: 
    noise = uniform(-noise_level, noise_level) * bernoulli(p)
    '''
    if cuda:
        noise = torch.ones_like(X).uniform_(-noise_level, noise_level) * \
            torch.normal(torch.ones_like(X)*mean, std).cuda()
        X = X.cuda()
    else:
        noise = torch.ones_like(
            X).uniform_(-noise_level, noise_level)*torch.normal(torch.ones_like(X)*mean, std)
    if IID:
        X = torch.relu(gain*(X + noise))
    else:
        X = torch.relu(gain*(X + noise*X.mean(axis=0)))
    return X

class FA(FactorAnalysis):
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
