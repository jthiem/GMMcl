# -*- coding: utf-8 -*-
"""GMM based clssifier"""
import numpy as np
import sklearn.base
from sklearn.mixture import GMM

# A classifier where each class is modeled by a GMM
class GMMcl(sklearn.base.ClassifierMixin):
    '''The GMMcl is a sckits-learn style classifier where each class
    is represented by a GMM.  It is in effect an array of GMMs.

    Parameters
    ----------
    model : GMM-like class
        Default is sklearn.mixture.GMM (default), but other possibilities
        are sklearn.mixture.DPGMM or sklearn.mixture.VBGMM.

    minll : float
        Minimum value log-likelihood can be.  Allows for labels that are
        never seen in the training data.

    kwargs : additional keyword arguments
        Parameters to be passed through to `model`
    '''
    nClasses = None
    mixtures = None
    model = None
    minll = None

    def __init__(self, model=GMM, minll=-100, **kwargs):
        self.model = model
        self.minll = minll
        self.kwargs = kwargs

    def fit(self, X, y):
        self.nClasses = int(np.max(y)+1)
        self.mixtures = [self.model(**self.kwargs) for k in range(self.nClasses)]
        for k, mix in enumerate(self.mixtures):
            if any(y==k):
                mix.fit(X[y==k])
        return self

    def _scoremix(self, m, X):
        minvals = self.minll*np.ones(X.shape[0])
        # check the mixture was trained
        if 'means_' in m.__dict__:
            return np.fmax(m.score(X), minvals)
        else:
            return minvals[:]

    def predict(self, X):
        ll = [ self._scoremix(mix, X) for mix in self.mixtures ]
        return np.vstack(ll).argmax(axis=0)

    def loglike_per_class(self, X, y=None):
        ll = [ self._scoremix(mix, X) for mix in self.mixtures ]
        return np.vstack(ll)
