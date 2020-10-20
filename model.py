import pymc3 as pm
import pymc3.distributions.transforms as tr
from pymc3 import Model, Normal, Slice
from pymc3 import sample
from pymc3 import traceplot
from pymc3.distributions import Interpolated
import matplotlib.pyplot as plt
import matplotlib as mpl
from theano import as_op
import theano.tensor as tt
import numpy as np
from scipy import stats

def from_posterior(param, samples):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return Interpolated(param, x, y)

"""
def create_model(batsmen, bowlers, id1, id2, X):
    testval = [[-5,0,1,2,3.5,5] for i in range(0, 9)]
    l = [i for i in range(9)]
    model = pm.Model()
    with model:
        delta_1 = pm.Uniform("delta_1", lower=0, upper=1)
        delta_2 = pm.Uniform("delta_2", lower=0, upper=1)
        inv_sigma_sqr = pm.Gamma("sigma^-2", alpha=1.0, beta=1.0)
        inv_tau_sqr = pm.Gamma("tau^-2", alpha=1.0, beta=1.0)
        mu_1 = pm.Normal("mu_1", mu=0, sigma=1/pm.math.sqrt(inv_tau_sqr), shape=len(batsmen))
        mu_2 = pm.Normal("mu_2", mu=0, sigma=1/pm.math.sqrt(inv_tau_sqr), shape=len(bowlers))
        delta = pm.math.ge(l, 3) * delta_1 + pm.math.ge(l, 6) * delta_2
        eta = [pm.Deterministic("eta_" + str(i), delta[i] + mu_1[id1[i]] - mu_2[id2[i]]) for i in range(9)]
        cutpoints = pm.Normal("cutpoints", mu=0, sigma=1/pm.math.sqrt(inv_sigma_sqr), transform=pm.distributions.transforms.ordered, shape=(9,6), testval=testval)
        X_ = [pm.OrderedLogistic("X_" + str(i), cutpoints=cutpoints[i], eta=eta[i], observed=X[i]-1) for i in range(9)]
    return model
"""

def create_model(batsmen, bowlers, id1, id2, X):
    testval = [[-5,0,1,2,3.5,5] for i in range(0, 9)]
    l = [i for i in range(9)]
    model = pm.Model()
    with model:
        delta_1 = pm.Uniform("delta_1", lower=0, upper=1)
        delta_2 = pm.Uniform("delta_2", lower=0, upper=1)
        inv_sigma_sqr = pm.Gamma("sigma^-2", alpha=1.0, beta=1.0)
        inv_tau_sqr = pm.Gamma("tau^-2", alpha=1.0, beta=1.0)
        mu_1 = pm.Normal("mu_1", mu=0, sigma=1/pm.math.sqrt(inv_tau_sqr), shape=len(batsmen))
        mu_2 = pm.Normal("mu_2", mu=0, sigma=1/pm.math.sqrt(inv_tau_sqr), shape=len(bowlers))
        delta = pm.math.ge(l, 3) * delta_1 + pm.math.ge(l, 6) * delta_2
        eta = [pm.Deterministic("eta_" + str(i), delta[i] + mu_1[id1[i]] - mu_2[id2[i]]) for i in range(9)]
        cutpoints = pm.Normal("cutpoints", mu=0, sigma=1/pm.math.sqrt(inv_sigma_sqr), transform=pm.distributions.transforms.ordered, shape=(9,6), testval=testval)
        X_ = [pm.OrderedLogistic("X_" + str(i), cutpoints=cutpoints[i], eta=eta[i], observed=X[i]-1) for i in range(9)]
    return model


def update_model(batsmen, bowlers, id1, id2, X, trace):
    testval = [[-5,0,1,2,3.5,5] for i in range(0, 9)]
    l = [i for i in range(9)]
    model = pm.Model()
    with model:
        delta_1 = from_posterior("delta_1", trace["delta_1"])
        delta_2 = from_posterior("delta_2", trace["delta_2"])
#         inv_sigma_sqr = from_posterior("sigma^-2", trace["sigma^-2"])
#         inv_tau_sqr = from_posterior("tau^-2", trace["tau^-2"])
        mu_1 = from_posterior("mu_1", trace["mu_1"])
        mu_2 = from_posterior("mu_2", trace["mu_2"])
        delta = pm.math.ge(l, 3) * delta_1 + pm.math.ge(l, 6) * delta_2
        eta = [pm.Deterministic("eta_" + str(i), delta[i] + mu_1[id1[i]] - mu_2[id2[i]]) for i in range(9)]
        cutpoints = from_posterior("cutpoints", trace["cutpoints"])
        X_ = [pm.OrderedLogistic("X_" + str(i), cutpoints=cutpoints[i], eta=eta[i], observed=X[i]-1) for i in range(9)]
        """
        delta_2 = pm.Uniform("delta_2", lower=0, upper=1)
        inv_sigma_sqr = pm.Gamma("sigma^-2", alpha=1.0, beta=1.0)
        inv_tau_sqr = pm.Gamma("tau^-2", alpha=1.0, beta=1.0)
        mu_1 = pm.Normal("mu_1", mu=0, sigma=1/pm.math.sqrt(inv_tau_sqr), shape=len(batsmen))
        mu_2 = pm.Normal("mu_2", mu=0, sigma=1/pm.math.sqrt(inv_tau_sqr), shape=len(bowlers))
        delta = pm.math.ge(l, 3) * delta_1 + pm.math.ge(l, 6) * delta_2
        eta = [pm.Deterministic("eta_" + str(i), delta[i] + mu_1[id1[i]] - mu_2[id2[i]]) for i in range(9)]
        cutpoints = pm.Normal("cutpoints", mu=0, sigma=1/pm.math.sqrt(inv_sigma_sqr), transform=pm.distributions.transforms.ordered, shape=(9,6), testval=testval)
        X_ = [pm.OrderedLogistic("X_" + str(i), cutpoints=cutpoints[i], eta=eta[i], observed=X[i]-1) for i in range(9)]
        """
    return model

