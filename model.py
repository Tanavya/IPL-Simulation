import pymc3 as pm
import pymc3.distributions.transforms as tr

def create_model(batsmen, bowlers, id1, id2, X):
    testval = [[-5 + x * (2 * 5)/5.0 for x in range(6)] for i in range(0, 9)]
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