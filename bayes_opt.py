import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class BayesianOptimizer(object):

    def __init__(self, feature_meta,
                 init_observations=[],
                 kernel=Matern(nu=.5)):
        self.feature_names = feature_meta.keys()
        self.feature_bounds = np.stack(feature_meta.values())
        self.features_dim = len(self.feature_names)
        self.observations = init_observations
        self.i = 0
        self.kernel = kernel
        self.model = GaussianProcessRegressor(kernel=self.kernel)
        self.acquisition_params = {
            'type': 'ucb',
            'u': 0.5  # TODO: this is an arbitrary value
        }
        self.acquisition = self._ucb

    def update(self, features, target):
        self.observations.append([features, target])
        X, y = np.stack(self.observations)
        self.model.fit(X, y)
        return self

    def suggest(self):
        samples = np.random.uniform(
            self.feature_bounds[:, 0],
            self.feature_bounds[:, 1],
            size=(100, self.feature_bounds.shape[0]))
        optimum_val = -np.inf
        for sample in samples:
            opt_res = minimize(
                fun=self.acquisition,
                x0=sample,
                bounds=self.feature_bounds)
            if -opt_res.min() >= optimum_val:
                optimum_val = -opt_res.fun.min()
                optimum = opt_res.x

        return optimum

    def step(self, features, target):
        self.update(features, target)
        return self.suggest()

    def _ucb(self, x):
        u = self.acquisition_params['u']
        mean, std = self.model.predict(x, return_std=True)
        return - mean + u * std
