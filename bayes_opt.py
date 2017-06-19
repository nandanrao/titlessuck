import numpy as np
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
        self.kernel = Matern(nu=.5)
        self.model = GaussianProcessRegressor(kernel=self.kernel)
        self.acquisition_params = {
            'type': 'ucb',
            'u': 0.5  # TODO: this is an arbitrary value
        }

    def update(self, features, target):
        self.observations.append([features, target])
        X, y = np.stack(self.observations)
        self.model.fit(X, y)
        return self

    def suggest(self):
        samples = np.random.uniform(
            self.feature_bounds[:, 0],
            self.feature_bounds[:, 1],
            size=(1000, self.feature_bounds.shape[0]))
        predictions = self.model.predict(samples)
        predictions
        # TODO: Maximize acquisition function and return optimum

    def step(self, features, target):
        self.update(features, target)
        return self.suggest()

    def _ucb(self, x):
        u = self.acquisition_params['u']
        mean, std = self.model.predict(x, return_std=True)
        return mean + u * std
