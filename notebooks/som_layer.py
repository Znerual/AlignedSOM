from typing import Tuple
import numpy as np
from minisom import MiniSom


class Layer(MiniSom):
    def __init__(self,
                 dimension: Tuple[int, int],
                 input_len,
                 initial_codebook,
                 sigma=1,
                 learning_rate=0.5,
                 neighborhood_function='gaussian',
                 activation_distance='euclidean',
                 random_seed=None):
        super().__init__(
            x=dimension[0],
            y=dimension[1],
            input_len=input_len,
            sigma=sigma,
            learning_rate=learning_rate,
            neighborhood_function=neighborhood_function,
            topology='rectangular',
            activation_distance=activation_distance,
            random_seed=random_seed)

        self._weights = initial_codebook

    # changed update to include the distance to the layer in the neighborhood
    # todo: not sure if only winning unit updated or whole neighbourhood for other layers
    def update(self, x, win, layer_dist, t, max_iteration):
        eta = self._decay_function(self._learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = self._decay_function(self._sigma, t, max_iteration)
        # improves the performances
        g = self.neighborhood(win, sig) * eta * layer_dist
        # w_new = eta * neighborhood_function * (x-w)
        self._weights += np.einsum('ij, ijk->ijk', g, x-self._weights)
