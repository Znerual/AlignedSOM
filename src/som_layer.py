from typing import Tuple
import numpy as np
from minisom import MiniSom


class Layer(MiniSom):
    """One layer of the Alignd SOM extending the MiniSom library"""
    def __init__(self,
                 dimension: Tuple[int, int],
                 input_len: int,
                 initial_codebook: np.ndarray,
                 sigma: float = 1.0,
                 learning_rate: float = 0.5,
                 neighborhood_function: str = 'gaussian',
                 activation_distance: str = 'euclidean',
                 random_seed: int = None) -> None:
        """Constructs one layer of the Alignd SOM

        Args:
            dimension (Tuple[int, int]): x and y dimensions of the resulting SOM
            input_len (int): Dimension of the training data
            initial_codebook (np.ndarray): Weight vectors of the initial codebook with dimensions (x, y, input_len)
            sigma (float, optional): Initial spread of the neighborhood function. Defaults to 1.0.
            learning_rate (float, optional): initial Learning rate. Defaults to 0.5.
            neighborhood_function (str, optional):
                Type of function to use for computing the neighborhood.
                Possible values: 'gaussian', 'mexican_hat', 'bubble', 'triangle'
                Defaults to 'gaussian'.
            activation_distance (str, optional):
                Type of function used for computing the distances between unit weight vectors and feature vectors
                Possible values: 'euclidean', 'cosine', 'manhattan', 'chebyshev'
                Defaults to 'euclidean'.
            random_seed (int, optional):
                Random seed used for all operations which use randomness.
                Defaults to None
        """

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

        # after initialization of the weights by MiniSom override them with pre defined codebook
        self._weights = initial_codebook

    # changed update to include the distance to the layer in the neighborhood
    def update(self,
               input_vector: np.array,
               winner_position: Tuple[int, int],
               layer_dist: float,
               time_point: int,
               max_iteration: int) -> None:
        """Update the SOM codebook similar to normal SOM update including the distance between layers
            (extended update function of the MiniSom library)

        Args:
            input_vector (np.array): 1d input vector used for training
            winner_position (Tuple[int, int]): Tuple indicating the position of the winning unit [x, y]
            layer_dist (float):
                fraction representing the distance between layers
                max 1.0 which is normal SOM update for same layer
            time_point (int):
                current number of iteration of the training algorithm used for
                determining the decay of learning rate and neighborhood size
            max_iteration (int):
                number of iterations used for training the map used for
                determining the decay of learning rate and neighborhood size
        """
        eta = self._decay_function(self._learning_rate, time_point, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = self._decay_function(self._sigma, time_point, max_iteration)
        # improves the performances
        g = self.neighborhood(winner_position, sig) * eta * layer_dist
        # w_new = eta * neighborhood_function * (x-w)
        self._weights += np.einsum('ij, ijk->ijk', g, input_vector-self._weights)
