from typing import Tuple, List
import numpy as np
from random import randrange
from minisom import MiniSom
from tqdm import tqdm

from som_layer import Layer

# todo: how to normalize weights ?


class AlignedSom(MiniSom):
    def __init__(self,
                 dimension: Tuple[int, int],
                 data: np.ndarray,  # 2d numpy array
                 concept_indices: List[bool],  # boolean list if feature belongs to concept A or concept B
                 num_layers: int = 100,
                 layer_distance_ratio: float = 0.1,
                 sigma: float = 1.0,
                 learning_rate: float = 0.5,
                 neighborhood_function: str = 'gaussian',
                 activation_distance: str = 'euclidean',
                 initial_codebook_inizialization: str = 'random',  # random or pretrained
                 random_seed=None):
        super().__init__(
            x=dimension[0],
            y=dimension[1],
            input_len=data.shape[1],
            sigma=sigma,
            learning_rate=learning_rate,
            neighborhood_function=neighborhood_function,
            topology='rectangular',
            activation_distance=activation_distance,
            random_seed=random_seed)
        self.dimension = dimension
        self.data = data
        self.concept_indices = concept_indices
        self.num_layers = num_layers
        self.layer_distance_ratio = layer_distance_ratio
        self._neighborhood_function = neighborhood_function
        self._initial_codebook_inizialization = initial_codebook_inizialization
        self.random_seed = random_seed

        self.weights_by_layer: np.ndarray = self._create_weights_by_layer()
        self.layers: List[Layer] = self._create_layers()

    def train(self,
              data: np.ndarray,  # 2d numpy array,
              num_iterations):
        n_observations = data.shape[0]
        for t in tqdm(range(num_iterations)):
            selected_layer = randrange(0, self.num_layers)
            selected_observation = randrange(0, n_observations)
            # print(f'selected layer: {selected_layer}')
            # print(f'selected observation: {selected_observation}')
            winner = self.layers[selected_layer].winner(
                data[selected_observation] * self.weights_by_layer[selected_layer])
            for i, layer in enumerate(self.layers):
                # print(f'current layer: {i}')
                ĺayer_dist = self.layer_distance(t, num_iterations, np.abs(selected_layer - i))
                # print(f'distance: {ĺayer_dist}')
                layer.update(data[selected_observation] * self.weights_by_layer[i],
                             winner,
                             ĺayer_dist,
                             t,
                             num_iterations)

    # the distance between one layer and the next is defined by the distance between neighboring units
    # multiplyed by some fraction "layer_distance_ratio"
    def layer_distance(self, t, max_iteration, grid_distance):
        if grid_distance == 0.0:
            return 1.0
        sig = self._decay_function(self._sigma, t, max_iteration)
        distance_neighboring_units = self.neighborhood((0, 0), sig)[(0, 1)]
        return distance_neighboring_units * (self.layer_distance_ratio / grid_distance)

    # return the codebook weights for all layers
    def get_layer_weights(self) -> List[np.ndarray]:
        return [layer.get_weights() for layer in self.layers]

    # create a weights matrix for two concepts in a feature matrix
    # the shape corresponds to shape (num_layers, input_len))
    # where num_soms is the number of soms trained
    def _create_weights_by_layer(self):
        if self.concept_indices.shape[0] != self._input_len:
            raise AttributeError('concept_indices has to have the same dimension as input_len')
        column_weights = []
        weights_concept_1 = np.linspace(0, 1, self.num_layers)
        weights_concept_2 = np.linspace(1, 0, self.num_layers)
        for i in self.concept_indices:
            if i:
                column_weights.append(weights_concept_1)
            else:
                column_weights.append(weights_concept_2)
        return np.column_stack(column_weights)

    # initialize all layers of the aligned SOM
    def _create_layers(self) -> List[Layer]:
        layers = []
        if self._initial_codebook_inizialization == 'random':
            inital_weights = self._create_random_weights()
        elif self._initial_codebook_inizialization == 'pretrained':
            inital_weights = self._create_weights_by_training_one_some()
        else:
            raise AttributeError('initial_codebook_inizialization has to be "random" or "pretrained"')
        for weights in self.weights_by_layer:
            layers.append(Layer(
                dimension=self.dimension,
                input_len=self._input_len,
                initial_codebook=np.array(inital_weights * weights, dtype=np.float32),
                sigma=self._sigma,
                learning_rate=self._learning_rate,
                neighborhood_function=self._neighborhood_function,
                activation_distance=self._activation_distance,
                random_seed=self.random_seed))
        return layers

    def _create_random_weights(self):
        if self.random_seed:
            np.random.seed(self.random_seed)
        return np.random.random((self.dimension[0], self.dimension[1], self._input_len))

    def _create_weights_by_training_one_some(self):
        # som trained on not weighted features (same as middle layer)
        middle_som = MiniSom(
            x=self.dimension[0],
            y=self.dimension[1],
            input_len=self._input_len,
            sigma=self._sigma,
            learning_rate=self._learning_rate,
            neighborhood_function=self._neighborhood_function,
            topology='rectangular',
            activation_distance=self._activation_distance,
            random_seed=self.random_seed)
        middle_som.train(self.data, 1000)
        return middle_som.get_weights()
