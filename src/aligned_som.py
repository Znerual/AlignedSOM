from typing import Tuple, List
import numpy as np
from random import randrange
from tqdm import tqdm
from minisom import MiniSom

from src.som_layer import Layer


class AlignedSom():
    """Alignd SOM implementation

    Details of the algorithm can be found in:
    Pampalk, Elias. "Aligned self-organizing maps." Proceedings of the Workshop on Self-Organizing Maps. 2003.
    DOI: https://www.researchgate.net/publication/2887633_Aligned_Self-Organizing_Maps (no DOI found)

    """
    def __init__(self,
                 dimension: Tuple[int, int],
                 data: np.ndarray,
                 aspect_selection: List[bool],
                 num_layers: int = 100,
                 layer_distance_ratio: float = 0.1,
                 sigma: float = 1.0,
                 learning_rate: float = 0.5,
                 neighborhood_function: str = 'gaussian',
                 activation_distance: str = 'euclidean',
                 codebook_inizialization_type: str = 'random',
                 random_seed: int = None):
        """Construction of Alignd SOM 
 
        Args:
            dimension (Tuple[int, int]): _description_
            data (np.ndarray): _description_
            aspect_selection (List[bool]): _description_
            num_layers (int, optional): _description_. Defaults to 100.
            layer_distance_ratio (float, optional): _description_. Defaults to 0.1.
            sigma (float, optional): _description_. Defaults to 1.0.
            learning_rate (float, optional): _description_. Defaults to 0.5.
            neighborhood_function (str, optional): _description_. Defaults to 'gaussian'.
            activation_distance (str, optional): _description_. Defaults to 'euclidean'.
            codebook_inizialization_type (str, optional): _description_. Defaults to 'random'.
            random_seed (int, optional): _description_. Defaults to None.
        """
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
        self.data = data
        self._dimension = dimension
        self._aspect_selection = aspect_selection
        self._num_layers = num_layers
        self._layer_distance_ratio = layer_distance_ratio
        self._neighborhood_function = neighborhood_function
        self._codebook_inizialization_type = codebook_inizialization_type
        self._random_seed = random_seed

        self._weights_by_layer: np.ndarray = self._create_weights_by_layer()
        self.layers: List[Layer] = self._create_layers()
        self._layer_distances = self._create_layer_distances()

    def train(self, num_iterations: int):
        n_observations = self.data.shape[0]
        for t in tqdm(range(num_iterations)):
            selected_layer = randrange(0, self._num_layers)
            selected_observation = randrange(0, n_observations)
            # print(f'selected layer: {selected_layer}')
            # print(f'selected observation: {selected_observation}')
            winner = self.layers[selected_layer].winner(
                self.data[selected_observation] * self._weights_by_layer[selected_layer])
            for i, layer in enumerate(self.layers):
                # print(f'current layer: {i}')
                # ĺayer_dist = self.layer_distance(t, num_iterations, np.abs(selected_layer - i))
                ĺayer_dist = self._layer_distances[np.abs(selected_layer - i)]
                # print(f'distance: {ĺayer_dist}')
                layer.update(self.data[selected_observation] * self._weights_by_layer[i],
                             winner,
                             ĺayer_dist,
                             t,
                             num_iterations)

    # return the codebook weights for all layers
    def get_layer_weights(self) -> List[np.ndarray]:
        return [layer.get_weights() for layer in self.layers]

    def _create_layer_distances(self):
        # used default gaussian with sigma = 1.0 for distance between layers
        x_mash, y_mash = np.meshgrid(np.arange(1), np.arange(self._num_layers - 1))
        d = 2
        ax = np.exp(-np.power(x_mash-x_mash[0], 2)/d)
        ay = np.exp(-np.power(y_mash-y_mash[0], 2)/d)
        layer_distances = (ax * ay).T[0]
        layer_distances *= self._layer_distance_ratio
        layer_distances = np.insert(layer_distances, 0, 1.0)  # distence to layer itself
        return layer_distances

    # create a weights matrix for two concepts in a feature matrix
    # the shape corresponds to shape (num_layers, input_len))
    # where num_soms is the number of soms trained
    def _create_weights_by_layer(self):
        if self._aspect_selection.shape[0] != self._input_len:
            raise AttributeError('aspect_selection has to have the same dimension as input_len')
        column_weights = []
        weights_concept_1 = np.linspace(0, 1, self._num_layers)
        weights_concept_2 = np.linspace(1, 0, self._num_layers)
        for i in self._aspect_selection:
            if i:
                column_weights.append(weights_concept_1)
            else:
                column_weights.append(weights_concept_2)
        return np.column_stack(column_weights)

    # initialize all layers of the aligned SOM
    def _create_layers(self) -> List[Layer]:
        layers = []
        if self._codebook_inizialization_type == 'random':
            inital_weights = self._create_random_weights()
        elif self._codebook_inizialization_type == 'pretrained':
            inital_weights = self._create_weights_by_training_one_some()
        else:
            raise AttributeError('codebook_inizialization_type has to be "random" or "pretrained"')
        for weights in self._weights_by_layer:
            layers.append(Layer(
                dimension=self._dimension,
                input_len=self._input_len,
                initial_codebook=np.array(inital_weights * weights, dtype=np.float32),
                sigma=self._sigma,
                learning_rate=self._learning_rate,
                neighborhood_function=self._neighborhood_function,
                activation_distance=self._activation_distance,
                random_seed=self._random_seed))
        return layers

    def _create_random_weights(self):
        if self._random_seed:
            np.random.seed(self._random_seed)
        return np.random.random((self._dimension[0], self._dimension[1], self._input_len))

    def _create_weights_by_training_one_some(self):
        # som trained on not weighted features (same as middle layer)
        middle_som = MiniSom(
            x=self._dimension[0],
            y=self._dimension[1],
            input_len=self._input_len,
            sigma=self._sigma,
            learning_rate=self._learning_rate,
            neighborhood_function=self._neighborhood_function,
            topology='rectangular',
            activation_distance=self._activation_distance,
            random_seed=self._random_seed)
        middle_som.train(self.data, 1000)
        return middle_som.get_weights()
