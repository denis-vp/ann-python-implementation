import numpy as np

from layer.input_layer import InputLayer
from layer.layer import Layer
from neural_network.data_point import DataPoint


class NeuralNetwork:
    def __init__(self, num_input_neurons: int, num_output_neurons: int,
                 hidden_layer_sizes: list[int], activation_functions: list[callable]):

        self.num_input_neurons = num_input_neurons
        self.num_output_neurons = num_output_neurons
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_functions = activation_functions

        self.layers = []
        self._initialize_layers()

    def _initialize_layers(self):
        input_layer = InputLayer(self.num_input_neurons, self.hidden_layer_sizes[0])
        self.layers.append(input_layer)

        for i in range(len(self.hidden_layer_sizes) - 1):
            hidden_layer = Layer(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i + 1],
                                 self.activation_functions[i])
            self.layers.append(hidden_layer)

        output_layer = Layer(self.hidden_layer_sizes[-1], self.num_output_neurons, self.activation_functions[-1])
        self.layers.append(output_layer)

    def compute_outputs(self, data: DataPoint) -> np.ndarray:
        outputs = data.inputs
        for layer in self.layers:
            outputs = layer.compute_outputs(outputs)
        return outputs

    @staticmethod
    def normalize_outputs(outputs: np.ndarray) -> np.ndarray:
        def normalize(output: float) -> float:
            return output / outputs_sum

        outputs_sum = sum(outputs)
        normalize = np.vectorize(normalize)

        return normalize(outputs)

    @staticmethod
    def get_decision(outputs: list[float]) -> int:
        return outputs.index(max(outputs))
