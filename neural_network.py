from activation_functions import rectifier
from layer.input_layer import InputLayer
from layer.layer import Layer


class NeuralNetwork:
    def __init__(self, num_input_neurons: int, num_output_neurons: int,
                 num_hidden_layers: int, hidden_layer_size: int, activation_functions: list[callable] = None):
        self.num_input_neurons = num_input_neurons
        self.num_output_neurons = num_output_neurons
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        if activation_functions:
            self.activation_functions = activation_functions
        else:
            self.activation_functions = [rectifier for _ in range(num_hidden_layers)]

        self.layers = []
        self._initialize_layers()

    def _initialize_layers(self):
        input_layer = InputLayer(self.num_input_neurons, self.hidden_layer_size)
        self.layers.append(input_layer)

        for _ in range(self.num_hidden_layers):
            hidden_layer = Layer(self.hidden_layer_size, self.hidden_layer_size, self.activation_functions[_])
            self.layers.append(hidden_layer)

        output_layer = Layer(self.hidden_layer_size, self.num_output_neurons, self.activation_functions[-1])
        self.layers.append(output_layer)

    def compute_outputs(self, inputs: list[float]) -> list[float]:
        outputs = inputs
        for layer in self.layers:
            outputs = layer.compute_outputs(outputs)
        return outputs

    @staticmethod
    def normalize_outputs(outputs: list[float]) -> list[float]:
        return [output / sum(outputs) for output in outputs]

    @staticmethod
    def get_decision(outputs: list[float]) -> int:
        return outputs.index(max(outputs))
