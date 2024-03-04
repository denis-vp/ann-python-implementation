from layer.input_layer import InputLayer
from layer.layer import Layer
from neural_network.data_point import DataPoint


class NeuralNetwork:
    def __init__(self, num_input_neurons: int, num_output_neurons: int,
                 num_hidden_layers: int, hidden_layer_sizes: list[int],
                 activation_functions: list[callable]):

        self.num_input_neurons = num_input_neurons
        self.num_output_neurons = num_output_neurons
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_functions = activation_functions

        self.layers = []
        self._initialize_layers()

    def _initialize_layers(self):
        input_layer = InputLayer(self.num_input_neurons, self.hidden_layer_sizes[0])
        self.layers.append(input_layer)

        for i in range(self.num_hidden_layers - 1):
            hidden_layer = Layer(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i + 1],
                                 self.activation_functions[i])
            self.layers.append(hidden_layer)

        output_layer = Layer(self.hidden_layer_sizes[-1], self.num_output_neurons, self.activation_functions[-1])
        self.layers.append(output_layer)

    def compute_outputs(self, data: DataPoint) -> list[float]:
        outputs = data.inputs
        for layer in self.layers:
            outputs = layer.compute_outputs(outputs)
        return outputs

    @staticmethod
    def normalize_outputs(outputs: list[float]) -> list[float]:
        return [output / sum(outputs) for output in outputs]

    @staticmethod
    def get_decision(outputs: list[float]) -> int:
        return outputs.index(max(outputs))

#     Neural network training

    @staticmethod
    def neuron_cost(expected: float, actual: float) -> float:
        # Power of 2 to make the cost always positive and emphasize the difference
        return (expected - actual) ** 2

    def total_cost(self, data: DataPoint) -> float:
        outputs = self.compute_outputs(data)
        return sum([self.neuron_cost(data.expected_outputs[i], outputs[i]) for i in range(len(outputs))])

    def total_cost_avg(self, data_set: list[DataPoint]) -> float:
        return sum([self.total_cost(data) for data in data_set]) / len(data_set)

    def apply_gradients(self, learning_rate: float):
        for layer in self.layers:
            layer.apply_gradients(learning_rate)

    def learn(self, data_set: list[DataPoint], learning_rate: float):
        h = 0.0001
        original_cost = self.total_cost_avg(data_set)

        for layer in self.layers:
            for neuron_in in range(layer.num_neuron_in):
                for neuron_out in range(layer.num_neuron_out):
                    layer.weights[neuron_in][neuron_out] += h
                    delta_cost = self.total_cost_avg(data_set) - original_cost
                    layer.weights[neuron_in][neuron_out] -= h
                    layer.gradient_weights[neuron_in][neuron_out] = delta_cost / h

            for bias in range(layer.num_neuron_out):
                layer.biases[bias] += h
                delta_cost = self.total_cost_avg(data_set) - original_cost
                layer.biases[bias] -= h
                layer.gradient_biases[bias] = delta_cost / h

        self.apply_gradients(learning_rate)
