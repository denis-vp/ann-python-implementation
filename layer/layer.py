import random


class Layer:
    def __init__(self, num_neuron_in: int, num_neuron_out: int, activation_function: callable):
        self.num_neuron_in: int = num_neuron_in
        self.num_neuron_out: int = num_neuron_out
        self.activation_function: callable = activation_function

        self.weights: list[float] = []
        self.biases: list[float] = []
        self._initialize_weights_and_biases()

    def _initialize_weights_and_biases(self):
        # todo: tweak the initial weights and biases
        self.weights = [random.random() for _ in range(self.num_neuron_in * self.num_neuron_out)]
        self.biases = [random.random() for _ in range(self.num_neuron_out)]

    def compute_outputs(self, inputs: list[float]) -> list[float]:
        outputs = []

        # compute the weighted sum for each output neuron with bias
        for node_out in range(self.num_neuron_out):
            weighted_sum = self.biases[node_out]
            for node_in in range(self.num_neuron_in):
                weighted_sum += inputs[node_in] * self._get_weight(node_in, node_out)
            outputs.append(weighted_sum)

        # apply activation function
        outputs = [self.activation_function(output) for output in outputs]

        return outputs

    def _get_weight(self, input_neuron_index: int, output_neuron_index: int) -> float:
        flat_index = output_neuron_index * self.num_neuron_in + input_neuron_index
        return self.weights[flat_index]
