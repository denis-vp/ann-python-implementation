import random


class Layer:
    def __init__(self, num_neuron_in: int, num_neuron_out: int, activation_function: callable):
        self.num_neuron_in: int = num_neuron_in
        self.num_neuron_out: int = num_neuron_out
        self.activation_function: callable = activation_function

        self.weights: list[list[float]] = []
        self.biases: list[float] = []
        self._initialize_weights_and_biases()

        self.gradient_weights: list[list[float]] = [[0 for _ in range(self.num_neuron_out)] for _ in range(self.num_neuron_in)]
        self.gradient_biases: list[float] = [0 for _ in range(self.num_neuron_out)]

    def _initialize_weights_and_biases(self):
        # Todo: tweak the initial weights and biases
        self.weights = [[random.random() for _ in range(self.num_neuron_out)] for _ in range(self.num_neuron_in)]
        self.biases = [random.random() for _ in range(self.num_neuron_out)]

    def compute_outputs(self, inputs: list[float]) -> list[float]:
        outputs = []

        # Compute the weighted sum for each output neuron with bias
        for node_out in range(self.num_neuron_out):
            weighted_sum = self.biases[node_out]
            for node_in in range(self.num_neuron_in):
                weighted_sum += inputs[node_in] * self.weights[node_in][node_out]
            outputs.append(weighted_sum)

        # Apply activation function
        outputs = [self.activation_function(output) for output in outputs]

        return outputs

#     Neural network training

    def apply_gradients(self, learning_rate: float):
        for neuron_out in range(self.num_neuron_out):
            self.biases[neuron_out] -= learning_rate * self.gradient_biases[neuron_out]
            for neuron_in in range(self.num_neuron_in):
                self.weights[neuron_in][neuron_out] -= learning_rate * self.gradient_weights[neuron_in][neuron_out]
