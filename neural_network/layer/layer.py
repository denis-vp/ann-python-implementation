import random
import numpy as np


class Layer:
    def __init__(self, num_neuron_in: int, num_neuron_out: int, activation_function: callable):
        self.num_neuron_in: int = num_neuron_in
        self.num_neuron_out: int = num_neuron_out
        self.activation_function: callable = np.vectorize(activation_function)

        self.weights: np.ndarray[(num_neuron_out, num_neuron_in + 1), float]
        self.bias: float
        self._initialize_weights_and_biases()

    def _initialize_weights_and_biases(self):
        # TODO: tweak the way weights are initialized
        random_weights = np.random.rand(self.num_neuron_in, self.num_neuron_out)
        self.weights = np.vstack([random_weights, np.ones(self.num_neuron_out)])
        self.bias = random.random()

    def compute_outputs(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.shape != (1, self.num_neuron_out):
            raise ValueError(f"Expected input shape {self.num_neuron_in}, but got {inputs.shape}")

        inputs = np.concatenate(([self.bias], inputs))

        outputs = np.dot(self.weights, inputs)
        outputs = self.activation_function(outputs)

        return outputs
