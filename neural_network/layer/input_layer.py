import numpy as np

from neural_network.layer.layer import Layer


class InputLayer(Layer):
    def __init__(self, num_neuron_in: int, num_neuron_out: int):
        super().__init__(num_neuron_in, num_neuron_out, lambda x: x)

    def _initialize_weights_and_biases(self):
        self.weights = np.ones((self.num_neuron_in + 1, self.num_neuron_out))
        self.bias = 0
