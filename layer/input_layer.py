from layer.layer import Layer


class InputLayer(Layer):
    def __init__(self, num_neuron_in: int, num_neuron_out: int):
        super().__init__(num_neuron_in, num_neuron_out, lambda x: x)

    def _initialize_weights_and_biases(self):
        self.biases = [0 for _ in range(self.num_neuron_out)]
        self.weights = [1 for _ in range(self.num_neuron_in * self.num_neuron_out)]
