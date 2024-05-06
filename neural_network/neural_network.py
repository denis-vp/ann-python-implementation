import numpy as np

from neural_network.functions.activation_functions import ActivationFunction
from neural_network.functions.loss_functions import LossFunction
from neural_network.layer import LayerLearnData, Layer


class NeuralNetwork:
    def __init__(self, hidden_layer_sizes: list[int], activation_functions: list[ActivationFunction],
                 loss_function: LossFunction):

        self.loss_function = loss_function

        self.layers = []
        self._initialize_layers(hidden_layer_sizes, activation_functions)

        self.learn_data = [LayerLearnData(layer.num_neuron_out) for layer in self.layers]

    def _initialize_layers(self, hidden_layer_sizes: list[int], activation_functions: list[callable]):
        for i in range(len(hidden_layer_sizes) - 1):
            hidden_layer = Layer(hidden_layer_sizes[i], hidden_layer_sizes[i + 1],
                                 activation_functions[i])
            self.layers.append(hidden_layer)

    def forward(self, inputs: np.ndarray, learn: bool = True) -> np.ndarray:
        """
        Forward pass through the network.
        Calculate the output of the network given the inputs.
        Store intermediate data in the learn_data is learn is True.
        :param inputs: Inputs to the network, 1D numpy array
        :param learn: Whether to store intermediate data for learning, bool
        :return: Output values of the network, 1D numpy array
        """
        outputs = inputs
        for i, layer in enumerate(self.layers):
            if learn:
                outputs = layer.forward_learn(outputs, self.learn_data[i])
            else:
                outputs = layer.forward(outputs)
        return outputs

    def get_loss(self, expected_outputs: np.ndarray, outputs: np.ndarray) -> float:
        """
        Calculate the loss of the network given the outputs and the expected outputs.
        :param expected_outputs: True values of the outputs, 1D numpy array
        :param outputs: Predicted values of the outputs, 1D numpy array
        :return: Loss value, float
        """
        return self.loss_function(expected_outputs, outputs)

    def backward(self, expected_outputs: np.ndarray):
        """
        Calculate the gradients of the network given the expected outputs.
        :param expected_outputs: True values of the outputs, 1D numpy array
        """
        # Update the gradients for the output layer
        output_layer = self.layers[-1]
        output_layer_data = self.learn_data[-1]

        output_layer.calculate_output_layer_loss(expected_outputs, output_layer_data, self.loss_function)
        output_layer.update_gradients(output_layer_data)

        # Update the gradients for the hidden layers
        # Start from the second last layer and go backwards until the first hidden layer
        for i in range(len(self.layers) - 2, 0, -1):
            hidden_layer = self.layers[i]
            hidden_layer_data = self.learn_data[i]
            previous_layer_data = self.learn_data[i + 1]

            hidden_layer.calculate_hidden_layer_loss(hidden_layer_data, self.layers[i + 1],
                                                     previous_layer_data.loss_values)
            hidden_layer.update_gradients(hidden_layer_data)

    def apply_gradients(self, learning_rate: float, regularization_rate: float, momentum: float):
        """
        Update the weights and biases of the network using the gradients.
        :param learning_rate: Learning rate, float
        :param regularization_rate: Regularization rate, float
        :param momentum: Momentum, float
        """
        # Start from the second layer and go until the last layer
        for layer in self.layers:
            layer.apply_gradients(learning_rate, regularization_rate, momentum)

    def zero_grad(self):
        """
        Zero the gradients of all the layers in the network.
        """
        self.learn_data = [LayerLearnData(layer.num_neuron_out) for layer in self.layers]
        for layer in self.layers:
            layer.zero_grad()
