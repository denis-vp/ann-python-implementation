import numpy as np

from neural_network.functions.activation_functions import ActivationFunction, ReLU, Tanh
from neural_network.functions.loss_functions import LossFunction


class LayerLearnData:
    """
    This class is used to store intermediate data during the forward pass that is needed during the backward pass.
    input: The input to the layer
    weighted_inputs: The weighted sum + biases of the input
    activations: The output of the layer after applying the activation function
    loss_values: The loss values for each neuron in the layer
    """

    def __init__(self, num_output_neurons: int):
        self.inputs: np.ndarray = None
        self.weighted_inputs: np.ndarray = None
        self.activations: np.ndarray = None
        self.loss_values: np.ndarray = np.array([0 for _ in range(num_output_neurons)])


class Layer:
    def __init__(self, num_neuron_in: int, num_neuron_out: int, activation_function: ActivationFunction):
        self.num_neuron_in: int = num_neuron_in
        self.num_neuron_out: int = num_neuron_out
        self.activation_function: ActivationFunction = activation_function

        self.weights: np.ndarray[(num_neuron_out, num_neuron_in), np.float64]
        self.weights_gradients: np.ndarray[(num_neuron_out, num_neuron_in), np.float64]
        self.weights_velocities: np.ndarray[(num_neuron_out, num_neuron_in), np.float64]

        self.biases: np.ndarray[num_neuron_out, np.float64]
        self.biases_gradients: np.ndarray[num_neuron_out, np.float64]
        self.biases_velocities: np.ndarray[num_neuron_out, np.float64]

        self._initialize_parameters()

    def _initialize_parameters(self):
        self.weights = np.random.rand(self.num_neuron_out, self.num_neuron_in)
        if isinstance(self.activation_function, ReLU):
            # He initialization
            self.weights = self.weights * np.sqrt(2 / self.num_neuron_in)
        elif isinstance(self.activation_function, Tanh):
            # Xavier initialization
            self.weights = self.weights * np.sqrt(1 / self.num_neuron_in)

        self.weights_gradients = np.zeros((self.num_neuron_out, self.num_neuron_in))
        self.weights_velocities = np.zeros((self.num_neuron_out, self.num_neuron_in))

        self.biases = np.zeros(self.num_neuron_out)
        self.biases_gradients = np.zeros(self.num_neuron_out)
        self.biases_velocities = np.zeros(self.num_neuron_out)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        Calculate the output of the layer given the inputs.
        :param inputs: Inputs to the layer, 1D numpy array
        :return: Output values of the layer, 1D numpy array
        """
        # Calculate the weighted sum of the inputs
        outputs = np.dot(self.weights, inputs)
        # Add the biases
        outputs += self.biases

        # Apply the activation function
        outputs = self.activation_function(outputs)
        outputs += 1e-10  # Add a small constant to prevent underflow / log(0) in the loss function

        return outputs

    def forward_learn(self, inputs: np.ndarray, learn_data: LayerLearnData) -> np.ndarray:
        """
        Forward pass through the layer.
        Calculate the output of the layer given the inputs and save the intermediate data in the learn_data object.
        :param inputs: Inputs to the layer, 1D numpy array
        :param learn_data: Object to save the intermediate data, LayerLearnData
        :return: Output values of the layer, 1D numpy array
        """
        learn_data.inputs = inputs.copy()

        # Calculate the weighted sum of the inputs
        outputs = np.dot(self.weights, inputs)
        # Add the biases
        outputs += self.biases
        learn_data.weighted_inputs = outputs.copy()

        # Apply the activation function
        outputs = self.activation_function(outputs)
        outputs += 1e-10  # Add a small constant to prevent underflow / log(0) in the loss function
        learn_data.activations = outputs.copy()

        return outputs

    def apply_gradients(self, learning_rate: float, regularization_rate: float, momentum: float):
        """
        Update the weights and biases of the layer using the gradients.
        Also reset the gradients to zero.
        :param learning_rate: Learning rate, float
        :param regularization_rate: Regularization rate, float
        :param momentum: Momentum, float
        """
        weight_decay: float = 1 - learning_rate * regularization_rate

        for i in range(self.weights.shape[0]):  # Iterate over rows
            for j in range(self.weights.shape[1]):  # Iterate over columns
                self.weights_velocities[i][j] = momentum * self.weights_velocities[i][j] - learning_rate * \
                                                self.weights_gradients[i][j]
                self.weights[i][j] = weight_decay * self.weights[i][j] + self.weights_velocities[i][j]

        for i in range(self.num_neuron_out):
            self.biases_velocities[i] = momentum * self.biases_velocities[i] - learning_rate * self.biases_gradients[i]
            self.biases[i] += self.biases_velocities[i]

    def zero_grad(self):
        """
        Reset the gradients of the layer to zero.
        """
        self.weights_gradients = np.zeros((self.num_neuron_out, self.num_neuron_in))
        self.biases_gradients = np.zeros((self.num_neuron_out, 1))

    def update_gradients(self, layer_learn_data: LayerLearnData):
        """
        Update the gradients of the layer using the loss values in the learn_data object.
        :param layer_learn_data: Object containing the loss values, LayerLearnData
        """
        for i in range(self.weights.shape[0]):  # Iterate over rows
            for j in range(self.weights.shape[1]):  # Iterate over columns
                self.weights_gradients[i][j] += layer_learn_data.loss_values[i] * layer_learn_data.inputs[j]

        for i in range(self.num_neuron_out):
            self.biases_gradients[i] += layer_learn_data.loss_values[i]

    def calculate_output_layer_loss(self, expected_outputs: np.ndarray, layer_learn_data: LayerLearnData,
                                    loss_function: LossFunction):
        """
        Calculate the loss values for the output layer.
        The loss values are calculated with respect to the expected outputs.
        :param expected_outputs: True values, 1D numpy array
        :param layer_learn_data: Object to save the intermediate data, LayerLearnData
        :param loss_function: Loss function to use, LossFunction
        """
        loss_derivatives = loss_function.derivative(expected_outputs, layer_learn_data.activations)
        outputs_derivatives = self.activation_function.derivative(layer_learn_data.weighted_inputs)
        layer_learn_data.loss_values = loss_derivatives * outputs_derivatives

    def calculate_hidden_layer_loss(self, layer_learn_data: LayerLearnData, previous_layer: 'Layer',
                                    previous_loss_values: np.ndarray):
        """
        Calculate the loss values for a hidden layer.
        The loss values are calculated with respect to the previous layer's loss values.
        :param layer_learn_data: Object to save the intermediate data, LayerLearnData
        :param previous_layer: Previous layer in the backpropagation, Layer
        :param previous_loss_values: Previous layer's loss values in the backpropagation, 1D numpy array
        """
        outputs_derivatives = self.activation_function.derivative(layer_learn_data.weighted_inputs)
        layer_learn_data.loss_values = np.dot(previous_loss_values, previous_layer.weights) * outputs_derivatives
