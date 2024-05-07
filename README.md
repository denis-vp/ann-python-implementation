# Artificial Neural Network Python Implementation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-purple)

This repository contains a Python implementation of a neural network from scratch utilizing the NumPy library.  
The implementation is tested on various datasets from Kaggle, such as the mushroom and iris datasets.  

## Overview

The main goal of this project is to provide a clear and simple implementation of a feed forward neural network and  
to see how it performs on various datasets.

## Features

- Implementation of a feedforward neural network architecture.
- Training over various datasets from Kaggle.
- Evaluation of model performance using accuracy metrics.
- Python code designed for readability and easy understanding.

## Usage

The implementation uses numpy arrays for data handling.

The `neural_network.py` file contains the NeuralNetwork class.  
The `activation_functions.py` file contains the ActivationFunction classes used in the neural network.  
The `loss_functions.py` file contains the LossFunction classes used in the neural network.  

Simply import the `NeuralNetwork` class and create an instance of the class with the desired parameters.
#### Expected parameters:
- `layer_sizes`: A list of integers representing the number of neurons in each layer. (e.g. ['input_size', 16, 'ouput_size'])
- `activation_functions`: A list of ActivationFunction objects for each layer. (e.g. [ReLU(), Softmax()])
- `loss_function`: A LossFunction object for the loss calculation. (e.g. CrossEntropy())

```python
from neural_network import NeuralNetwork
from activation_functions import ReLU, Softmax
from loss_functions import CrossEntropyLoss

layer_sizes = [22, 16, 4]
activation_functions = [ReLU(), Softmax()]
loss_function = CrossEntropyLoss()

model = NeuralNetwork(layer_sizes, activation_functions, loss_function)
```

### Training

Choose the training parameters and train the model on the dataset.

```python
learning_rate = 0.1
regularization = 0.01
momentum = 0.9
epochs = 1000
```

The `forward` method is called with one sample at a time.  
The `get_loss` method is called after each forward pass to calculate the loss. (optional)  
The `backward` method is called after each forward pass to calculate the gradients.  
The `apply_gradients` method is called after each backward pass to update the weights.

```python
for epoch in range(epochs):
    loss = 0
    for x, y in zip(X_train, y_train):
        y_pred = model.forward(x)

        loss += model.get_loss(y, y_pred)

        model.backward(y, y_pred)

        model.apply_gradients(learning_rate, regularization, momentum)
    loss /= len(X_train)
```

### Evaluation

After training the model, evaluate the model on the test dataset.  
The `forward` method is called with one sample at a time with the parameter `learn=False` to disable learning.

```python
   for x, y in zip(X_test, y_test):
        y_pred = model.forward(x)
        y_pred = np.argmax(y_pred)
        y = np.argmax(y)
        if y_pred == y:
            correct += 1
    accuracy = correct / len(X_test)
```


## Requirements

- Python 3.x
- NumPy
- Matplotlib (for visualization, optional)
- scikit-learn (for dataset splitting, optional)
- pandas (for dataset loading, optional)
- Jupyter Notebook (for running the example, optional)

## Contributions

Contributions to improve the codebase, add new features, or fix issues are welcome.  
Please fork the repository and submit a pull request outlining your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
