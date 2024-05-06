from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the activation function for the given input.
        :param x: 1D numpy array
        :return: 1D numpy array
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the activation function for the given input.
        :param x: 1D numpy array
        :return: 1D numpy array
        """
        pass


class Threshold(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)


class Sigmoid(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self(x) * (1 - self(x))


class Tanh(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # e_pos = np.exp(x)
        # e_neg = np.exp(-x)
        # return (e_pos - e_neg) / (e_pos + e_neg)
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - self(x) ** 2


class ReLU(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)


class LeakyReLU(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, 0.01 * x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0.01)


class Softmax(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self(x) * (1 - self(x))


class Linear(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
