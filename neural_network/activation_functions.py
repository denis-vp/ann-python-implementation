import numpy as np


def threshold(x: float) -> float:
    return 1 if x >= 0 else -1


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def tanh(x: float) -> float:
    return np.tanh(x)


def rectifier(x: float) -> float:
    return max(0, x)
