from dataclasses import dataclass


@dataclass
class DataPoint:
    inputs: list[float]
    expected_outputs: list[float]
