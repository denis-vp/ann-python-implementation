import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

BATCH_SIZE = 32


def my_flatten(data: torch.Tensor) -> list[float]:
    return data.flatten().tolist()


if __name__ == '__main__':
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor())

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    for batch, (inputs, targets) in enumerate(train_dataloader):
        if batch == 1:
            break

        inputs = [my_flatten(input) for input in inputs]
        print(inputs[0])
