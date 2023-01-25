import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose

from shared.mnist_neural_network import NeuralNetwork, train, test

import sys

if len(sys.argv) - 1 != 1:
    print("usage: python model_mnist_evaluate.py <batch_size: 64>")
    sys.exit(1)

batch_size = int(sys.argv[1])
model_name = "mnist"

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=Compose([
    ToTensor(),
    Normalize(
        (0.1307,), (0.3081,))
    ])
)

test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

model = NeuralNetwork()
model.load_state_dict(torch.load(f"{model_name}.pth"))

# verification
model.eval()

correct_count = 0
total_count = 0

for t in test_data:
    x, y = t[0], t[1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = pred[0].argmax(0).item(), y
        # print(f'Predicted: "{predicted}", Actual: "{actual}"')
        if predicted == actual:
            correct_count += 1
        total_count += 1

print("\nStats\n=============")
print(f"Correct Count: {correct_count}")
print(f"Incorrect Count: {total_count - correct_count}")
print(f"Total Count: {total_count}")
