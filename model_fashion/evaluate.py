import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import neural_network
import sys

if len(sys.argv) - 1 != 1:
    print("usage: python evaluate.py <batch_size: 64>")
    sys.exit(1)

batch_size = int(sys.argv[1])

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

model = neural_network.NeuralNetwork()
model.load_state_dict(torch.load(f"{neural_network.model_name}.pth"))

# verification
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()

correct_count = 0
total_count = 0

for t in test_data:
    x, y = t[0], t[1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
        if predicted == actual:
            correct_count += 1
        total_count += 1

print("\nStats\n=============")
print(f"Correct Count: {correct_count}")
print(f"Incorrect Count: {total_count - correct_count}")
print(f"Total Count: {total_count}")
