import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose

from shared.mnist_neural_network import NeuralNetwork, train, test

import sys

if len(sys.argv) - 1 != 2:
    print("usage: python model_mnist_train.py <batch_size: 64> <epochs: 5>")
    sys.exit(1)

batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
model_name = "mnist"

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=Compose([
        ToTensor(),
        Normalize(
            (0.1307,), (0.3081,))
        ]
    )
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=Compose([
        ToTensor(),
        Normalize(
            (0.1307,), (0.3081,))
        ]
    )
)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(device, train_dataloader, model, loss_fn, optimizer)
    test(device, test_dataloader, model, loss_fn)

print("Training completed!")

# Save model
torch.save(model.state_dict(), f"{model_name}.pth")
print(f"Saved PyTorch Model State to {model_name}.pth")
