import torch
import torchvision.transforms as transforms
import torchvision

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define train parameters
batch_size = 32

# Load dataset
cifar_train = torchvision.datasets.CIFAR10('../cifar10/', download=True, train=True, transform=transform)
cifar_test = torchvision.datasets.CIFAR10('../cifar10/', download=True, train=False, transform=transform)

cifar_train_dataloader = torch.utils.data.DataLoader(cifar_train, batch_size=batch_size)
cifar_test_dataloader = torch.utils.data.DataLoader(cifar_test, batch_size=batch_size)

print("Loaded")

import cnn_train
train_acc, train_loss, test_acc, test_loss = cnn_train.train(cifar_train_dataloader, cifar_test_dataloader)
