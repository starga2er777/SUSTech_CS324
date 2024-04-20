import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import time

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

import cnn_train
start_time = time.time()

train_acc, train_loss, test_acc, test_loss = cnn_train.train(cifar_train_dataloader, cifar_test_dataloader)

end_time = time.time()
execution_time = end_time - start_time
print("Time Elapsed:", execution_time, "s")

data_to_save = np.column_stack((train_acc, train_loss, test_acc, test_loss))
np.savetxt('output.txt', data_to_save, header='train_acc train_loss test_acc test_loss')

print("Train Complete!")