from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

input_size = 3 * 32 * 32
hidden_size = 128
hidden_size2 = 64
num_classes = 10


class Model(nn.Module):
    
    def __init__(self):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(Model, self).__init__()
        
        self.layers = nn.ModuleList()
    
        # add layers
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_size, hidden_size2))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_size2, num_classes))

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        out = x.view(-1, 3 * 32 * 32)
        for layer in self.layers:
            out = layer(out)
        return out


def train(train_loader, test_loader, learning_rate, num_epochs):
    model = Model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # for output
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            pred = model(inputs)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(pred.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss.append(running_loss / len(train_loader))
        train_acc.append(correct_train / total_train)

        # Evaluating on test set
        correct_test = 0
        total_test = 0
        test_running_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = model(images)
                test_running_loss += loss_fn(outputs, labels).item()

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_loss.append(test_running_loss / len(test_loader))
        test_acc.append(correct_test / total_test)

        # print process
        print(f'Epoch: {epoch}, Test Accuracy: {test_acc[-1]:.4f}.')

    return train_acc, train_loss, test_acc, test_loss