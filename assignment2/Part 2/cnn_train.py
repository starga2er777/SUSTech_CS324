from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from cnn_model import CNN
import torch.optim as optim
import torch.nn as nn
import torch

# Default constants
INPUT_CHANNEL = 3
NUM_CLASSES = 10
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 800
EVAL_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'

FLAGS = None

# def accuracy(predictions, labels):
#     """
#     Computes the prediction accuracy, i.e., the average of correct predictions
#     of the network.
#     Args:
#         predictions: 2D float array of size [number_of_data_samples, n_classes]
#         labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
#     Returns:
#         accuracy: scalar float, the accuracy of predictions.
#     """
#     pass
    

def train(train_loader, test_loader, learning_rate=LEARNING_RATE_DEFAULT, num_epochs=MAX_EPOCHS_DEFAULT, optimizer_option=OPTIMIZER_DEFAULT):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE

    # Use GPU if available
    if torch.cuda.is_available():
        print("CUDA available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create model
    model = CNN(INPUT_CHANNEL, NUM_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss()
    if optimizer_option == 'ADAM':
        print("optimizer = ADAM")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else: 
        print("optimizer = SGD")
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # For output
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []

    print('Start training...')

    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        for i, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            batch_losses.append(loss.item())

        # Evaluate
        if epoch % EVAL_FREQ_DEFAULT == 0:
            # Evaluate the model on train set
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for inputs, labels in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            train_accuracy = correct / total
            train_acc.append(train_accuracy)
            train_loss.append(np.mean(batch_losses))

            # Evaluate the model on test set
            test_losses = []
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                test_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            test_accuracy = correct / total
            test_acc.append(test_accuracy)
            test_loss.append(np.mean(test_losses))
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Test Acc: {test_accuracy:.4f}')
    
    torch.save(model, 'cnn_model.pth')
    print('Model Saved')

    return train_acc, train_loss, test_acc, test_loss
