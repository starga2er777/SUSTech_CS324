from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from pytorch_mlp import MLP
import torch
import torch.nn as nn
import torch.optim as optim

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = [20]
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    predicted_classes = torch.argmax(predictions, dim=1)
    true_classes = torch.argmax(targets, dim=1)
    # get accuracy with mean
    accuracy = torch.sum(predicted_classes == true_classes).item() / len(targets)

    return accuracy

def train(dataset, dnn_hidden_units=DNN_HIDDEN_UNITS_DEFAULT, learning_rate=LEARNING_RATE_DEFAULT, max_steps=MAX_EPOCHS_DEFAULT, eval_freq=EVAL_FREQ_DEFAULT):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE

    # Read dataset
    x_train = torch.tensor(dataset['x_train'], dtype=torch.float32)
    y_train = torch.tensor(dataset['y_train'], dtype=torch.float32)
    x_test = torch.tensor(dataset['x_test'], dtype=torch.float32)
    y_test = torch.tensor(dataset['y_test'], dtype=torch.float32)
        
    mlp = MLP(x_train.shape[1], dnn_hidden_units, y_train.shape[1])
    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.SGD(mlp.parameters(), lr=learning_rate)

    # For output
    test_loss = []
    test_acc = []

    for step in range(max_steps):

    # TODO: Implement the training loop
    # 1. Forward pass
    # 2. Compute loss
    # 3. Backward pass (compute gradients)
    # 4. Update weights

        # Set to train mode
        mlp.train()

        # Forward pass
        pred = mlp.forward(x_train)
        loss = loss_fn(pred, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update layers
        optimizer.step()
        

        # Set to eval mode
        mlp.eval()
        test_pred = mlp(x_test)
        test_loss.append(loss_fn(test_pred, y_test).item())
        test_acc.append(accuracy(test_pred, y_test))
        if step % eval_freq == 0:
            print(f"Step: {step}, Test Loss: {test_loss[-1]}, Test Accuracy: {test_acc[-1]}%")

    return test_acc, test_loss

# def main():
#     """
#     Main function
#     """
#     train()

# if __name__ == '__main__':
#     # Command line arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
#                       help='Comma separated list of number of units in each hidden layer')
#     parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
#                       help='Learning rate')
#     parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
#                       help='Number of epochs to run trainer.')
#     parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
#                           help='Frequency of evaluation on the test set')
#     FLAGS, unparsed = parser.parse_known_args()
#     main()