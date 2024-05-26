from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torch
from torch.utils.data import DataLoader, random_split

from dataset import PalindromeDataset
from lstm import LSTM
from utils import AverageMeter, accuracy

# Default params
INPUT_LENGTH_DEFAULT = 4
INPUT_DIM_DEFAULT = 1
NUM_CLASSES_DEFAULT = 10
NUM_HIDDEN_DEFAULT = 128
BATCH_SIZE_DEFAULT = 128
LEARNING_RATE_DEFAULT = 0.001
MAX_EPOCH_DEFAULT = 100
MAX_NORM_DEFAULT = 10.0
DATA_SIZE_DEFAULT = 100000
TRAIN_PROP = 0.8


def train(model, data_loader, optimizer, criterion, device):
    model.train()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        output = model.forward(batch_inputs)
        loss = criterion(output, batch_targets)
        loss.backward()

        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=MAX_NORM_DEFAULT)

        optimizer.step()

        losses.update(loss.item())
        accuracies.update(accuracy(output, batch_targets))
        # if step % 10 == 0:
        #     print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    return losses.avg, accuracies.avg


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        output = model.forward(batch_inputs)
        loss = criterion(output, batch_targets)

        losses.update(loss.item())
        accuracies.update(accuracy(output, batch_targets))

        # if step % 10 == 0:
        #     print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    return losses.avg, accuracies.avg


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    # Initialize the model that we are going to use
    model = LSTM(INPUT_LENGTH_DEFAULT, INPUT_DIM_DEFAULT, NUM_HIDDEN_DEFAULT, NUM_CLASSES_DEFAULT, BATCH_SIZE_DEFAULT, device)
    model.to(device)

    # For output:
    _train_loss = []
    _train_acc = []
    _val_loss = []
    _val_acc = []

    # Initialize the dataset and data loader
    dataset = PalindromeDataset(INPUT_LENGTH_DEFAULT, DATA_SIZE_DEFAULT, False)
    # Split dataset into train and validation sets
    train_size = int(dataset.total_len * TRAIN_PROP)
    val_size = int(dataset.total_len - train_size)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # Create data loaders for training and validation
    train_dloader = DataLoader(train_dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=False, drop_last=True)
    val_dloader = DataLoader(val_dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=False, drop_last=True)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE_DEFAULT)
    # scheduler = ...  # fixme

    for epoch in range(MAX_EPOCH_DEFAULT):
        # Train the model for one epoch
        train_loss, train_acc = train(
            model, train_dloader, optimizer, criterion, device)

        # Evaluate the trained model on the validation set
        val_loss, val_acc = evaluate(
            model, val_dloader, criterion, device)
        
        _train_loss.append(train_loss)
        _train_acc.append(train_acc)
        _val_loss.append(val_loss)
        _val_acc.append(val_acc)
        print(f'[{epoch}/{MAX_EPOCH_DEFAULT}]', f'Train Acc: {train_acc}, Test Acc: {val_acc}, Train Loss: {train_loss}, Test Loss: {val_loss}')

    print('Done training.')

main()
