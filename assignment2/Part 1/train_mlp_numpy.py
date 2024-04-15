import argparse
import numpy as np
from mlp_numpy import MLP  
from modules import CrossEntropy

DNN_HIDDEN_UNITS_DEFAULT = [20]
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10
BATCH_SIZE_DEFAULT = 800

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.
    
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding
    
    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    # TODO: Implement the accuracy calculation
    # Hint: Use np.argmax to find predicted classes, and compare with the true classes in targets
    
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets, axis=1)
    # get accuracy with mean
    acc = np.mean(predicted_classes == true_classes)
    
    return acc

def train(dataset, dnn_hidden_units=DNN_HIDDEN_UNITS_DEFAULT, learning_rate=LEARNING_RATE_DEFAULT, max_steps=MAX_EPOCHS_DEFAULT, eval_freq=EVAL_FREQ_DEFAULT, batch_size=BATCH_SIZE_DEFAULT):
    """
    Performs training and evaluation of MLP model.
    
    Args:
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set
        NOTE: Add necessary arguments such as the data, your model...
    """
    # TODO: Load your data here
    
    # Get dataset from parameters
    x_train = dataset['x_train']
    y_train = dataset['y_train']
    x_test = dataset['x_test']
    y_test = dataset['y_test']

    # TODO: Initialize your MLP model and loss function (CrossEntropy) here

    mlp = MLP(x_train.shape[1], dnn_hidden_units, y_train.shape[1])

    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []

    for step in range(max_steps):

        # TODO: Implement the training loop
        # 1. Forward pass
        # 2. Compute loss
        # 3. Backward pass (compute gradients)
        # 4. Update weights

        acc = 0
        loss = 0

        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # Forward pass
            pred = mlp.forward(x_batch)

            # Compute loss
            cur_loss = mlp.loss_fn.forward(pred, y_batch)
            loss += cur_loss

            # Compute accuracy
            batch_acc = accuracy(pred, y_batch)
            acc += batch_acc

            # Compute gradients
            dout = mlp.loss_fn.backward(pred, y_batch)

            # Backward pass
            mlp.backward(dout)

            # Update weights
            for layer in mlp.layers:
                if hasattr(layer, 'params'):
                    layer.params['weight'] -= learning_rate * layer.grads['weight'] / len(x_batch)
                    layer.params['bias'] -= learning_rate * layer.grads['bias'] / len(x_batch)

        train_acc.append(acc / len(x_train) * batch_size)
        train_loss.append(loss / len(x_train) * batch_size)


        # perform test

        test_pred = mlp.forward(x_test)
        loss = mlp.loss_fn.forward(test_pred, y_test)

        test_acc.append(accuracy(test_pred, y_test))
        test_loss.append(loss)


        # update weights
        for layer in mlp.layers:
            if hasattr(layer, 'params'):
                # here the learning rate is too big
                layer.params['weight'] -= learning_rate * layer.grads['weight'] / len(x_train)
                layer.params['bias'] -= learning_rate * layer.grads['bias'] / len(x_train)

        if step % eval_freq == 0 or step == max_steps - 1:
            # TODO: Evaluate the model on the test set
            # 1. Forward pass on the test set
            # 2. Compute loss and accuracy
            print(f"Step: {step}, Loss: {train_loss[-1]}, Accuracy: {test_acc[-1]}")
    
    print("Training complete!")
    return train_acc, train_loss, test_acc, test_loss




# def main():
#     """
#     Main function.
#     """
#     # Parsing command line arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
#                         help='Comma separated list of number of units in each hidden layer')
#     parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
#                         help='Learning rate')
#     parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
#                         help='Number of epochs to run trainer')
#     parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
#                         help='Frequency of evaluation on the test set')
#     FLAGS = parser.parse_known_args()[0]
    
#     train(FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq)

# if __name__ == '__main__':
#     main()
