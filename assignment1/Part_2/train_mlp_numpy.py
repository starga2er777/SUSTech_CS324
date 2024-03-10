import argparse
import numpy as np
from mlp_numpy import MLP  
from modules import CrossEntropy


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

def train(x_train, y_train, x_test, y_test, dnn_hidden_units, learning_rate, max_steps, eval_freq):
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

    # TODO: Initialize your MLP model and loss function (CrossEntropy) here

    mlp = MLP(x_train.shape[1], dnn_hidden_units, y_train.shape[1])

    for step in range(max_steps):
        # TODO: Implement the training loop
        # 1. Forward pass
        # 2. Compute loss
        # 3. Backward pass (compute gradients)
        # 4. Update weights

        # forward pass
        pred = mlp.forward(x_train)

        # compute loss
        loss = mlp.loss_fn.forward(pred, y_train)

        # compute gradients
        dout = mlp.loss_fn.backward(pred, y_train)

        # backward pass
        mlp.backward(dout)

        # update weights
        for layer in mlp.layers:
            if hasattr(layer, 'params'):  # Check if the layer has parameters
                layer.params['weight'] -= learning_rate * layer.grads['weight']
                layer.params['bias'] -= learning_rate * layer.grads['bias']

        if step % eval_freq == 0 or step == max_steps - 1:
            # TODO: Evaluate the model on the test set
            # 1. Forward pass on the test set
            # 2. Compute loss and accuracy
            print(f"Step: {step}, Loss: {loss}, Accuracy: {accuracy(mlp.forward(x_test), y_test)}")
    
    print("Training complete!")




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
