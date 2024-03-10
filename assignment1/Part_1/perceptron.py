import numpy as np

class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=100, learning_rate=0.1):
        """
        Initializes the perceptron object.
        - n_inputs: Number of inputs.
        - max_epochs: Maximum number of training cycles.
        - learning_rate: Magnitude of weight changes at each training cycle.
        - weights: Initialize weights (including bias).
        """
        self.n_inputs = n_inputs  # Fill in: Initialize number of inputs
        self.max_epochs = max_epochs  # Fill in: Initialize maximum number of epochs
        self.learning_rate = learning_rate  # Fill in: Initialize learning rate
        self.weights = np.zeros(n_inputs + 1)  # Fill in: Initialize weights with zeros
        
    def forward(self, input_vec):
        """
        Predicts label from input.
        Args:
            input_vec (np.ndarray): Input array of training data, input vec must be all samples
        Returns:
            arrays: Predicted labels.
        """
        # insert a column for bias term
        biased_input_vec = np.insert(input_vec, 0, 1, axis=1)
        # prediction
        pred = np.dot(biased_input_vec, self.weights)
        label_array = np.where(pred >= 0, 1, -1)
        return label_array


        
    # def train(self, training_inputs, labels):
    #     """
    #     Trains the perceptron.
    #     Args:
    #         training_inputs (list of np.ndarray): List of numpy arrays of training points.
    #         labels (np.ndarray): Array of expected output values for the corresponding point in training_inputs.
    #     """
    #     # we need max_epochs to train our model
    #     for _ in range(self.max_epochs): 
    #         """
    #             What we should do in one epoch ? 
    #             you are required to write code for 
    #             1.do forward pass
    #             2.calculate the error
    #             3.compute parameters' gradient 
    #             4.Using gradient descent method to update parameters(not Stochastic gradient descent!,
    #             please follow the algorithm procedure in "perceptron_tutorial.pdf".)
    #         """
    #         # shuffle inputs
    #         rand_idx = np.arange(len(labels))
    #         np.random.shuffle(rand_idx)
    #         X_shuffled = training_inputs[rand_idx]
    #         Y_shuffled = labels[rand_idx]

    #         # forward pass
    #         cur_pred = self.forward(X_shuffled)

    #         # examine each prediction
    #         for i in range(len(cur_pred)):
    #             if cur_pred[i] * Y_shuffled[i] < 0:
    #                 # update weight
    #                 self.weights += self.learning_rate * Y_shuffled[i] * np.insert(X_shuffled[i], 0, 1)


    def train(self, training_inputs, labels):
        """
        Trains the perceptron.
        Args:
            training_inputs (list of np.ndarray): List of numpy arrays of training points.
            labels (np.ndarray): Array of expected output values for the corresponding point in training_inputs.
        Returns:
            list: Array recording the loss in each epoch.
        """
        loss = []
        # we need max_epochs to train our model
        for _ in range(self.max_epochs): 
            """
                What we should do in one epoch ? 
                you are required to write code for 
                1.do forward pass
                2.calculate the error
                3.compute parameters' gradient 
                4.Using gradient descent method to update parameters(not Stochastic gradient descent!,
                please follow the algorithm procedure in "perceptron_tutorial.pdf".)
            """
            # shuffle inputs
            rand_idx = np.arange(len(labels))
            np.random.shuffle(rand_idx)
            X_shuffled = training_inputs[rand_idx]
            Y_shuffled = labels[rand_idx]

            # forward pass
            cur_pred = self.forward(X_shuffled)

            # calculate loss for the current epoch
            epoch_loss = np.sum(cur_pred != Y_shuffled) / len(labels)
            loss.append(epoch_loss)

            # examine each prediction
            for i in range(len(cur_pred)):
                if cur_pred[i] * Y_shuffled[i] < 0:
                    # update weight
                    self.weights += self.learning_rate * Y_shuffled[i] * np.insert(X_shuffled[i], 0, 1)
        
        return loss