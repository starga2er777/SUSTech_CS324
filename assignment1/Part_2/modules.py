import numpy as np

class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Initializes a linear (fully connected) layer. 
        TODO: Initialize weights and biases.
        - Weights should be initialized to small random values (e.g., using a normal distribution).
        - Biases should be initialized to zeros.
        Formula: output = x * weight + bias
        """
        # Initialize weights and biases with the correct shapes.
        self.params = {'weight': np.random.normal(0, 0.01, size=(out_features, in_features)),
                       'bias': np.zeros((out_features, 1))}
        self.grads = {'weight': np.zeros((out_features, in_features)),
                      'bias': np.zeros((out_features, 1))}
        self.x = None


    def forward(self, x):
        """
        Performs the forward pass using the formula: output = Wx + b
        TODO: Implement the forward pass.
        """
        self.x = x
        return np.dot(self.params['weight'], x) + self.params['bias']

    def backward(self, dout):
        """
        Backward pass to calculate gradients of loss w.r.t. weights and inputs.
        TODO: Implement the backward pass.
        """
        self.grads['bias'] += dout
        self.grads['weight'] += np.dot(self.x.T, dout)

        return None

class ReLU(object):
    def forward(self, x):
        """
        Applies the ReLU activation function element-wise to the input.
        Formula: output = max(0, x)
        TODO: Implement the forward pass.
        """
        return np.maximum(0, x)

    def backward(self, dout):
        """
        Computes the gradient of the ReLU function.
        TODO: Implement the backward pass.
        Hint: Gradient is 1 for x > 0, otherwise 0.
        """
        return None

class SoftMax(object):
    def forward(self, x):
        """
        Applies the softmax function to the input to obtain output probabilities.
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        TODO: Implement the forward pass using the Max Trick for numerical stability.
        """
        m = np.max(x)
        return m / np.sum(np.exp(x - m))

    def backward(self, dout):
        """
        The backward pass for softmax is often directly integrated with CrossEntropy for simplicity.
        TODO: Keep this in mind when implementing CrossEntropy's backward method.
        """
        return dout

class CrossEntropy(object):
    def forward(self, x, y):
        """
        Computes the CrossEntropy loss between predictions and true labels.
        Formula: L = -sum(y_i * log(p_i)), where p is the softmax probability of the correct class y.
        TODO: Implement the forward pass.
        """
        return -np.sum(y * np.log(x))

    def backward(self, x, y):
        """
        Computes the gradient of CrossEntropy loss with respect to the input.
        TODO: Implement the backward pass.
        Hint: For softmax output followed by cross-entropy loss, the gradient simplifies to: p - y.
        """
        return x - y
