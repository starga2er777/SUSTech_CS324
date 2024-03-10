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

        # x: num * in_features
        # w: in_features * out_features
        # x * w  gives  num * out_features
        self.params = {'weight': np.random.normal(0, 10, (in_features, out_features)),
                       'bias': np.zeros(out_features)}
        self.grads = {'weight': np.zeros((in_features, out_features)),
                      'bias': np.zeros(out_features)}
        self.x = None


    def forward(self, x):
        """
        Performs the forward pass using the formula: output = xW + b
        TODO: Implement the forward pass.
        """
        # self.x = x
        # return np.dot(x, self.params['weight']) + self.params['bias']
        self.x = x

        self.out = np.dot(x, self.params['weight']) + self.params['bias']

        return self.out

    def backward(self, dout):
        """
        Backward pass to calculate gradients of loss w.r.t. weights and inputs.
        TODO: Implement the backward pass.
        """
        # self.grads['weight'] = np.dot(self.x.T, dout)
        # self.grads['bias'] = np.sum(dout, axis = 0)

        # dout = np.dot(dout, self.params['weight'].T)

        # return dout
        self.grads['weight'] = np.dot(self.x.T, dout)
        self.grads['bias'] = np.sum(dout, axis = 0)

        dout = np.dot(dout, self.params['weight'].T)

        return dout

class ReLU(object):
    def forward(self, x):
        """
        Applies the ReLU activation function element-wise to the input.
        Formula: output = max(0, x)
        TODO: Implement the forward pass.
        """
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        """
        Computes the gradient of the ReLU function.
        TODO: Implement the backward pass.
        Hint: Gradient is 1 for x > 0, otherwise 0.
        """
        # return dout * (self.x > 0).astype(float)
        drelu = np.zeros_like(self.x)
        drelu[self.x > 0 ] = 1

        return dout * drelu
    
class SoftMax(object):
    def forward(self, x):
        """
        Applies the softmax function to the input to obtain output probabilities.
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        TODO: Implement the forward pass using the Max Trick for numerical stability.
        """
        # b = np.max(x)
        # y = np.exp(x - b)
        # return y / y.sum()
        b = x.max()
        y = np.exp(x-b)
        x = (y.T / y.sum(axis = 1)).T

        self.x = x

        return x

    def backward(self, dout):
        """
        The backward pass for softmax is often directly integrated with CrossEntropy for simplicity.
        TODO: Keep this in mind when implementing CrossEntropy's backward method.
        """
        # return dout
        dx = np.zeros(dout.shape, dtype=np.float64)

        for i in range(0, dout.shape[0]):
            delta = self.x[i, :].reshape(-1, 1)
            delta = np.diagflat(delta) - np.dot(delta, delta.T)
            dx[i, :] = np.dot(delta, dout[i, :])

        return dx

class CrossEntropy(object):
    def forward(self, x, y):
        """
        Computes the CrossEntropy loss between predictions and true labels.
        Formula: L = -sum(y_i * log(p_i)), where p is the softmax probability of the correct class y.
        TODO: Implement the forward pass.
        """
        # return -np.sum(y * np.log(x + 1e-10)) / len(y)

        out = -np.log(x[np.arange(x.shape[0]), y.argmax(1)]).mean()
        return out

    def backward(self, x, y):
        """
        Computes the gradient of CrossEntropy loss with respect to the input.
        TODO: Implement the backward pass.
        Hint: For softmax output followed by cross-entropy loss, the gradient simplifies to: p - y.
        """
        # return x - y
        
        dx = -(y / x) / y.shape[0]
        return dx
