import numpy as np
import matplotlib.pyplot

class MLP:
    def __init__(self):
        pass

#use this as input layer as well
class Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_L2=5e-4, bias_regularizer_L2=5e-4):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))

        self.weight_momentums = np.zeros_like(self.weights)
        self.bias_momentums = np.zeros_like(self.biases)

        self.weight_regularizer_L2 = weight_regularizer_L2
        self.bias_regularizer_L2 = bias_regularizer_L2

    def forward(self, inputs):
        self.inputs = inputs #need this for backprop
        self.output = np.dot(inputs, self.weights) + self.biases
    
    """At this level, we are receiving the gradient of the Loss with respect to the outputs of this layer.
        The dot product of the transposed inputs and the gradients means we are given the total gradient for
        each weight. Since the bias is added the same way for every sample in the batch, we sum the gradient over
        the batch axis. Since the gradient goes in the opposite direction, we use the transpose, often denoted as 
        W^T. So we take the gradient flowing in from the layer after it, and use the chain rule to compute gradients 
        for its own weights and biases and send further back via dinputs. 
        """
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)\
        
        #project the loss gradient back through the weight matrix, using W^T
        self.dinputs = np.dot(dvalues, self.weights.T)

#if output < 0, make it 0
#else, keep it the same
class ReLu: 
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        #make a copy since we need to modify the original variable
        self.dinputs = dvalues.copy()

        self.dinputs[self.inputs <= 0 ] = 0

class Softmax:
    def forward(self, inputs):
        #first subtract the values from the max in the vector and exponentiate
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 

        #now do the division
        probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probs

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

#so this will be taking in the softmax outputs, already normalized
#formula is (sum of) y_true_i * log(y_pred)
class CategoricalCrossEntropy:
    def forward(self, y_pred, y_true):
        sample_nums = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        #probabilities for categorical labels
        #if categorical, just index to find the guess
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(sample_nums), y_true]
        #if not, then we apply the dot product logic from the formula
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        #now do the losses
        losses = -np.log(correct_confidences)
        #average loss across batch
        return np.mean(losses)
    
    #if using softmax for the activation of the output, pass those in as values
    def backward(self, values, y_true):
        samples = len(values)

        labels = len(values[0])
        #if sparse, one hot encode them
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        #now calculate gradient
        self.dinputs = -y_true / values
        #and now normalize
        self.dinputs = self.dinputs / samples

    def regularization_loss(self, layer):
        regularization_loss = 0

        #we calculate these only when factor is greater than 0
        if layer.weight_regularizer_L2 > 0:
            regularization_loss += layer.weight_regularizer_L2 * np.sum(layer.weights * layer.weights)

        if layer.bias_regularizer_L2 > 0:
            regularization_loss += layer.bias_regularizer_L2 * np.sum(layer.biases * layer.biases)

        return regularization_loss


class MSELoss:
    def forward(self, y_true, y_pred):
       loss = np.mean((y_pred - y_true) ** 2)
       self.loss = loss
       return loss

    def backward(self, y_true, y_pred):
       num_of_elements = y_true.size
       return 2 * (y_pred - y_true) / num_of_elements 
    
class SGDOptimizer:
    def  __init__(self, learning_rate = 1.0, momentum = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update_params(self, layer):
        #use momentum now
        layer.weight_momentums = self.momentum * layer.weight_momentums - self.learning_rate * layer.dweights
        layer.bias_momentums - self.momentum * layer.bias_momentums - self.learning_rate * layer.dbiases 

        layer.weights += layer.weight_momentums
        layer.biases += layer.bias_momentums
        


    


