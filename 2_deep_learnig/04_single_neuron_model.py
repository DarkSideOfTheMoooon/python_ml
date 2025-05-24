import numpy as np
import torch
import torch.nn as nn


'''
Write a Python function that simulates a single neuron 
with a sigmoid activation function for binary classification, 
handling multidimensional input features. 
The function should take a list of feature vectors 
(each vector representing multiple features for an example), 
associated true binary labels, 
and the neuron's weights (one for each feature) and bias as input.
It should return the predicted probabilities 
after sigmoid activation and
the mean squared error between the predicted probabilities and the true labels,
both rounded to four decimal places.

Input:
features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], 
labels = [0, 1, 0], 
weights = [0.7, -0.4], bias = -0.1
Output:
([0.4626, 0.4134, 0.6682], 0.3349)
'''
def single_neuron_model(features, labels, weights, bias):
    def sigmoid(x):
        return np.round(1 / (1 + np.exp(-x)), 4)
    features = np.array(features)
    weights = np.array(weights)
    z = np.dot(features, weights) + bias
    prob = sigmoid(z)
    mse = np.round(np.mean((np.array(labels) - prob) ** 2), 4)
    return prob.tolist(), mse


'''
features = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]],
labels = [1, 0, 0], 
initial_weights = [0.1, -0.2],
initial_bias = 0.0,
learning_rate = 0.1, 
epochs = 2

output
updated_weights = [0.1036, -0.1425], 
updated_bias = -0.0167, 
mse_values = [0.3033, 0.2942]
'''

def train_single_neuron(features, labels, weights, bias, lr=0.1, epochs=100):
    features = np.array(features)
    labels = np.array(labels)
    weights = np.array(weights)
    bias = np.array(bias)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(x):
        return x * (1 - x)
    n = len(labels)
    mse_loss = []
    for epoch in range(epochs):
        # foward
        z = np.dot(features,weights) + bias
        predict = sigmoid(z)
        # mse loss
        loss = np.mean((np.array(labels) - predict) ** 2)
        mse_loss.append(loss)
        #gradient descent
        dloss_dpred = 2 * (predict - labels) / n
        dpred_dz = sigmoid_derivative(predict)
        dz_dw = features 
        dz_db = 1

        grad_z = dloss_dpred * dpred_dz
        grad_w = np.dot(grad_z, dz_dw) 
        grad_b = np.sum(grad_z) * dz_db
        # update weights and bias
        weights -= lr*grad_w
        bias -= lr*grad_b

    return weights , bias, mse_loss
