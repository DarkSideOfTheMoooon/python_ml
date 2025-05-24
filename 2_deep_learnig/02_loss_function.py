import numpy as np
import math

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # 避免对数为0
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
def cross_entropy_loss_derivative(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # 避免对数为0
    return - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred))

def binary_cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # 避免对数为0
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
def binary_cross_entropy_loss_derivative(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # 避免对数为0
    return - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred))

def categorical_cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # 避免对数为0
    return -np.sum(y_true * np.log(y_pred), axis=1) 
def categorical_cross_entropy_loss_derivative(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # 避免对数为0
    return - (y_true / y_pred)

def categorical_accuracy(y_true, y_pred):
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    return np.mean(y_pred_classes == y_true_classes)
def binary_accuracy(y_true, y_pred):
    y_pred_classes = np.round(y_pred)
    return np.mean(y_pred_classes == y_true)


# Compute Multi-class Cross-Entropy Loss
def cross_entropy_loss(true_labels, predicted_probs):
    """
    true_labels: list of list, one-hot encoded true labels
    predicted_probs: list of list, predicted probabilities for each class
    """
    n = len(true_labels)
    loss = 0.0
    for i in range(n):
        for j in range(len(true_labels[i])):
            if true_labels[i][j] == 1:
                loss -= math.log(predicted_probs[i][j] + 1e-12)
    return loss / n

# 示例
predicted_probs = [[0.7, 0.2, 0.1], [0.3, 0.6, 0.1]]
true_labels = [[1, 0, 0], [0, 1, 0]]
print(cross_entropy_loss(true_labels, predicted_probs))
