import numpy as np
# def binary_cross_entropy(y_true, y_pred):
#     return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

# def binary_cross_entropy_prime(y_true, y_pred):
#     return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

import numpy as np

def log_sum_exp(x):
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    return np.mean(loss)

def binary_cross_entropy_prime(y_true, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    dy = ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
    return dy
