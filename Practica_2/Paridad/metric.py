'''
funciones: accuracy y MSE
'''
import numpy as np

def accuracy(y_pred, y_true):
    y_pred_bin = np.sign(y_pred)
    return np.mean(y_pred_bin == y_true)

def MSE(y_pred, y_true):
    return np.mean(np.sum((y_pred - y_true)**2, axis=1))