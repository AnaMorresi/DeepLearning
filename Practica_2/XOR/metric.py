'''
funciones: accuracy y MSE
'''
import numpy as np

def accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

def MSE(y_pred, y_true):
    return np.mean(np.sum((y_pred - y_true)**2, axis=1))