'''
Loss: Interfaz de las funciones de costo donde se define el uso del metodo
__call__
y gradient.
MSE: Clase donde se implementa la funcion de costo mse.
'''
import numpy as np

class Loss:
    def __call__(self,y_true,y_pred):
        raise NotImplementedError
    def gradient(self,y_true,y_pred):
        raise NotImplementedError

class MSE(Loss):
    def __call__(self,y_true,y_pred):
        return np.mean((y_pred-y_true)**2)
    def gradient(self, y_true, y_pred):
        N = y_true.shape[0]
        return (2.0/N)*(y_pred-y_true)
    
class MSE_L2(Loss):
    def __call__(self,y_true,y_pred,pesos=None,lreg=0.0):
        reg_loss=0.0
        for w in pesos:
            reg_loss+=lreg*np.sum(w*w)
        return np.mean((y_pred-y_true)**2)+reg_loss
    def gradient(self, y_true, y_pred):
        N = y_true.shape[0]
        return (2.0/N)*(y_pred-y_true)
