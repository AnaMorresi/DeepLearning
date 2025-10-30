import metric
import numpy as np
'''
Optimizador: Interfaz para los optimizadores.
SGD: clase que implementa el optimizador stochastic gradient descendent.
'''
class Optimizador:
    def __init__(self, lr=0.01):
        self.lr=lr

    def gradient_descent(self,x,y,forward,loss,backward,history_loss,history_acc):
        y_pred=forward(x)          # Forward
        loss_val=loss(y,y_pred)    # Loss
        history_loss.append(loss_val)
        history_acc.append(metric.accuracy(y_pred,y))
        grad_loss=loss.gradient(y,y_pred)      # Gradiente loss
        backward(grad_loss)  # Backward

class SGD(Optimizador):
    def __init__(self, lr=0.01):
        super().__init__(lr)

    def __call__(self,x,y,forward,loss,backward,history_loss,history_acc):
        permut = np.random.permutation(x.shape[0])
        x_sh = x[permut]
        y_sh = y[permut]
        for i in range(x.shape[0]):
            self.gradient_descent(x_sh[i:i+1],y_sh[i:i+1],forward,loss,backward,history_loss,history_acc)

class GD(Optimizador):
    def __init__(self, lr=0.01):
        super().__init__(lr)

    def __call__(self,x,y,forward,loss,backward,history_loss,history_acc):
        self.gradient_descent(x,y,forward,loss,backward,history_loss,history_acc)

class Mini_Batch(Optimizador):
    def __init__(self, lr=0.01, batch_size=1):
        super().__init__(lr)
        self.batch_size = batch_size

    def __call__(self,x,y,forward,loss,backward,history_loss,history_acc):
        num_train = x.shape[0]
        num_batches = int(np.ceil(num_train/self.batch_size))
        permut = np.random.permutation(num_train)
        x_sh = x[permut]
        y_sh = y[permut]
        for i in range(num_batches):
            start = i*self.batch_size
            end = min(start + self.batch_size, num_train)
            xb = x_sh[start:end]
            yb = y_sh[start:end]
            self.gradient_descent(xb,yb,forward,loss,backward,history_loss,history_acc)

