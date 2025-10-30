import metric
import numpy as np
'''
Optimizador: Interfaz para los optimizadores.
SGD: clase que implementa el optimizador stochastic gradient descendent.
'''
class Optimizador:
    def __init__(self, lr=0.01, lreg=0.0):
        self.lr=lr
        self.lreg=lreg

    def gradient_descent(self,x,y,model):
        y_pred=model.forward(x)          # Forward
        pesos = model.get_weights()
        loss_val=float(model.loss(y,y_pred,pesos,self.lreg))    # Loss
        history_loss=(loss_val)
        history_acc=(metric.accuracy(y_pred,y))
        grad_loss=model.loss.gradient(y,y_pred)      # Gradiente loss
        model.backward(grad_loss)  # Backward

        return history_loss,history_acc

class SGD(Optimizador):
    def __init__(self, lr=0.01, lreg=0.0):
        super().__init__(lr,lreg)

    def __call__(self,x,y,model):
        permut = np.random.permutation(x.shape[0])
        x_sh = x[permut]
        y_sh = y[permut]
        h_loss_vec=[]
        h_acc_vec=[]
        for i in range(x.shape[0]):
            h_loss,h_acc=self.gradient_descent(x_sh[i:i+1],y_sh[i:i+1],model)
            h_loss_vec.append(h_loss)
            h_acc_vec.append(h_acc)
        return h_loss_vec,h_acc_vec

class GD(Optimizador):
    def __init__(self, lr=0.01, lreg=0.0):
        super().__init__(lr,lreg)

    def __call__(self,x,y,model):
        h_loss,h_acc=self.gradient_descent(x,y,model)
        h_loss_vec=[h_loss]
        h_acc_vec=[h_acc]
        return h_loss_vec,h_acc_vec

class Mini_Batch(Optimizador):
    def __init__(self, lr=0.01, batch_size=1, lreg=0.0):
        super().__init__(lr,lreg)
        self.batch_size = batch_size

    def __call__(self,x,y,model):
        num_train = x.shape[0]
        num_batches = int(np.ceil(num_train/self.batch_size))
        permut = np.random.permutation(num_train)
        x_sh = x[permut]
        y_sh = y[permut]
        h_loss_vec=[]
        h_acc_vec=[]
        for i in range(num_batches):
            start = i*self.batch_size
            end = min(start + self.batch_size, num_train)
            xb = x_sh[start:end]
            yb = y_sh[start:end]
            h_loss,h_acc=self.gradient_descent(xb,yb,model)
            h_loss_vec.append(h_loss)
            h_acc_vec.append(h_acc)
        return h_loss_vec,h_acc_vec

