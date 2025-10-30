'''
Network: Clase que implementa una red neuronal feedfoward.
'''
import numpy as np
import metric
from tqdm import tqdm
import time

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
    
    def add(self,layer):
        '''Agrega una capa a la red'''
        self.layers.append(layer)

    def forward(self,x):
        '''Metodo forward de toda la red'''
        out=x
        for layer in self.layers:
            out=layer.forward(out)
        return out

    def backward(self,grad_output):
        '''Metodo backward de toda la red'''
        input=grad_output
        for layer in reversed(self.layers):
            input=layer.backward(input,self.optimizer.lr,self.optimizer.lreg)
        return input
    
    def set_loss(self,loss):
        '''Setea la loss function'''
        self.loss=loss

    def set_optimizer(self,optimizer):
        '''Setea el optimizador'''
        self.optimizer=optimizer
    
    def train(self,x,y,epochs=10,x_val=None,y_val=None):
        '''Entrenamiento completo de la red'''
        history_loss=[]
        history_acc=[]
        history_loss_val=[]
        history_acc_val=[]

        print("\nIniciando entrenamiento...")
        start_time = time.time()

        with tqdm(total=epochs, desc="Entrenamiento") as pbar:
            for epoch in range(epochs):
                h_loss,h_acc=self.optimizer(x,y,self)
                if x_val is not None and y_val is not None:
                    y_pred_test=self.forward(x_val)
                    history_loss_val.append(self.loss(y_val,y_pred_test,self.get_weights(),self.optimizer.lreg))
                    history_acc_val.append(metric.accuracy(y_pred_test,y_val))

                history_loss.append(np.mean(h_loss))
                history_acc.append(np.mean(h_acc))

                if epoch%10 == 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / (epoch+1)) * (epochs - epoch - 1)
                    
                    current_loss = history_loss[-1]
                    current_acc = history_acc[-1]

                    pbar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'acc': f'{current_acc:.2%}',
                        'val_acc': f'{history_acc_val[-1]:.2%}',
                        'eta': f'{eta:.0f}s'
                    })
                    pbar.update(10)
                
                if x_val is not None and y_val is not None:
                    if (abs(history_acc_val[-1]-history_acc[-1]) > 0.09):
                        print('\nFinalizacion antes en epoca:', epoch)
                        print(f"\nEntrenamiento completado en {time.time() - start_time:.1f} segundos")
                        return history_loss,history_acc,history_loss_val,history_acc_val

        print(f"\nEntrenamiento completado en {time.time() - start_time:.1f} segundos")
        return history_loss,history_acc,history_loss_val,history_acc_val

    def predict(self,x):
        return self.forward(x)
    
    def get_weights(self):
        """Retorna lista de matrices de pesos de capas con parámetros"""
        weights = []
        for layer in self.layers:
            if hasattr(layer, "W"):
                weights.append(layer.W)
        return weights

class SkipNetwork(Network):
    def __init__(self, hidden, output):
        super().__init__()  # importante para tener loss, optimizer, etc.
        self.hidden = hidden    # capa que va a hidden
        self.output = output    # capas que van a output

    def forward(self, X):
        """Forward con conexión skip"""
        h = self.hidden.forward(X)
        y_pred = self.output.forward(h, X)  # pasa salida oculta y entrada original
        return y_pred

    def backward(self, grad_output):
        """Backward: propaga error desde la capa de salida hacia la oculta"""
        grad_hidden = self.output.backward(grad_output, self.optimizer.lr)
        self.hidden.backward(grad_hidden, self.optimizer.lr)

    def train(self, X, y, epochs=1000):
        """Entrenamiento completo del modelo con skip connection"""
        history_loss = []
        history_acc=[]

        for epoch in range(epochs):
            self.optimizer(X,y,self.forward,self.loss,self.backward,history_loss,history_acc)

        return history_loss, history_acc

