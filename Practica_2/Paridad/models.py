'''
Network: Clase que implementa una red neuronal feedfoward.
'''
import numpy as np
import metric

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
            input=layer.backward(input,self.optimizer.lr)
        return input
    
    def set_loss(self,loss):
        '''Setea la loss function'''
        self.loss=loss

    def set_optimizer(self,optimizer):
        '''Setea el optimizador'''
        self.optimizer=optimizer
    
    def train(self,x,y,epochs=10):
        '''Entrenamiento completo de la red'''
        history_loss=[]
        history_acc=[]

        for epoch in range(epochs):
            self.optimizer(x,y,self.forward,self.loss,self.backward,history_loss,history_acc)

        return history_loss,history_acc

    def predict(self,x):
        return self.forward(x)


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

