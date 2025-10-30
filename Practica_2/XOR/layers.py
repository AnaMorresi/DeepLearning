'''
BaseLayer:  Clase generica de cualquier tipo de capa.
Input:      Representa la capa de entrada de la red neuronal que
            hereda las funcionalidades basicas de la clase BaseLayer.
Layer:      Clase generica de cualquier tipo de capa con pesos.
Dense:      Representa una capa densa que hereda las funcionalidades
de la clase Layer.
'''
import numpy as np

class BaseLayer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward(self, input_data):
        raise NotImplementedError
    def backward(self, grad_output, lr):
        raise NotImplementedError

class Input(BaseLayer):
    def forward(self, input_data):
        self.output = input_data
        return self.output  # entrada simple sin pesos

    def backward(self, grad_output, lr):
        return grad_output  # no tenemos parametros

class Layer(BaseLayer):
    def __init__(self, input_size, output_size, activation):
        super().__init__()  # llama a BaseLayer.__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation()
        self.W = None
        self.b = None

    def _init_params(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output, lr):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size, activation):
        super().__init__(input_size, output_size, activation)
        # inicializamos pesos y bias
        #np.random.seed(5)
        limit1 = np.sqrt(6/(input_size+output_size))  # Inicializacion Glorot
        self.W = np.random.uniform(-limit1,limit1,(input_size,output_size)).astype(np.float32)
        self.b = np.zeros((1, output_size))
        self.z = None

    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(self.input,self.W)+self.b
        self.output = self.activation(self.z)
        return self.output
    
    def backward(self, grad_output, lr, lreg=0.0):
        # grad_output = dL/dy de la capa anterior
        grad_z = grad_output*self.activation.gradient(self.z)   # dL/dz
        grad_W = np.dot(self.input.T, grad_z)
        if lreg>0:
            grad_W+=lreg*self.W
        grad_b = np.sum(grad_z, axis=0, keepdims=True)
        grad_input = np.dot(grad_z, self.W.T)

        self.W -= lr*grad_W
        self.b -= lr*grad_b

        return grad_input 

class Skip(Dense):
    """
    Capa densa con conexión directa (skip) desde la entrada original.
    y = f(W1*h + Wskip*x + b)
    """
    def __init__(self, input_size, hidden_size, output_size, activation):
        #np.random.seed(5)
        self.activation = activation()

        # Pesos del camino desde la capa oculta
        limit1 = np.sqrt(6/(hidden_size+output_size))  # Inicializacion Glorot
        self.W = np.random.uniform(-limit1,limit1,(hidden_size,output_size)).astype(np.float32)
        # Pesos del camino skip directo desde la entrada
        limit2 = np.sqrt(6/(input_size+output_size))  # Inicializacion Glorot
        self.W_skip = np.random.uniform(-limit2,limit2,(input_size,output_size)).astype(np.float32)
        
        self.b = np.zeros((1, output_size))
        self.z = None
        self.hidden_output = None
        self.skip_input = None
    
    def forward(self, hidden_output, skip_input):
        self.hidden_output = hidden_output
        self.skip_input = skip_input
        self.z = np.dot(hidden_output, self.W) + np.dot(skip_input, self.W_skip) + self.b
        self.output = self.activation(self.z)
        return self.output
    
    def backward(self, grad_output, lr):
        grad_z = grad_output * self.activation.gradient(self.z)

        grad_W_hidden = np.dot(self.hidden_output.T, grad_z)
        grad_W_skip = np.dot(self.skip_input.T, grad_z)
        grad_b = np.sum(grad_z, axis=0, keepdims=True)

        # Gradiente hacia capa anterior (oculta)
        grad_hidden = np.dot(grad_z, self.W.T)

        # Actualización
        self.W -= lr * grad_W_hidden
        self.W_skip -= lr * grad_W_skip
        self.b -= lr * grad_b

        return grad_hidden

