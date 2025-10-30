import numpy as np

class ReLU:
    def __call__(self,input):
        return np.maximum(0,input)
    
    def gradient(self,z):
        return (z>0).astype(np.float32)
    
class Tanh:
    def __call__(self,input):
        return np.tanh(input)
    
    def gradient(self,z):
        return (1-np.tanh(z)**2)
    
class Sigmoid:
    def __call__(self,input):
        return 1.0/(1.0+np.exp(-input))
    
    def gradient(self,z):
        s=1.0/(1.0+np.exp(-z))
        return s*(1.0-s)

    