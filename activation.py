import numpy as np
from layer import Layer
class Activation(Layer): 
    def __init__(self, activation, activation_prime) -> None:
        
        self.activation = activation
        self.activation_prime = activation_prime
        
    def forward(self, input): 
        self.input = input  
        return self.activation(self.input)
    
    
    def backward(self, da, learning_rate): 
        '''
        da (ndarray): gradient of this activation output with respect to Loss
        '''
        # Multiply elementwise 
        dz = da * self.activation_prime(self.input)
        return dz