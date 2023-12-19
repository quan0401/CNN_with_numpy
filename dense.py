import numpy as  np
from layer import Layer
from initializer import Initializer

class Dense(Layer): 
    def __init__(self, n_units, n_f) -> None:
        """_summary_

        Args:
            n_units (int): number of layer unit
            n_f (n_f): number of previous a output (or number of features)
        """
        self.n_units = n_units
        self.n_f = n_f
        self.W = Initializer.he_normal((n_units, n_f))
        self.b = np.zeros((n_units, 1))

    def forward(self, input: np.ndarray): 
        """_summary_
        Args:
            input (ndarray, (n_f, m)): previous layer input
        """
        self.input = input
        
        assert self.input.shape[0] == self.n_f, f'Invalid input shape {self.input.shape[0]} != {self.n_f}'
        z = np.dot(self.W, input) + self.b
        return z
    
    def backward(self, dz: np.ndarray, learning_rate):
        """_summary_

        Args:
            dz (ndarray, (n_f, m)): derivative of layer output with respect to Loss
            learning_rate (float): learning rate
        """
        
        dW = np.dot(dz, self.input.T)
        # Gradient of input
        da_prev = np.dot(self.W.T, dz)
        
        
        # Sum all over all examples
        db = dz.sum(axis=1, keepdims=True)
        
        assert dW.shape == self.W.shape, 'dW and W should have shape (n_units, n_f)'
        assert db.shape == self.b.shape, 'db and b should have shape (n_units, 1)'
        assert da_prev.shape[0] == self.n_f, 'wrong shape for da_prev'
        
        self.W = self.W - learning_rate*dW
        self.b = self.b - learning_rate*db
        return da_prev