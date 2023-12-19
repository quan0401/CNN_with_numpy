import numpy as np
from numpy import ndarray
from initializer import Initializer
from scipy.signal import correlate2d
from layer import Layer
  
  
class Convlutional(Layer): 

    def __init__(self, input_shape: tuple, kernel_shape: tuple) -> None:
        """_summary_

        Args:
            input_shape (tuple, (input_depth, n, n)): output from the previous layer
            kernel_shape (tuple, (n_filtfers, input_depth, f, f)): kernel 
        """
        n = input_shape[1]
        self.n_filters, self.input_depth, f, f = kernel_shape
        
        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        
        self.output_shape = (self.n_filters, n - f + 1, n - f + 1)
        
        # He normal intialization
        self.kernels = Initializer.he_normal(self.kernel_shape)
        self.biases = np.zeros(self.output_shape)
        
    def forward(self, input) -> ndarray: 
        '''
        Arguments: 
            input (ndarray, (input_depth, n, n))
        Returns: 
            y (ndarray, (n_filters, n - f + 1, n - f + 1))
        '''
        self.input = input
        # initialize y with 0 then add biases == y copy biases
        self.y = np.copy(self.biases)
        for i in range(self.n_filters): 
            for j in range(self.input_depth): 
                self.y[i] += correlate2d(self.input[j], self.kernels[i, j], 'valid')
                
                
        return self.y
    
    def backward(self, dy: ndarray, learning_rate: float) -> ndarray: 
        '''
        Arguments: 
            dy (ndarray, (n_filters, n - f + 1, n - f + 1)): derivative of this layer forward output (y) with respect to Loss
            learning_rate (float)
        Returns: 
            dkernels (ndarray, (n_filters, n - f + 1, n - f + 1))
        '''
        # derivative of kernel respect to Loss
        dkernels = np.zeros(self.kernel_shape)
        dinput = np.zeros(self.input_shape)
        
        for i in range(self.n_filters): 
            for j in range(self.input_depth): 
                dkernels[i, j] = correlate2d(self.input[j], dy[i], 'valid')
                dinput[j] += correlate2d(dy[j], self.kernels[i, j], 'full')
        self.kernels = self.kernels - learning_rate * dkernels
        self.biases = self.biases - learning_rate * dy
        
        return dinput
    
    def get_output_shape(self):
        return self.output_shape
    
    
    
