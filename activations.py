import numpy as np
from activation import Activation

class Sigmoid(Activation): 
    def __init__(self) -> None:
        
        def sigmoid(x: np.ndarray): 
            a = 1 / (1 + np.exp(-x))
            
            return a
        
        def sigmoid_prime(x): 
            a = sigmoid(x)
            return a * (1-a)
        
        super().__init__(sigmoid, sigmoid_prime)
        
        
class Relu(Activation): 
    def __init__(self) -> None:
        def relu(x): 
            a = np.maximum(0, x)

            return a
        
        def relu_prime(x): 
            da = np.zeros(x.shape)
            da[x > 0] = 1
            return da
            
        
        super().__init__(relu, relu_prime)


class Softmax(Activation): 
    def __init__(self) -> None:
        def softmax(x): 
            exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # for numerical stability
            a = exp_x / np.sum(exp_x, axis=0, keepdims=True)

            return a

        def softmax_prime(x): 
            
            a = softmax(x)
            da = a * (1 - a)
            # print('softmax x', x.shape)
            # print('softmax a', a.shape)
            # print('softmax da', da.shape)
            return da
        super().__init__(softmax, softmax_prime)