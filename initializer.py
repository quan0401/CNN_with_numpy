import numpy as np

class Initializer:
    @staticmethod
    def he_normal(conv_shape: tuple):
        fan_in = conv_shape[1] * conv_shape[2] * conv_shape[3] if len(conv_shape) == 4 else conv_shape[0]  # Calculate fan_in
        stddev = np.sqrt(2.0 / fan_in)
        return np.random.normal(0, stddev, size=conv_shape)
    
    
    
    
  