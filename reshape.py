
from layer import Layer

class Reshape(Layer): 
    def __init__(self, input_shape: tuple, output_shape: tuple) -> None:
        """_summary_

        Args:
            input_shape (tuple): the shape of the input
            output_shape (tuple): the desired output shape
        """
        assert len(input_shape) > 1, f'input_shape must have at least 2 dimensions got {input_shape}'
        assert len(output_shape) > 1, f'output_shape must have at least 2 dimensions got {output_shape}'
        
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward(self, input):
        return input.reshape(self.output_shape)
    
    def backward(self, da, learning_rate):
        return da.reshape(self.input_shape)
    