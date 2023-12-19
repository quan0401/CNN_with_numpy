class Layer(): 
    def forward(self, input): 
        # This is method meant to be overwritten be subclass
        pass
    def backward(self, da, learning_rate): 
        # This is method meant to be overwritten be subclas
        pass
    
    def get_output_shape(self): 
        pass
    
    