from layer import Layer
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def preprocess_data(x: np.ndarray, y: np.ndarray, limit: int): 
    # Shuffle the data
    x_train, _, y_train, _ = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)
    
    x_train = x_train [:limit]
    y_train = y_train [:limit]

    x_train = x_train / 255.
    y_train = np.array(tf.one_hot(y_train, depth=len(np.unique(y_train)), on_value=1, off_value=0))
    
    return x_train, y_train
    

class Network(): 
    def __init__(self, network: list[Layer]) :
        self.network = network
    
    def predict(self, input): 
        x = input
        x = np.expand_dims(x, axis=0)   
        for layer in self.network: 
            x = layer.forward(x)
        return x

    def train(self, X_train, y_train, loss_fn, loss_prime_fn, epochs =1000, learning_rate=0.015, verbose=True): 
        """_summary_

        Args:
            X_train (list[ndarray]): training example
            y_train (list[ndarray]): ground true labels
            loss_fn (function(y, output)): loss function
            loss_prime_fn (function(y, output)): loss prime function
            epochs (int): number of epochs. Defaults to 1000.
            learning_rate (float): learning rate. Defaults to 0.015.
            verbose (True): set true to print Loss. Defaults to True.
        """

        for i in range(epochs): 
            L = 0
            m = len(X_train)
            for x, y in zip(X_train, y_train): 
                y = np.expand_dims(y, axis=-1)
                
                # Forward
                ypred = self.predict(x)
                
                # Loss
                L += loss_fn(y, ypred)
                
                # Backward
                # Compute derivate of ypred with respect to y
                grad = loss_prime_fn(y, ypred)

                for layer in reversed(self.network): 
                    layer: Layer
                    grad = layer.backward(grad, learning_rate)
            L /= m                
            if verbose and i % 10 == 0: 
                print(f'Epoch {i}, loss: {L}')
                