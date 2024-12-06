import numpy as np
from enum import Enum
from models.nn.layers import Module, Linear

class Autoencoder(Module):

    def __init__(self, n_in, n_hidden_layers, non_lin='relu', loss_fn=None):
        self.n_out = n_in  
        self.hidden_sizes = [n_in] + n_hidden_layers
        self.encoder_layers = [Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1], non_lin=non_lin, tag=f'encoder_{i + 1}') for i in range(len(n_hidden_layers))]
        # for i in range(len(n_hidden_layers)):
        #     print(self.hidden_sizes[-(i + 1)], self.hidden_sizes[-(i + 2)])

        self.decoder_layers = [Linear(self.hidden_sizes[-(i + 1)], self.hidden_sizes[-(i + 2)], non_lin=non_lin, tag=f'decoder_{i + 1}') for i in range(len(n_hidden_layers))]
        self.loss_fn = loss_fn

    def __call__(self, input, target=None): ## target added for Compatability with Trainer
        target = input
        x = input
        for layer in self.encoder_layers:
            x = layer(x)
        encoded = x
        reconstructed = encoded
        for layer in self.decoder_layers:
            reconstructed = layer(reconstructed)
        # print(reconstructed.shape)
        
        loss = None
        if self.loss_fn is not None:
            loss = self.loss_fn(reconstructed, target)
        
        return (encoded, reconstructed), loss

    def predict(self, x):
        (encoded, reconstructed), _ = self(x)
        return reconstructed.data, encoded.data 
    
    def get_latent(self, x):
        encoded, _, _ = self(x)
        return encoded.data 
    
    def parameters(self):
        params = []
        for layer in self.encoder_layers:
            params += layer.parameters()
        for layer in self.decoder_layers:
            params += layer.parameters()
        return params
