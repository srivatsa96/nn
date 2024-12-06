import numpy as np 

class Optimiser:

    def __init__(self,parameters,lr=0.001):
        self.parameters = parameters
        self.lr = lr 
    
    def step(self):
        raise NotImplementedError

class GradientDescent(Optimiser):

    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad

class Adam(Optimiser):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {id(param): 0.0 for param in parameters}  
        self.v = {id(param): 0.0 for param in parameters}
        self.t = 0  # Time step

    def step(self):
        self.t += 1
        for param in self.parameters:
            param_id = id(param)
            self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * param.grad
            self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (param.grad ** 2)
            m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

