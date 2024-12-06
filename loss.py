import numpy as np
from models.nn.tensor import Tensor

## Fused Softmax Layer and Cross Entropy Loss (In Case of Single Label Classification)
class CELossWithLogits():


    def __call__(self,logits,targets):
        if not isinstance(logits,Tensor):
            raise ValueError("Logits must be a tensor")
        if not isinstance(targets,Tensor):
            raise ValueError("targets must be a tensor")
        if logits.data.shape[0] != targets.data.shape[0]:
            raise ValueError("Incompatible samples of logits and target")
        
        ## Compute Softmax Probablities
        probs = CELossWithLogits._softmax(logits.data)
        ## Compute Cross Entropy Loss
        data = np.asarray([CELossWithLogits._compute_log_loss(probs,targets.data)]).reshape(1,1)

        out = Tensor(data,(logits,),_op='celoss')

        def _backward():
            one_hot_target = np.zeros((logits.data.shape[0],logits.data.shape[1]))
            one_hot_target[np.arange(0,logits.data.shape[0]), targets.data] += 1
            logits.grad += (probs - one_hot_target)/logits.data.shape[0] * out.grad
        
        out._backward = _backward
        
        return out

    @staticmethod
    def _softmax(logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Stability improvement
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    @staticmethod
    def _compute_log_loss(probs, targets):
        N = targets.shape[0]  # Number of samples
        # Use the one-hot encoding method
        log_loss = -np.mean(np.log(probs[np.arange(N), targets]))
        return log_loss

class BCELossWithLogits():

    def __call__(self,logits,targets):
        if not isinstance(logits,Tensor):
            raise ValueError("Logits must be a tensor")
        if not isinstance(targets,Tensor):
            raise ValueError("targets must be a tensor")
        if logits.shape != targets.shape:
            raise ValueError("Incompatible samples of logits and target")
        
        probabilities = BCELossWithLogits._sigmoid(logits.data)
        bce = np.mean(-1*(targets.data * np.log(probabilities + 1e-15) + (1 - targets.data) * np.log(1 - probabilities + 1e-15)))
        out = Tensor(bce, _op='bceloss')
        out._prev = (logits,)

        def _backward():
            logits.grad = (probabilities.data - targets.data)/logits.shape[0]
        
        out._backward = _backward

        return out

    @staticmethod
    def _sigmoid(logits):
        """Compute the sigmoid of the input logits."""
        return 1 / (1 + np.exp(-logits))
        
        

## Mean Square Error Loss
class MSE():
    
    def __call__(self,predicted_values,true_values):
        if not isinstance(predicted_values,Tensor):
            raise ValueError(f"Logits must be a tensor Got {type(predicted_values)}")
        if not isinstance(true_values,Tensor):
            raise ValueError(f"targets must be a tensor Got {type(true_values)}")
        if predicted_values.shape != true_values.shape:
            raise ValueError("Incompatible shapes of predictions and true values")
    
        predicted_values = predicted_values
        true_values = true_values

        ## Compute MSE Loss
        residuals = predicted_values - true_values
        mse_per_sample = np.mean((residuals ** 2).data, axis=1)  # Average over predictions per sample
        data = np.mean(mse_per_sample).reshape(1, 1)  # Mean across samples

        out = Tensor(data,(predicted_values,),_op='mseloss')

        def _backward():
            N = true_values.shape[0]
            M = true_values.shape[1]
            predicted_values.grad = (2/(N*M)) * residuals.data * out.grad
        
        out._backward = _backward
    
        return out