import numpy as np
from models.nn.tensor import Tensor


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)
    
    def fit(self,X,y,
            X_val=None,
            y_val=None,
            X_test=None,
            y_test=None,
            batch_size=1,
            epoch=10,
            lr=0.005,
            perf_metric=None, ## Function to compute metric
            early_stop_patience = None,  ## No of Epoch before which if val loss dosent reduce, the training will be preempted
            enable_wandb_tracker=False,
            wandb_project_name="",
            wandb_experiment_name="",
            optimiser=None):
        from models.nn.optimiser import GradientDescent
        from models.nn.trainer import Trainer
        trainer = Trainer(
            model = self,
            train_data = (X,y),
            val_data = (X_val,y_val),
            test_data = (X_test,y_test),
            batch_size=batch_size,
            epoch=epoch,
            optimiser = GradientDescent(self.parameters(),lr=lr) if optimiser is None else optimiser,
            perf_metric=perf_metric,
            early_stop_patience=early_stop_patience,
            enable_wandb_tracker=enable_wandb_tracker,
            wandb_experiment_name=wandb_experiment_name,
            wandb_project_name=wandb_project_name
        )
        trainer.train()
    
    def parameters(self):
        return []


class Linear(Module):

    def __init__(self,n_in, n_out, non_lin=None,add_bias=True,tag='0'):
        self.non_lin = non_lin
        self.n_in = n_in
        self.n_out = n_out
        self.W = Tensor(self._xavier_initialization(n_in,n_out),_op=f'linear_weight_{tag}')
        self.add_bias = add_bias

        if self.add_bias:
            self.b = Tensor(np.zeros(n_out),_op=f'linear_bias_{tag}')

    '''
    Xavier Intialisation Helper function
    '''
    def _xavier_initialization(self,M, N):
        np.random.seed(42) ## Setting this for reproducibility
        limit = np.sqrt(6 / (M + N))  # Calculate the limit based on the input and output dimensions
        return np.random.uniform(-limit, limit, size=(M, N))  # Uniform distribution in the range [-limit, limit]
    
    '''
    Forward Call over one layer
    '''
    def __call__(self,x):
        out = x @ self.W
        if self.add_bias:
            out = out + self.b
        #print(out.data)
        ## Apply Non Linearity
        if self.non_lin == 'relu':
            return out.relu()
        elif self.non_lin == 'tanh':
            return out.tanh()
        elif self.non_lin == 'sigmoid':
            return out.sigmoid()
        else:
            return out
    
    def parameters(self):
        return [self.W,self.b]
        