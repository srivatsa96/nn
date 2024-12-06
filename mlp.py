import numpy as np
from enum import Enum
import pickle

from models.nn.layers import Module, Linear


class MLP(Module):

    class TaskType(Enum):
        SINGLE_LABEL_CLASSIFICATION = 1
        MULTI_LABEL_CLASSIFICATION = 2
        REGRESSION = 3

    def __init__(self,n_in,n_out,n_layer=[],non_lin='relu',loss_fn=None,task_type=TaskType.SINGLE_LABEL_CLASSIFICATION):
        # if len(n_layer) == 0:
        #     raise ValueError('MLP should have atleast one layer')
        
        self.n_layer = [n_in] + n_layer
        self.total_layers = len(self.n_layer)
        self.module_list = [Linear(self.n_layer[idx],self.n_layer[idx+1],non_lin=non_lin,tag=f'{idx+1}') for idx in range(0,len(self.n_layer)-1)] 
        self.module_list += [Linear(self.n_layer[-1],n_out,tag=f'{len(self.n_layer)}')]
        self.loss_fn = loss_fn
        self.task_type = task_type
        self.non_lin = non_lin

    def __call__(self,x,target=None):
        for idx, module in enumerate(self.module_list):
            x = module(x)
        loss = None
        if target != None:
            loss = self.loss_fn(x,target)
        return x, loss
    
    def predict(self,x,prob_thresh=0.5):
        if self.task_type == MLP.TaskType.SINGLE_LABEL_CLASSIFICATION:
            pred,_ = self(x)
            return np.argmax(pred.data,axis=1)
        elif self.task_type == MLP.TaskType.MULTI_LABEL_CLASSIFICATION:
            pred,_ = self(x)
            return (pred.data > prob_thresh).astype(int)
        elif self.task_type == MLP.TaskType.REGRESSION:
            pred,_ = self(x)
            return pred.data


    def parameters(self):
        params = []
        for module in self.module_list:
            params += module.parameters()
        return params

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {file_path}")

    @staticmethod
    def load_model_from_checkpoint(file_path):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {file_path}")
        return model 