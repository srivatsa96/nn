import numpy as np
import wandb

from models.nn.tensor import Tensor

class Trainer:

    def __init__(self,
                 model,
                 train_data, ## Tuple of Feature Array (B X N), Label Array  (B X 1)
                 optimiser,
                 val_data=None, ## Tuple of Feature Array (B X N), Label Array  (B X 1)
                 test_data=None, ## Tuple of Feature Array (B X N), Label Array  (B X 1)
                 perf_metric=None, ## Function to compute metric
                 batch_size=None,
                 epoch=5,
                 early_stop_patience = None,  ## No of Epoch before which if val loss dosent reduce, the training will be preempted
                 enable_wandb_tracker=False,
                 wandb_sweep_enabled=False,
                 wandb_project_name="",
                 wandb_experiment_name=""):
        self.model = model 
        self.train_data = train_data
        self.val_data = val_data if val_data is not None else self.train_data
        self.test_data = test_data
        self.perf_metric = perf_metric if isinstance(perf_metric, list) else ( [perf_metric] if perf_metric is not None else None)
        self.batch_size = batch_size
        self.epoch = epoch
        self.early_stop_patience = early_stop_patience
        self.optimiser = optimiser
        self.enable_wandb_tracker = enable_wandb_tracker
        self.wandb_sweep_enabled = wandb_sweep_enabled
        if self.enable_wandb_tracker and not self.wandb_sweep_enabled:
            wandb.init(
                # set the wandb project where this run will be logged
                project=wandb_project_name,
                name=wandb_experiment_name,
                # track hyperparameters and run metadata
                config={
                "learning_rate": self.optimiser.lr,
                "architecture": self.model.__class__.__name__,
                "epochs": self.epoch,
                "batch_size": self.batch_size,
                "early_stop_patience": self.early_stop_patience,
                "perf_metric": [ metric.__name__ for metric in self.perf_metric] if self.perf_metric is not None else None,
                **self.model.__dict__
                }
            )
    
    def train(self):

        ## Batch Gradient Descent
        if self.batch_size == None:
            self.batch_size = len(self.train_data[0].shape[0])
        
        if not self.batch_size >= 1:
            raise ValueError('Batch Size should be atleast or greater than 1')
        
        _,val_loss = self.model(Tensor(self.val_data[0]),Tensor(self.val_data[1]))
        print(f'Val Loss before training {val_loss.item():.4f}')

        ## ES Params
        min_val_loss = np.inf
        patience_count = 0
        min_loss_epoch = 0

        for idx in range(0,self.epoch):
            acc_train_loss = 0.0
            # One Batch at at time
            steps = 0
            for x, y in self._data_generator(self.train_data[0],self.train_data[1],self.batch_size):
                x,y = Tensor(x), Tensor(y) ## Wrap Dataset as Tensor
                self.model.zero_grad()
                _, loss = self.model(x,y)
                acc_train_loss += loss.item()
                loss.backward()
                self.optimiser.step()
                steps += 1
            
            avg_train_loss = acc_train_loss/steps
            _,val_loss = self.model(Tensor(self.val_data[0]),Tensor(self.val_data[1]))

            ## Early Stop Algorithm
            patience_count += 1
            if min_val_loss > val_loss.item():
                min_val_loss = val_loss.item()
                patience_count = 0
                min_loss_epoch = idx
            if self.early_stop_patience is not None and self.early_stop_patience < patience_count:
                print(f'Early stopping as val loss didnt improve till paitence count {self.early_stop_patience}.')
                break
            

            # Calculate validation performance metrics
            val_perf_metrics = {}
            if self.perf_metric is not None:
                for metric in self.perf_metric:
                    val_perf_metrics[metric.__name__] = metric(self.val_data[1], self.model.predict(Tensor(self.val_data[0])))

            # Print the metrics
            perf_metrics_str = " ".join([f"Validation {name}: {val_perf_metrics[name]:.4f}" for name in val_perf_metrics])
            print(f'Epoch {idx}/{self.epoch} - Train Loss: {avg_train_loss.item():.4f} Val Loss: {val_loss.item():.4f} {perf_metrics_str}')

            # Log the metrics to wandb
            log_dict = {'train/loss': avg_train_loss.item(), 'val/loss': val_loss.item()}
            log_dict.update({f'val/{name}': val_perf_metrics[name] for name in val_perf_metrics})
            if self.enable_wandb_tracker:
                wandb.log(log_dict)

        print(f'Training completed. Min Val Loss {min_val_loss:.4f} @ Epoch {min_loss_epoch}')

        if self.test_data != None:
            ## Compute Test Measures
            _,test_loss = self.model(Tensor(self.test_data[0]),Tensor(self.test_data[1]))
            # Calculate test performance metrics
            test_perf_metrics = {}
            if self.perf_metric is not None:
                for metric in self.perf_metric:
                    test_perf_metrics[metric.__name__] = metric(self.test_data[1], self.model.predict(Tensor(self.test_data[0])))

            # Log the metrics to wandb
            log_dict = {'test/loss': test_loss.item()}
            log_dict.update({f'test/{name}': test_perf_metrics[name] for name in test_perf_metrics})
            if self.enable_wandb_tracker:
                wandb.log(log_dict)

            # Print the test results
            perf_metrics_str = " ".join([f"Test {name}: {test_perf_metrics[name]:.4f}" for name in test_perf_metrics])
            print(f'Test Result - Test Loss: {test_loss.item():.4f} {perf_metrics_str}')
        

    def _data_generator(self, x, y, batch_size):
        np.random.seed(42) ## For Reproducibility
        # Ensure the input arrays have the same number of samples
        assert x.shape[0] == y.shape[0], "x and y must have the same number of samples."

        # Create an array of indices and shuffle it
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)

        # Loop through the indices in batches
        for start_idx in range(0, len(indices), batch_size):
            end_idx = min(start_idx + batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            # Yield the batch of data
            yield x[batch_indices], y[batch_indices]
            
            
        