# basic imports
import numpy as np
# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# typing
from typing import Any, TypedDict
from numpy.typing import NDArray
EvalDict = TypedDict('EvalDict', {'loss': float, 'acc': float})
MetricsHist = TypedDict('MetricsHist', {'loss': list[float], 'acc': list[float]})
MetricsDict = TypedDict('MetricsDict', {'training': MetricsHist, 'test': MetricsHist})


class ModelWrapper():
    
    def __init__(self, model: nn.Sequential | nn.Module, optimizer: torch.optim.Optimizer,
                 loss: nn.modules.loss._Loss, device: str = 'cpu') -> None:
        '''
        This class wraps around a torch model and provides functions for training and evaluation. 
        
        Parameters
        ----------
        model :                             The network model.
        optimizer :                         The optimizer that will be used for training.
        loss :                              The loss function that will be used for training.
        device :                            The name of the device that the model will stored on (\'cpu\' by default).
        
        Returns
        ----------
        None
        '''
        self.model = model
        self.optimizer = optimizer
        self.criterion = loss
        self.device = torch.device(device)
        self.model.to(self.device)
        
    def predict_on_batch(self, batch: NDArray | torch.Tensor) -> NDArray:
        '''
        This function computes network predictions for a batch of input samples.
        
        Parameters
        ----------
        batch :                             The batch of input samples.
        
        Returns
        ----------
        predictions :                       A batch of network predictions.
        '''
        with torch.inference_mode():
            if type(batch) is np.ndarray:
                batch = torch.tensor(batch, device=self.device)
            else:
                assert type(batch) is torch.Tensor
                batch = batch.to(device=self.device)
            return self.model(batch).detach().cpu().numpy()
        
    def fit(self, data: DataLoader, epochs: int = 1, data_eval: None | DataLoader = None,
            evaluate: bool = False, verbose: bool = False, callbacks: list[callable] | None = None) -> MetricsDict:
        '''
        This function fits the model on a given data set.
        
        Parameters
        ----------
        data :                              The data set that the model will be fit on.
        epochs :                            The number of epochs that the model will be fit for.
        data_eval :                         An additional test data set that the model will be evaluated on.
        evaluate :                          A flag indicating whether the model should be evaluated after each epoch.
        verbose :                           A flag indicating whether the training progress should be printed to console.
        
        Returns
        ----------
        metrics :                           A dictionary containing the average loss and average accuracy for each training epoch.
        '''
        assert epochs > 0
        metrics: MetricsDict = {'training': {'loss': [], 'acc': []}, 'test': {'loss': [], 'acc': []}}
        for epoch in range(epochs):
            running_loss = 0.0
            for i, batch in enumerate(data):
                inputs, targets = batch
                if isinstance(inputs, tuple):
                    inputs = tuple(input_tensor.to(self.device) for input_tensor in inputs)
                else:
                    inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(*inputs) # Unpack tuple into img1 and img2
                loss = self.criterion(predictions, targets)
                loss.backward()
                self.optimizer.step()
                # print statistics
                running_loss += loss.item()
                if (i + 1) % 100 == 0 and verbose:
                    print('Epoch %d, Batch %d, Loss: %f' % (epoch + 1, i + 1, running_loss/100))
                    running_loss = 0.0
            if evaluate:
                # training set
                metrics_epoch = self.evaluate(data)
                metrics['training']['loss'].append(metrics_epoch['loss'])
                metrics['training']['acc'].append(metrics_epoch['acc'])
                # test set
                if data_eval is not None:
                    metrics_epoch = self.evaluate(data_eval)
                    metrics['test']['loss'].append(metrics_epoch['loss'])
                    metrics['test']['acc'].append(metrics_epoch['acc'])
            if callbacks:
                for callback in callbacks:
                    callback(self.model, epoch, metrics_epoch['loss'])

        return metrics
                    
    def evaluate(self, data: DataLoader) -> EvalDict:
        '''
        This function evaluates the model on a given data set.
        
        Parameters
        ----------
        data :                              The data set that the model will be evaluated on.
        
        Returns
        ----------
        metrics :                           A dictionary containing the average loss and average accuracy.
        '''
        metrics: MetricsHist = {'loss': [], 'acc': []}
        for batch in data:
            with torch.inference_mode():
                inputs, targets = batch
                if isinstance(inputs, tuple):
                    inputs = tuple(input_tensor.to(self.device) for input_tensor in inputs)
                else:
                    inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                predictions = self.model(*inputs)
                loss = self.criterion(predictions, targets)
                metrics['loss'].append(loss.item())
                if len(targets.shape) == 1 or len(targets.shape) == 2 and targets.shape[1] == 1:
                    metrics['acc'].append((predictions.argmax(axis=1) == targets).sum().detach().cpu()/targets.shape[0])
                else:
                    metrics['acc'].append(0.)
        
        return {'loss': float(np.mean(metrics['loss'])), 'acc': float(np.mean(metrics['acc']))}
