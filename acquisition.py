import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import CustomTensorDataset

from utils import H

class Acquirer:
  """
  Base class for acquisition functions.
  """
  def __init__(self, query_size: int, device: torch.device, processing_batch_size: int=128,
               reverse: bool=True):
    """
    Arguments:
      query_size: int -- number of samples to query
      device: torch.device -- device to use for computation
      reverse: bool -- whether to sort in ascending or descending order
      processing_batch_size: int -- batch size for processing the pool
    """
    self.query_size = query_size
    self.processing_batch_size = processing_batch_size
    self.device = device
    self.reverse = reverse
    
  @staticmethod
  def score(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Acquisition scoring function.
    
    Arguments:
      model: torch.nn.Module -- the NN
      x: torch.Tensor -- data to be scored
      
    Returns:
      torch.Tensor -- scores
    """
    return torch.zeros(x.shape[0])
  
  def score_pool(self, model: nn.Module, pool_data: CustomTensorDataset):
    """
    Scores the pool.
    
    Arguments:
      model: torch.nn.Module -- the NN
      pool_data: torch.utils.data.Dataset -- the pool dataset
      
    Returns:
      best_pool_idx: torch.Tensor -- indices of the best samples in the pool
    """
    pool_loader = DataLoader(pool_data, batch_size=self.processing_batch_size, 
                             pin_memory=True, shuffle=False)
    scores = torch.zeros(len(pool_data)).to(self.device)
    for idx, (data, _) in enumerate(pool_loader):
      data = data.to(self.device)
      start_idx = idx * self.processing_batch_size
      end_idx = start_idx + data.shape[0]
      scores[start_idx:end_idx] = self.score(model, data)
    best_local_idx = torch.argsort(scores, descending=self.reverse)[:self.query_size]
    best_idx = torch.arange(len(pool_data.pool_mask))[pool_data.pool_mask][best_local_idx]
    return best_idx
      
  def __call__(self, model: nn.Module, pool_data: CustomTensorDataset) -> torch.Tensor:
    """
    Acquisition function.
    
    Arguments:
      model: torch.nn.Module -- the NN
      pool_data: torch.utils.data.Dataset -- the pool dataset
      
    Returns:
      best_pool_idx: torch.Tensor -- indices of the best samples in the pool
    """
    return self.score_pool(model, pool_data)
  
class Random(Acquirer):
  """
  Random acquisition function.
  """
  def __init__(self, query_size: int, device: torch.device, processing_batch_size: int=128):
    super().__init__(query_size, device, processing_batch_size, reverse=False)
    
  @staticmethod
  def score(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return torch.rand(x.shape[0])
  
class Variance(Acquirer):
  """
  Variance acquisition function.
  """
  def __init__(self, query_size: int, device: torch.device, processing_batch_size: int=128):
    super().__init__(query_size, device, processing_batch_size, reverse=False)
    
  @staticmethod
  def score(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return torch.var(model(x), dim=1)
  
class Entropy(Acquirer):
  """
  Entropy acquisition function.
  """
  def __init__(self, query_size: int, device: torch.device, processing_batch_size: int=128):
    super().__init__(query_size, device, processing_batch_size, reverse=True)
    
  @staticmethod
  def score(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return H(model(x))
  
class Margin(Acquirer):
  """
  Margin acquisition function.
  """
  def __init__(self, query_size: int, device: torch.device, processing_batch_size: int=128):
    super().__init__(query_size, device, processing_batch_size, reverse=False)
    
  @staticmethod
  def score(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    sorted_scores, _ = torch.sort(model(x), dim=1, descending=True)
    return sorted_scores[:, 0] - sorted_scores[:, 1]
  
class VarRatio(Acquirer):
  """
  Variance ratio acquisition function.
  """
  def __init__(self, query_size: int, device: torch.device, processing_batch_size: int=128):
    super().__init__(query_size, device, processing_batch_size, reverse=True)
    
  @staticmethod
  def score(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return 1 - torch.max(model(x), dim=1)[0]
  
class BALD(Acquirer):
  """
  BALD acquisition function.
  """
  def __init__(self, query_size: int, device: torch.device, processing_batch_size: int=128):
    super().__init__(query_size, device, processing_batch_size, reverse=True)
  
  @staticmethod
  def score(model: nn.Module, x: torch.Tensor, k: int=100) -> torch.Tensor:
    Y = torch.stack([model(x) for _ in range(k)], dim=0)
    H_ = H(Y.mean(dim=0))
    EH = H(Y.swapaxes(1, 2)).mean(dim=0)
    return H_ - EH

class BatchBald(Acquirer):
  """
  BatchBALD acquisition function.
  """
  def __init__(self, query_size: int, device: torch.device, processing_batch_size: int=128):
    super().__init__(query_size, device, processing_batch_size, reverse=True)
  
  @staticmethod
  def score(model: nn.Module, x: torch.Tensor, k: int=100) -> torch.Tensor:
    raise NotImplementedError('BatchBALD is not implemented yet.')
  
def get_acquirer(name: str, query_size: int, device: torch.device, 
                 processing_batch_size: int=128) -> Acquirer:
  """
  Returns an acquisition function by name.
  
  Arguments:
    name: str -- name of the acquisition function
    query_size: int -- number of samples to query
    device: torch.device -- device to use for computation
    processing_batch_size: int -- batch size for processing the pool
    
  Returns:
    acquirer: Acquirer -- acquisition function
  """
  if name == 'random':
    return Random(query_size, device, processing_batch_size)
  elif name == 'variance':
    return Variance(query_size, device, processing_batch_size)
  elif name == 'entropy':
    return Entropy(query_size, device, processing_batch_size)
  elif name == 'margin':
    return Margin(query_size, device, processing_batch_size)
  elif name == 'var_ratio':
    return VarRatio(query_size, device, processing_batch_size)
  elif name == 'bald':
    return BALD(query_size, device, processing_batch_size)
  elif name == 'batch_bald':
    return BatchBald(query_size, device, processing_batch_size)
  else:
    raise ValueError(f'Unknown acquisition function: {name}')