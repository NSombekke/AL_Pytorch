import torch
import numpy as np
import random

def H(x: torch.Tensor, eps: float=1e-8):
  """
  Compute entropy of x.
  
  Arguments:
    x: torch.Tensor -- input tensor (probs in {0, 1})
    eps: float -- small number to avoid log(0)
  
  Returns:
    entropy: torch.Tensor -- entropy of x
  """
  return -torch.sum(x * torch.log(x + eps), dim=1)

def set_seed(seed: int):
  """
  Set random seed for all libraries.
  
  Arguments:
    seed: int -- random seed
  """
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)