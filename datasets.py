import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

class CustomTensorDataset(Dataset):
  def __init__(self, X: torch.Tensor, y: torch.Tensor, transform=None, pool_mask=None):
    """
    Arguments:
      X: torch.Tensor -- data
      y: torch.Tensor -- labels
      transform: torchvision.transforms -- transforms for data
    """
    self.X = X
    self.y = y
    self.transform = transform
    self.pool_mask = pool_mask
    
  def __len__(self) -> int:
    return len(self.X)

  def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
    x, y = self.X[idx], self.y[idx]
    if self.transform:
      x = self.transform(x)
    return x, y

class ALDataset(Dataset):
  def __init__(self, X_train: torch.Tensor, y_train: torch.Tensor,
               initial_size: int,
               train_transform=None, pool_transform=None):
    """
    Arguments:
      X_train: torch.Tensor -- training data
      y_train: torch.Tensor -- training labels
      initial_size: int -- number of samples to start with
      train_transform: torchvision.transforms -- transforms for training data
      pool_transform: torchvision.transforms -- transforms for pool data
    """
    self.X_train = X_train
    self.y_train = y_train
    self.num_classes = len(torch.unique(y_train))
    self.N = len(X_train)
    self.initial_size = initial_size
    self.train_transform = train_transform
    self.pool_transform = pool_transform
    self.pool_mask = torch.ones(len(self.X_train), dtype=bool)
    initial_train_indices, _ = train_test_split(torch.arange(self.N), train_size=self.initial_size,
                                                stratify=self.y_train)
    self.pool_mask[initial_train_indices] = False
  
  def __len__(self) -> int:
    return len(self.X_train[~self.pool_mask])
  
  def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
    x, y = self.X_train[~self.pool_mask][idx], self.y_train[~self.pool_mask][idx]
    if self.train_transform:
      x = self.train_transform(x)
    return x, y
  
  def get_pool_data(self) -> Dataset:
    X_pool = self.X_train[self.pool_mask]
    y_pool = self.y_train[self.pool_mask]
    return CustomTensorDataset(X_pool, y_pool, transform=self.pool_transform, pool_mask=self.pool_mask)
    
    
def get_data(data_dir: str, dataset_size: float=1.0, dataset_name:str='mnist') -> \
  (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
  """
  Get X, y data for a given dataset.
  Arguments:
    data_dir: str -- directory to save the data
    dataset_size: float -- size of full dataset to use (decimal between 0 and 1)
    dataset_name: str -- name of the dataset
  Returns:
    X_train: torch.Tensor -- training data
    X_test: torch.Tensor -- test data
    y_train: torch.Tensor -- training labels
    y_test: torch.Tensor -- test labels
  """
  assert 0.0 < dataset_size <= 1.0, 'dataset_size must be between 0 and 1.'
  if dataset_name not in ['mnist']:
    raise ValueError(f'Unknown dataset {dataset_name}.')
  if dataset_name == 'mnist':
    trainset = datasets.EMNIST(data_dir, split='mnist', train=True, download=True)
    testset = datasets.EMNIST(data_dir, split='mnist', train=False, download=True)
  X_train, y_train = trainset.data.float(), trainset.targets.long()
  X_test, y_test = testset.data.float(), testset.targets.long()
  if dataset_size < 1.0:
    X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=dataset_size, stratify=y_train)
    X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=dataset_size, stratify=y_test)
  return X_train, X_test, y_train, y_test

def get_datasets(data_dir: str, dataset_size: float=1.0, dataset_name:str='mnist',
                 train_transform=None, test_transform=None,
                 initial_size: int=100) -> (ALDataset, CustomTensorDataset):
  """
  Returns active learning dataset (train + pool) and test dataset.
  Arguments:
    data_dir: str -- directory to save the data
    dataset_size: float -- size of full dataset to use (decimal between 0 and 1)
    dataset_name: str -- name of the dataset
    train_transform: torchvision.transforms -- transforms for training data
    test_transform: torchvision.transforms -- transforms for test data
    initial_size: int -- number of samples to start with
  Returns:
    ALset: ALDataset -- active learning dataset (train + pool)
    testset: CustomTensorDataset -- test dataset
  """
  X_train, X_test, y_train, y_test = get_data(data_dir, dataset_size, dataset_name)
  ALset = ALDataset(X_train, y_train, initial_size, train_transform, test_transform)
  testset = CustomTensorDataset(X_test, y_test, transform=test_transform)
  return ALset, testset