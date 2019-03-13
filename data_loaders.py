import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from constants import * 
from smallNorb import SmallNORB

def build_dataloaders(batch_size, valid_size, train_dataset, valid_dataset, test_dataset):
  # Compute validation split
  train_size = len(train_dataset)
  indices = list(range(train_size))
  split = int(np.floor(valid_size * train_size))
  np.random.shuffle(indices)
  train_idx, valid_idx = indices[split:], indices[:split]
  train_sampler = SubsetRandomSampler(train_idx)
  valid_sampler = SubsetRandomSampler(valid_idx)
  
  # Create dataloaders
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             sampler=train_sampler)
  valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                             batch_size=batch_size,
                                             sampler=valid_sampler)
  test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
  return train_loader, valid_loader, test_loader

def load_mnist(batch_size, valid_size=0.1):
  train_transform = transforms.Compose([
               transforms.RandomAffine(0, translate=[0.08,0.08]),      
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
           ])
  valid_transform = transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
           ])
  test_transform = transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
           ])
  
  train_dataset = datasets.MNIST('../data', 
                               train=True, 
                               download=True, 
                               transform=train_transform)
  valid_dataset = datasets.MNIST('../data',
                               train=True,
                               download=True,
                               transform=valid_transform)
  test_dataset = datasets.MNIST('../data', 
                                 train=False, 
                                 download=True, 
                                 transform=test_transform)

  return build_dataloaders(batch_size, valid_size, train_dataset, valid_dataset, test_dataset)



def load_small_norb(batch_size):
    path = SMALL_NORB_PATH
    train_transform = transforms.Compose([
                          transforms.Resize(48),
                          transforms.RandomCrop(32),
                          transforms.ColorJitter(brightness=32./255, contrast=0.5),
                          transforms.ToTensor(),
                          transforms.Normalize((0.0,), (0.3081,))
                      ])
    valid_transform = transforms.Compose([
                          transforms.Resize(48),
                          transforms.CenterCrop(32),
                          transforms.ToTensor(),
                          transforms.Normalize((0.,), (0.3081,))
                      ])
    test_transform = transforms.Compose([
                          transforms.Resize(48),
                          transforms.CenterCrop(32),
                          transforms.ToTensor(),
                          transforms.Normalize((0.,), (0.3081,))
                      ])
    
    train_dataset = SmallNORB(path, train=True, download=True, transform=train_transform)
    valid_dataset = SmallNORB(path, train=True, download=True, transform=valid_transform)
    test_dataset = SmallNORB(path, train=False, transform=test_transform)
    
    return build_dataloaders(batch_size, valid_size, train_dataset, valid_dataset, test_dataset)

def load_cifar10(batch_size, valid_size=0.1):
    train_transform = transforms.Compose([
                transforms.ColorJitter(brightness=63./255, contrast=0.8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0,0,0), (0.5, 0.5, 0.5))
            ])
    valid_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0,0,0), (0.5, 0.5, 0.5))
            ])
    test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0,0,0), (0.5, 0.5, 0.5))
        ])
    train_dataset = datasets.CIFAR10('../data',
                                    train=True,
                                    download=True,
                                    transform=train_transform)
    valid_dataset = datasets.CIFAR10('../data',
                                    train=True,
                                    download=True,
                                    transform=valid_transform)
    test_dataset = datasets.CIFAR10('../data',
                                    train=False,
                                    download=True,
                                    transform=test_transform)
    
    return build_dataloaders(batch_size, valid_size, train_dataset, valid_dataset, test_dataset)
