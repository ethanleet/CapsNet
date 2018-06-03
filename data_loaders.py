from torchvision import datasets, transforms
import torch
from constants import * 
from smallNorb import smallNORB
def load_mnist(batch_size):
  dataset_transform = transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
           ])
  
  train_dataset = datasets.MNIST('../data', 
                               train=True, 
                               download=True, 
                               transform=dataset_transform)
  test_dataset = datasets.MNIST('../data', 
                                 train=False, 
                                 download=True, 
                                 transform=dataset_transform)


  train_loader = torch.utils.data.DataLoader(train_dataset, 
                                             batch_size=batch_size,
                                             shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, 
                                             batch_size=batch_size,
                                             shuffle=False)
  return train_loader, test_loader



def load_small_norb(batch_size):
    path = SMALL_NORB_PATH
    train_transform = transforms.Compose([
                          transforms.Resize(48),
                          transforms.RandomCrop(28),
                          transforms.ColorJitter(brightness=32./255, contrast=0.5),
                          transforms.ToTensor(),
                          transforms.Normalize((0.0,), (0.3081,))
                      ])
    test_transform = transforms.Compose([
                          transforms.Resize(48),
                          transforms.CenterCrop(28),
                          transforms.ToTensor(),
                          transforms.Normalize((0.,), (0.3081,))
                      ])
    
    train_dataset = smallNORB(path, train=True, download=True, transform=train_transform)
    test_dataset = smallNORB(path, train=False, transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                               shuffle=False)
    
    return train_loader, test_loader
