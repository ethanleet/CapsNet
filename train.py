import os.path as path
import time
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import matplotlib.pyplot as plt

from constants import *
from data_loaders import *
from stats import *
from tools import *
from model import CapsNet
from options import create_options
from tqdm import tqdm

def get_alpha(epoch):
    # WARNING: Does not support alpha value saving when continuning training from a saved model
    if opts.anneal_alpha == "none":
        alpha = opts.alpha
    if opts.anneal_alpha == "1":
        alpha = opts.alpha * float(np.tanh(epoch/4 - np.pi) + 1) / 2
    if opts.anneal_alpha == "2":
        alpha = opts.alpha * float(np.tanh(epoch/8))
    return alpha

def onehot(tensor, num_classes=10):
    return torch.eye(num_classes).cuda().index_select(dim=0, index=tensor) # One-hot encode 

def transform_data(data,target,use_gpu, num_classes=10):
    data, target = Variable(data), Variable(target)
    if use_gpu:
        data, target = data.cuda(), target.cuda()
    target = onehot(target, num_classes=num_classes)
    return data, target

class GPUParallell(nn.DataParallel):
  
  def __init__(self, capsnet, device_ids):
    super(Test, self).__init__(capsnet, device_ids=device_ids)
    self.capsnet = capsnet
    self.num_classes = capsnet.num_classes
    
  def loss(self, images,labels, capsule_output,  reconstruction): 
    return self.capsnet.loss(images, labels, capsule_output, reconstruction)
  
  def forward(self, x, target=None):
    return self.capsnet(x, target)

def get_network(opts):
    if opts.dataset == "mnist":
        capsnet = CapsNet(reconstruction_type=opts.decoder,
                          routing_iterations = opts.routing_iterations,
                          batchnorm=opts.batch_norm,
                          loss=opts.loss_type,
                          leaky_routing=opts.leaky_routing)
    if opts.dataset == "small_norb":
        if opts.decoder == "conv":
            opts.decoder = "conv32"
        capsnet = CapsNet(reconstruction_type=opts.decoder,
                          imsize=32,
                          num_classes=5,
                          routing_iterations = opts.routing_iterations, 
                          primary_caps_gridsize=8,
                          num_primary_capsules=32,
                          batchnorm=opts.batch_norm,
                          loss = opts.loss_type,
                          leaky_routing=opts.leaky_routing)
    if opts.dataset == "cifar10":
        if opts.decoder == "conv":
            opts.decoder = "conv32"
        capsnet = CapsNet(reconstruction_type=opts.decoder,
                          imsize=32, 
                          routing_iterations = opts.routing_iterations,
                          primary_caps_gridsize=8,
                          img_channels=3, 
                          batchnorm=opts.batch_norm,
                          num_primary_capsules=32,
                          loss=opts.loss_type,
                          leaky_routing=opts.leaky_routing)
    if opts.use_gpu:
        capsnet.cuda()
    if opts.gpu_ids:
        capsnet = GPUParallell(capsnet, opts.gpu_ids)
        print("Training on GPU IDS:", opts.gpu_ids)
    return capsnet

def load_model(opts, capsnet): 
    model_path = path.join(SAVE_DIR, opts.filepath)
    if path.isfile(model_path):
        print("Saved model found")
        capsnet.load_state_dict(torch.load(model_path))
    else:
        print("Saved model not found; Model initialized.")
        initialize_weights(capsnet)
    

def get_dataset(opts):
    if opts.dataset == 'mnist':
        return load_mnist(opts.batch_size)
    if opts.dataset == 'small_norb':
        return load_small_norb(opts.batch_size)
    if opts.dataset == 'cifar10':
        return load_cifar10(opts.batch_size)
    raise ValueError("Dataset not supported:" + opts.dataset)
    

def main(opts):
    capsnet = get_network(opts)

    optimizer = torch.optim.Adam(capsnet.parameters(), lr=opts.learning_rate)

    """ Load saved model"""
    load_model(opts, capsnet)

    train_loader, test_loader = get_dataset(opts)
    stats = Statistics(LOG_DIR, opts.log_filepath)

    for epoch in range(opts.epochs):
        capsnet.train()
        
        # Annealing alpha
        alpha = get_alpha(epoch)

        for batch, (data, target) in tqdm(list(enumerate(train_loader)), ascii=True, desc="Epoch{:3d}".format(epoch)):
            optimizer.zero_grad()
            data, target = transform_data(data, target, opts.use_gpu, num_classes=capsnet.num_classes)

            capsule_output, reconstructions, _ = capsnet(data, target)
            predictions = torch.norm(capsule_output.squeeze(), dim=2)
            data = denormalize(data)
            loss, rec_loss, marg_loss = capsnet.loss(data, target, capsule_output, reconstructions, alpha)
            loss.backward()
            optimizer.step()
            
            stats.track_train(loss.data.item(), rec_loss.data.item(), marg_loss.data.item(), target, predictions)
        
        """Evaluate on test set"""
        capsnet.eval()
        for batch_id, (data, target) in tqdm(list(enumerate(test_loader)), ascii=True, desc="Test {:3d}".format(epoch)):
            data, target = transform_data(data, target, opts.use_gpu, num_classes=capsnet.num_classes)

            capsule_output, reconstructions, predictions = capsnet(data)
            data = denormalize(data)
            loss, rec_loss, marg_loss = capsnet.loss(data, target, capsule_output, reconstructions, alpha)


            stats.track_test(loss.data.item(),rec_loss.data.item(), marg_loss.data.item(), target, predictions)

        stats.save_stats(epoch)

        # Save reconstruction image from testing set
        if opts.save_images:
            data, target = iter(test_loader).next()
            data, _ = transform_data(data, target, opts.use_gpu)
            _, reconstructions, _ = capsnet(data)
            filename = "reconstruction_epoch_{}.png".format(epoch)
            if opts.dataset == 'cifar10':
                save_images_cifar10(IMAGES_SAVE_DIR, filename, data, reconstructions)
            else:
                save_images(IMAGES_SAVE_DIR, filename, data, reconstructions, imsize=capsnet.imsize)

        # Save model
        model_path = get_path(SAVE_DIR, "model{}.pt".format(epoch))
        torch.save(capsnet.state_dict(), model_path)
        capsnet.train()


if __name__ == '__main__':
    opts = create_options()
    main(opts)
