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

def onehot(tensor, num_classes=10):
    return torch.eye(num_classes).cuda().index_select(dim=0, index=tensor) # One-hot encode 

def transform_data(data,target,use_gpu, num_classes=10):
    data, target = Variable(data), Variable(target)
    if use_gpu:
        data, target = data.cuda(), target.cuda()
    target = onehot(target, num_classes=num_classes)
    return data, target

def get_network(opts):
    if opts.dataset == "mnist":
        capsnet = CapsNet(reconstruction_type=opts.decoder, alpha = opts.alpha, routing_iterations = opts.routing_iterations)
    if opts.dataset == "small_norb":
        capsnet = CapsNet(reconstruction_type=opts.decoder, alpha = opts.alpha, imsize=28, num_classes=5, routing_iterations = opts.routing_iterations)
    if opts.dataset == "cifar10":
        capsnet = CapsNet(reconstruction_type=opts.decoder, alpha = opts.alpha,imsize=32, routing_iterations = opts.routing_iterations,primary_caps_gridsize=8,img_channels=3)
    if opts.use_gpu:
        capsnet.cuda()
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
    stats = Statistics(LOG_DIR)

    for epoch in range(opts.epochs):
        capsnet.train()
        for batch, (data, target) in tqdm(list(enumerate(train_loader)), ascii=True, desc="Epoch:{:3d}, ".format(epoch)):
            optimizer.zero_grad()
            data, target = transform_data(data, target, opts.use_gpu, num_classes=capsnet.num_classes)

            capsule_output, reconstructions, _ = capsnet(data, target)
            data = denormalize(data)
            loss, rec_loss = capsnet.loss(data, target, capsule_output, reconstructions)
            loss.backward()
            optimizer.step()
            
            stats.track_train(loss.data.item(), rec_loss.data.item())
            """Evaluate on test set"""
            if batch % opts.display_step == 0:
                capsnet.eval()

                for batch_id, (data, target) in enumerate(test_loader):
                    data, target = transform_data(data, target, opts.use_gpu, num_classes=capsnet.num_classes)
                    
                    capsule_output, reconstructions, predictions = capsnet(data)
                    data = denormalize(data)
                    loss, rec_loss = capsnet.loss(data, target, capsule_output, reconstructions)


                    stats.track_test(loss.data.item(),rec_loss.data.item(), target, predictions)

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
                        save_images(IMAGES_SAVE_DIR, filename, data, reconstructions)
                
                # Save model
                model_path = get_path(SAVE_DIR, "model{}.pt".format(epoch))
                torch.save(capsnet.state_dict(), model_path)
                capsnet.train()

if __name__ == '__main__':
    opts = create_options()
    main(opts)
