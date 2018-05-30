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
from model_cifar10 import CapsNet
from options import create_options
from tqdm import tqdm

def onehot(tensor):
    return torch.eye(10).cuda().index_select(dim=0, index=tensor) # One-hot encode 

def transform_data(data,target,use_gpu):
    data, target = Variable(data), Variable(target)
    if use_gpu:
        data, target = data.cuda(), target.cuda()
    target = onehot(target)
    return data, target


def main(opts):
    capsnet = CapsNet(reconstruction_type=opts.decoder, alpha = opts.alpha)
    if opts.use_gpu:
        capsnet.cuda()
    optimizer = torch.optim.Adam(capsnet.parameters(), lr=opts.learning_rate)

    """ Load saved model"""
    model_path = path.join(SAVE_DIR, opts.filepath)
    if path.isfile(model_path) and opts.load_saved:
        print("Saved model found")
        capsnet.load_state_dict(torch.load(model_path))
    else:
        print("Saved model not found; Model initialized.")
        initialize_weights(capsnet)

    train_loader, test_loader = load_cifar10(opts.batch_size)
    stats = Statistics(LOG_DIR)

    for epoch in range(opts.epochs):
        capsnet.train()
        for batch, (data, target) in tqdm(list(enumerate(train_loader)), ascii=True, desc="Epoch:{:3d}, ".format(epoch)):
            optimizer.zero_grad()
            data, target = transform_data(data, target, opts.use_gpu)
            
            capsule_output, reconstructions, _ = capsnet(data, target)
            #print('Capsule output ',capsule_output.size(),',',reconstructions.size())
            data = denormalize(data)
            loss, rec_loss = capsnet.loss(data, target, capsule_output, reconstructions)
            loss.backward()
            optimizer.step()
            
            stats.track_train(loss.data.item(), rec_loss.data.item())
            """Evaluate on test set"""
            if batch % opts.display_step == 0:
                capsnet.eval()

                for batch_id, (data, target) in enumerate(test_loader):
                    data, target = transform_data(data, target, opts.use_gpu)
                    
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
                    save_images(IMAGES_SAVE_DIR, filename, data, reconstructions)
                
                # Save model
                model_path = get_path(SAVE_DIR, "model{}.pt".format(epoch))
                torch.save(capsnet.state_dict(), model_path)
                capsnet.train()

if __name__ == '__main__':
    opts = create_options()
    main(opts)
