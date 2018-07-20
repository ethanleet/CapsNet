import torch
import torch.nn as nn
import numpy as np
import os

import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt 
# The squash function specified in Dynamic Routing Between Capsules
# x: input tensor 
def squash(x, dim=-1):
  norm_squared = (x ** 2).sum(dim, keepdim=True)
  part1 = norm_squared / (1 +  norm_squared)
  part2 = x / torch.sqrt(norm_squared+ 1e-16)

  output = part1 * part2 
  return output

def weights_init_xavier(m):
    classname = m.__class__.__name__
    ignore_modules = [
        "SmallNorbConvReconstructionModule",
        "ConvReconstructionModule",
        "ConvLayer"
    ]
    
    if classname.find('Conv') != -1 and classname not in ignore_modules:
        nn.init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname == 'ClassCapsules': 
        nn.init.xavier_normal_(m.W.data, gain=0.002)
        nn.init.xavier_normal_(m.bias.data, gain=0.002)
        
        
def initialize_weights(capsnet):
    capsnet.apply(weights_init_xavier)
    
def denormalize(image):
    image = image - image.min()
    image = image / image.max()
    return image
  
    
def get_path(SAVE_DIR, filename):
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    path = os.path.join(SAVE_DIR, filename)
    return path
    
def save_images(SAVE_DIR, filename, images, reconstructions, num_images = 100, imsize=28):
    if len(images) < num_images or len(reconstructions) < num_images:
        print("Not enough images to save.")
        return

    big_image = np.ones((imsize*10, imsize*20+1))
    images = denormalize(images).view(-1, imsize, imsize)
    reconstructions = denormalize(reconstructions).view(-1, imsize, imsize)
    images = images.data.cpu().numpy()
    reconstructions = reconstructions.data.cpu().numpy()
    for i in range(num_images):
        image = images[i]
        rec = reconstructions[i]
        j = i % 10
        i = i // 10
        big_image[i*imsize:(i+1)*imsize, j*imsize:(j+1)*imsize] = image
        j += 10
        big_image[i*imsize:(i+1)*imsize, j*imsize+1:(j+1)*imsize+1] = rec

    path = get_path(SAVE_DIR, filename)
    plt.imsave(path, big_image, cmap="gray")

def save_images_cifar10(SAVE_DIR, filename, images, reconstructions, num_images = 100):
    if len(images) < num_images or len(reconstructions) < num_images:
        print("Not enough images to save.")
        return

    big_image = np.ones((3,32*10, 32*20+1))
    #print('Images : ',big_image.T.shape,',',reconstructions.size())
    images = denormalize(images).view(-1, 3 ,32, 32)
    reconstructions = denormalize(reconstructions).view(-1, 3 ,32, 32)
    images = images.data.cpu().numpy()
    reconstructions = reconstructions.data.cpu().numpy()
    for i in range(num_images):
        image = images[i]
        rec = reconstructions[i]
        j = i % 10
        i = i // 10
        big_image[:,i*32:(i+1)*32, j*32:(j+1)*32] = image
        j += 10
        big_image[:,i*32:(i+1)*32, j*32+1:(j+1)*32+1] = rec

    path = get_path(SAVE_DIR, filename)
    plt.imsave(path, big_image.T)
