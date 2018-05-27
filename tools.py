import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt 
# The squash function specified in Dynamic Routing Between Capsules
# x: input tensor 
def squash(x):
  norm_squared = (x ** 2).sum(-1, keepdim=True)
  part1 = norm_squared / (1 +  norm_squared)
  part2 = x / torch.sqrt(norm_squared)

  output = part1 * part2 
  return output

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname != "ConvReconstructionModule":
        nn.init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
        
def initialize_weights(capsnet):
	capsnet.conv_layer.conv.apply(weights_init_xavier)
	capsnet.primary_capsules.apply(weights_init_xavier)
	capsnet.decoder.apply(weights_init_xavier)
	#nn.init.xavier_normal_(capsnet.digit_caps.W)
    
def denormalize(image):
    image = image - image.min()
    image = image / image.max()
    return image
  
    
def get_path(SAVE_DIR, filename):
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    path = os.path.join(SAVE_DIR, filename)
    return path
    
def save_images(SAVE_DIR, filename, images, reconstructions, num_images = 100):
    if len(images) < num_images or len(reconstructions) < num_images:
        print("Not enough images to save.")
        return

    big_image = np.ones((28*10, 28*20+1))
    images = denormalize(images).view(-1, 28, 28)
    reconstructions = denormalize(reconstructions).view(-1, 28, 28)
    images = images.data.cpu().numpy()
    reconstructions = reconstructions.data.cpu().numpy()
    for i in range(num_images):
        image = images[i]
        rec = reconstructions[i]
        j = i % 10
        i = i // 10
        big_image[i*28:(i+1)*28, j*28:(j+1)*28] = image
        j += 10
        big_image[i*28:(i+1)*28, j*28+1:(j+1)*28+1] = rec

    path = get_path(SAVE_DIR, filename)
    plt.imsave(path, big_image, cmap="gray")