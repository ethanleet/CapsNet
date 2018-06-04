import torch.nn as nn
import torch.nn.functional as functional
from tools import squash
import torch
from torch.autograd import Variable
USE_GPU=True
# First Convolutional Layer
class ConvLayer(nn.Module):
  def __init__(self, 
               in_channels=1, 
               out_channels=256, 
               kernel_size=9):
    super(ConvLayer, self).__init__()
    
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
  def forward(self, x):
    output = self.conv(x)
    return output

class PrimaryCapules(nn.Module):
  
  def __init__(self, 
               num_capsules=32, 
               in_channels=256, 
               out_channels=8, 
               kernel_size=9,
               primary_caps_gridsize=6):

    super(PrimaryCapules, self).__init__()
    self.gridsize = primary_caps_gridsize
    self.capsules = nn.ModuleList([
      nn.Sequential(
      nn.Conv2d(in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=0),
          
      )
       for i in range(num_capsules)
    ])
  
  def forward(self, x):
    output = [caps(x) for caps in self.capsules]
    output = torch.stack(output, dim=1)
    output = output.view(x.size(0), 32*(self.gridsize)*(self.gridsize), -1)
    
    return squash(output)


class ClassCapsules(nn.Module):
  
  def __init__(self, 
               num_capsules=10,
               num_routes = 32*6*6,
               in_channels=8,
               out_channels=16,
               routing_iterations=3):
    super(ClassCapsules, self).__init__()
    
    self.in_channels = in_channels
    self.num_routes = num_routes
    self.num_capsules = num_capsules
    self.routing_iterations = routing_iterations
    
    self.W = nn.Parameter(torch.normal(mean = torch.zeros(1,
                                                          num_routes,
                                                          num_capsules,
                                                          out_channels,
                                                          in_channels), std=0.05))
    self.bias = nn.Parameter(torch.normal(mean = torch.zeros(1,1, num_capsules, out_channels,1), std=0.05))

  def forward(self, x):
    batch_size = x.size(0)
    x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
    
    W = torch.cat([self.W] * batch_size, dim=0)
    u_hat = torch.matmul(W, x)
    
    b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
    
    if USE_GPU:
      b_ij = b_ij.cuda()
    
    for it in range(self.routing_iterations):
      c_ij = functional.softmax(b_ij, dim=2) # Not sure if it should be dim=1

      c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

      s_j = (c_ij * u_hat).sum(dim=1, keepdim=True) + self.bias
      v_j = squash(s_j, dim=-2)

      if it < self.routing_iterations - 1: 
        uhatv_product = torch.matmul(u_hat.transpose(3,4),
                            torch.cat([v_j] * self.num_routes, dim=1))
        uhatv_product = uhatv_product.squeeze(4).mean(dim=0, keepdim=True)
        b_ij = b_ij + uhatv_product
      
    return v_j.squeeze(1)

class ReconstructionModule(nn.Module):
  def __init__(self, capsule_size=16, num_capsules=10, imsize=28,img_channel=1):
    super(ReconstructionModule, self).__init__()
    
    self.num_capsules = num_capsules
    self.capsule_size = capsule_size
    self.imsize = imsize
    self.img_channel = img_channel
    self.decoder = nn.Sequential(
          nn.Linear(capsule_size*num_capsules, 512),
          nn.ReLU(),
          nn.Linear(512, 1024),        
          nn.ReLU(),
          nn.Linear(1024, imsize*imsize*img_channel),
          nn.Sigmoid()
    )
        
  # TODO: remove data as parameter
  def forward(self, x, data, target=None):
    batch_size = x.size(0)
    if target is None:
      classes = torch.sqrt((x **2).sum(2))
      classes = functional.softmax(classes, dim=1)

      _, max_length_indices = classes.max(dim=1)
    else:
      max_length_indices = target.max(dim=1)[1].reshape(-1,1)
    masked = Variable(torch.eye(self.num_capsules))
    
    if USE_GPU:
      masked  = masked.cuda()
    masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)
    decoder_input = (x * masked[:, :, None, None]).view(batch_size, -1)

    reconstructions = self.decoder(decoder_input)
    reconstructions = reconstructions.view(-1, self.img_channel, self.imsize, self.imsize)
    return reconstructions, masked

class ConvReconstructionModule(nn.Module):
  def __init__(self, num_capsules=10, capsule_size=16, imsize=28,img_channels=1):
    
    super(ConvReconstructionModule, self).__init__()
    self.num_capsules = num_capsules
    self.capsule_size = capsule_size
    self.imsize = imsize
    self.img_channels = img_channels
    self.FC = nn.Sequential(
        nn.Linear(capsule_size * num_capsules, num_capsules * 6 * 6 ),
        nn.ReLU()
    )
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(in_channels=self.num_capsules, out_channels=32, kernel_size=9, stride=2),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=9, stride=1),  
      nn.BatchNorm2d(64), 
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=2, stride=1),
      nn.Sigmoid()
    )
    
  def forward(self, x, data, target=None):
    batch_size = x.size(0)
    if target is None:
      classes = torch.sqrt((x **2).sum(2))
      classes = functional.softmax(classes, dim=1)

      _, max_length_indices = classes.max(dim=1)
    else:
      max_length_indices = target.max(dim=1)[1].reshape(-1,1)
    masked = Variable(torch.eye(self.num_capsules))
    
    if USE_GPU:
      masked  = masked.cuda()
    masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)
    decoder_input = (x * masked[:, :, None, None]).view(batch_size, -1)
    decoder_input = self.FC(decoder_input)
    decoder_input = decoder_input.view(batch_size,self.num_capsules, 6, 6)
    reconstructions = self.decoder(decoder_input)
    reconstructions = reconstructions.view(-1, self.img_channels, self.imsize, self.imsize)
    
    return reconstructions, masked




class CapsNet(nn.Module):
  
  def __init__(self,
               alpha=0.0005, # Alpha from the loss function
               reconstruction_type = "FC",
               imsize=28,
               num_classes=10,
               routing_iterations=3,
               primary_caps_gridsize=6,
               img_channels = 1
              ):
    super(CapsNet, self).__init__()
    self.num_classes = num_classes
    self.conv_layer = ConvLayer(in_channels=img_channels)
    self.primary_capsules = PrimaryCapules(primary_caps_gridsize=primary_caps_gridsize)
    self.digit_caps = ClassCapsules(num_capsules=num_classes,num_routes=32*primary_caps_gridsize*primary_caps_gridsize, routing_iterations=routing_iterations)

    if reconstruction_type == "FC":
        self.decoder = ReconstructionModule(imsize=imsize, num_capsules=num_classes ,img_channel=img_channels)
    else:
        self.decoder = ConvReconstructionModule(num_capsules=num_classes,imsize=imsize,img_channels=img_channels)
    
    self.mse_loss = nn.MSELoss(reduce=False)
    self.alpha = alpha
  
  def forward(self, x, target=None):
    output = self.conv_layer(x)
    output = self.primary_capsules(output)
    output = self.digit_caps(output)
    reconstruction, masked = self.decoder(output, x, target)
    return output, reconstruction, masked
  
  def loss(self, images, labels, capsule_output,  reconstruction):
    marg_loss = self.margin_loss(capsule_output, labels)
    rec_loss = self.reconstruction_loss(images, reconstruction)
    total_loss = (marg_loss + self.alpha*rec_loss).mean()
    return total_loss, rec_loss.mean()
  
  def margin_loss(self, x, labels):
    batch_size = x.size(0)
    
    v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))
    
    left = functional.relu(0.9 - v_c).view(batch_size, -1) ** 2
    right = functional.relu(v_c - 0.1).view(batch_size, -1) ** 2

    loss = labels * left + 0.5 *(1-labels)*right
    loss = loss.sum(dim=1)
    return loss
  
  def reconstruction_loss(self, data, reconstructions):
    batch_size = reconstructions.size(0)
    loss = self.mse_loss(reconstructions.view(batch_size, -1),
                         data.view(batch_size, -1))
    loss = loss.sum(dim=1)
    return loss
