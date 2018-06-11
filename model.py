import torch.nn as nn
import torch.nn.functional as functional
from tools import squash
import torch
from torch.autograd import Variable
USE_GPU=True

def routing_algorithm(x, weight, bias, routing_iterations):
    """
    x: [batch_size, num_capsules_in, capsule_dim]
    weight: [1,num_capsules_in,num_capsules_out,out_channels,in_channels]
    bias: [1,1, num_capsules_out, out_channels]
    """
    num_capsules_in = x.shape[1]
    num_capsules_out = weight.shape[2]
    batch_size = x.size(0)
    
    x = x.unsqueeze(2).unsqueeze(4)

    #[batch_size, 32*6*6, 10, 16]
    u_hat = torch.matmul(weight, x).squeeze()

    b_ij = Variable(x.new(batch_size, num_capsules_in, num_capsules_out, 1).zero_())


    for it in range(routing_iterations):
      c_ij = functional.softmax(b_ij, dim=2)

      # [batch_size, 1, num_classes, capsule_size]
      s_j = (c_ij * u_hat).sum(dim=1, keepdim=True) + bias
      # [batch_size, 1, num_capsules, out_channels]
      v_j = squash(s_j, dim=-1)
      
      if it < routing_iterations - 1: 
        # [batch-size, 32*6*6, 10, 1]
        delta = (u_hat * v_j).sum(dim=-1, keepdim=True)
        b_ij = b_ij + delta
    
    return v_j.squeeze()

# First Convolutional Layer
class ConvLayer(nn.Module):
  def __init__(self, 
               in_channels=1, 
               out_channels=256, 
               kernel_size=9,
               batchnorm=False):
    super(ConvLayer, self).__init__()
    
    if batchnorm:
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    else:
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1),
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
               primary_caps_gridsize=6,
               batchnorm=False):

    super(PrimaryCapules, self).__init__()
    self.gridsize = primary_caps_gridsize
    self.num_capsules = num_capsules
    if batchnorm:
        self.capsules = nn.ModuleList([
          nn.Sequential(
          nn.Conv2d(in_channels=in_channels,
                    out_channels=num_capsules,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=0),
          nn.BatchNorm2d(num_capsules)
          )
           for i in range(out_channels)
        ])
    else:
        self.capsules = nn.ModuleList([
          nn.Sequential(
          nn.Conv2d(in_channels=in_channels,
                    out_channels=num_capsules,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=0),

          )
           for i in range(out_channels)
        ])
  
  def forward(self, x):
    output = [caps(x) for caps in self.capsules]
    output = torch.stack(output, dim=1)
    output = output.view(x.size(0), self.num_capsules*(self.gridsize)*(self.gridsize), -1)
    
    return squash(output)


class ClassCapsules(nn.Module):
  
  def __init__(self, 
               num_capsules=10,
               num_routes = 32*6*6,
               in_channels=8,
               out_channels=16,
               routing_iterations=3,
               leaky=False):
    super(ClassCapsules, self).__init__()
    

    self.in_channels = in_channels
    self.num_routes = num_routes
    self.num_capsules = num_capsules
    self.routing_iterations = routing_iterations
    
    self.W = nn.Parameter(torch.rand(1,num_routes,num_capsules,out_channels,in_channels))
    self.bias = nn.Parameter(torch.rand(1,1, num_capsules, out_channels))


  # [batch_size, 10, 16, 1]
  def forward(self, x):
    v_j = routing_algorithm(x, self.W, self.bias, self.routing_iterations)
    return v_j.unsqueeze(-1)


class ReconstructionModule(nn.Module):
  def __init__(self, capsule_size=16, num_capsules=10, imsize=28,img_channel=1, batchnorm=False):
    super(ReconstructionModule, self).__init__()
    
    self.num_capsules = num_capsules
    self.capsule_size = capsule_size
    self.imsize = imsize
    self.img_channel = img_channel
    if batchnorm:
        self.decoder = nn.Sequential(
              nn.Linear(capsule_size*num_capsules, 512),
              nn.BatchNorm1d(512),
              nn.ReLU(),
              nn.Linear(512, 1024),        
              nn.BatchNorm1d(1024),
              nn.ReLU(),
              nn.Linear(1024, imsize*imsize*img_channel),
              nn.Sigmoid()
        )
    else:
        self.decoder = nn.Sequential(
              nn.Linear(capsule_size*num_capsules, 512),
              nn.ReLU(),
              nn.Linear(512, 1024),        
              nn.ReLU(),
              nn.Linear(1024, imsize*imsize*img_channel),
              nn.Sigmoid()
        )
        
  def forward(self, x, target=None):
    batch_size = x.size(0)
    if target is None:
      classes = torch.norm(x, dim=2)
      max_length_indices = classes.max(dim=1)[1].squeeze()
    else:
      max_length_indices = target.max(dim=1)[1]
    
    masked = Variable(x.new_tensor(torch.eye(self.num_capsules)))
    
    masked = masked.index_select(dim=0, index=max_length_indices.data)
    decoder_input = (x * masked[:, :, None, None]).view(batch_size, -1)

    reconstructions = self.decoder(decoder_input)
    reconstructions = reconstructions.view(-1, self.img_channel, self.imsize, self.imsize)
    return reconstructions, masked

class ConvReconstructionModule(nn.Module):
  def __init__(self, num_capsules=10, capsule_size=16, imsize=28,img_channels=1, batchnorm=False):
    super(ConvReconstructionModule, self).__init__()
    self.num_capsules = num_capsules
    self.capsule_size = capsule_size
    self.imsize = imsize
    self.img_channels = img_channels
    self.grid_size = 6
    if batchnorm:
      self.FC = nn.Sequential(
        nn.Linear(capsule_size * num_capsules, num_capsules * (self.grid_size)**2 ),
        nn.BatchNorm1d(num_capsules * self.grid_size**2),
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
    else:
        self.FC = nn.Sequential(
            nn.Linear(capsule_size * num_capsules, num_capsules *(self.grid_size**2) ),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
          nn.ConvTranspose2d(in_channels=self.num_capsules, out_channels=32, kernel_size=9, stride=2),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=9, stride=1),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=2, stride=1),
          nn.Sigmoid()
        )
    
  def forward(self, x, target=None):
    batch_size = x.size(0)
    if target is None:
      classes = torch.norm(x, dim=2)
      max_length_indices = classes.max(dim=1)[1].squeeze()
    else:
      max_length_indices = target.max(dim=1)[1]
    
    masked = Variable(x.new_tensor(torch.eye(self.num_capsules)))
    masked = masked.index_select(dim=0, index=max_length_indices.data)

    decoder_input = (x * masked[:, :, None, None]).view(batch_size, -1)
    decoder_input = self.FC(decoder_input)
    decoder_input = decoder_input.view(batch_size,self.num_capsules, self.grid_size, self.grid_size)
    reconstructions = self.decoder(decoder_input)
    reconstructions = reconstructions.view(-1, self.img_channels, self.imsize, self.imsize)
    
    return reconstructions, masked




class SmallNorbConvReconstructionModule(nn.Module):
  def __init__(self, num_capsules=10, capsule_size=16, imsize=28,img_channels=1, batchnorm=False):
    super(SmallNorbConvReconstructionModule, self).__init__()
    self.num_capsules = num_capsules
    self.capsule_size = capsule_size
    self.imsize = imsize
    self.img_channels = img_channels
    
    self.grid_size = 4
    
    if batchnorm:
      self.FC = nn.Sequential(
            nn.Linear(capsule_size * num_capsules, num_capsules *self.grid_size*self.grid_size),
            nn.BatchNorm1d(num_capsules * self.grid_size**2),
            nn.ReLU()
        )
      self.decoder = nn.Sequential(
          nn.ConvTranspose2d(in_channels=num_capsules, out_channels=32, kernel_size=9, stride=2),
          nn.BatchNorm2d(32),            
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=9, stride=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=9, stride=1),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=128, out_channels=img_channels, kernel_size=2, stride=1),
          nn.Sigmoid()
        )
    else:
        self.FC = nn.Sequential(
            nn.Linear(capsule_size * num_capsules, num_capsules *(self.grid_size**2) ),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
          nn.ConvTranspose2d(in_channels=num_capsules, out_channels=32, kernel_size=9, stride=2),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=9, stride=1),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=9, stride=1),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=128, out_channels=img_channels, kernel_size=2, stride=1),
          nn.Sigmoid()
        )
    
  def forward(self, x, target=None):
    batch_size = x.size(0)
    if target is None:
      classes = torch.norm(x, dim=2)
      max_length_indices = classes.max(dim=1)[1].squeeze()
    else:
      max_length_indices = target.max(dim=1)[1]
    masked = Variable(x.new_tensor(torch.eye(self.num_capsules)))
    masked = masked.index_select(dim=0, index=max_length_indices.data)

    decoder_input = (x * masked[:, :, None, None]).view(batch_size, -1)
    decoder_input = self.FC(decoder_input)
    decoder_input = decoder_input.view(batch_size,self.num_capsules, self.grid_size, self.grid_size)
    reconstructions = self.decoder(decoder_input)
    reconstructions = reconstructions.view(-1, self.img_channels, self.imsize, self.imsize)
    
    return reconstructions, masked




class CapsNet(nn.Module):
  
  def __init__(self,
               reconstruction_type = "FC",
               imsize=28,
               num_classes=10,
               routing_iterations=3,
               primary_caps_gridsize=6,
               img_channels = 1,
               batchnorm = False,
               loss = "L2",
               num_primary_capsules=32,
               leaky_routing = False
              ):
    super(CapsNet, self).__init__()
    self.num_classes = num_classes
    if leaky_routing:
        num_classes += 1
        self.num_classes += 1
        
    self.imsize=imsize
    self.conv_layer = ConvLayer(in_channels=img_channels, batchnorm=batchnorm)
    self.leaky_routing = leaky_routing

    self.primary_capsules = PrimaryCapules(primary_caps_gridsize=primary_caps_gridsize,
                                           batchnorm=batchnorm,
                                           num_capsules = num_primary_capsules)
    
    self.digit_caps = ClassCapsules(num_capsules=num_classes,
                                    num_routes=num_primary_capsules*primary_caps_gridsize*primary_caps_gridsize,
                                    routing_iterations=routing_iterations,
                                    leaky=leaky_routing)

    if reconstruction_type == "FC":
        self.decoder = ReconstructionModule(imsize=imsize,
                                            num_capsules=num_classes,
                                            img_channel=img_channels, 
                                            batchnorm=batchnorm)
    elif reconstruction_type == "conv32":
        self.decoder = SmallNorbConvReconstructionModule(num_capsules=num_classes,
                                                         imsize=imsize, 
                                                         img_channels=img_channels, 
                                                         batchnorm=batchnorm)            
    else:
        self.decoder = ConvReconstructionModule(num_capsules=num_classes,
                                                imsize=imsize, 
                                                img_channels=img_channels,
                                                batchnorm=batchnorm)
    
    if loss == "L2":
        self.reconstruction_criterion = nn.MSELoss(reduce=False)
    if loss == "L1":
        self.reconstruction_criterion = nn.L1Loss(reduce=False)
  
  def forward(self, x, target=None):
    output = self.conv_layer(x)
    output = self.primary_capsules(output)
    output = self.digit_caps(output)
    reconstruction, masked = self.decoder(output, target)

    return output, reconstruction, masked
  
  def loss(self, images, labels, capsule_output,  reconstruction, alpha):
    marg_loss = self.margin_loss(capsule_output, labels)
    rec_loss = self.reconstruction_loss(images, reconstruction)
    total_loss = (marg_loss + alpha * rec_loss).mean()
    return total_loss, rec_loss.mean(), marg_loss.mean()
  
  def margin_loss(self, x, labels):
    batch_size = x.size(0)
    v_c = torch.norm(x, dim=2, keepdim=True)
    
    left = functional.relu(0.9 - v_c).view(batch_size, -1) ** 2
    right = functional.relu(v_c - 0.1).view(batch_size, -1) ** 2

    loss = labels * left + 0.5 *(1-labels)*right
    loss = loss.sum(dim=1)
    return loss
  
  def reconstruction_loss(self, data, reconstructions):
    batch_size = reconstructions.size(0)
    reconstructions = reconstructions.view(batch_size, -1)
    data = data.view(batch_size, -1)
    loss = self.reconstruction_criterion(reconstructions, data)
    loss = loss.sum(dim=1)
    return loss
