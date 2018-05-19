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

def initialize_weights(capsnet):
	capsnet.conv_layer.conv.apply(weights_init_xavier)
	capsnet.primary_capsules.apply(weights_init_xavier)
	capsnet.decoder.apply(weights_init_xavier)
	#nn.init.xavier_normal_(capsnet.digit_caps.W)

def main():
	capsnet = CapsNet(reconstruction_type="FC")
	if USE_GPU:
		capsnet.cuda()
	optimizer = torch.optim.Adam(capsnet.parameters())
	
	# Load saved model
	model_path = path.join(SAVE_DIR, SAVED_MODEL)
	if path.isfile(model_path):
		print("Saved model found")
		capsnet.load_state_dict(torch.load(model_path))
	else:
		print("Saved model not found; Model initialized.")
		initialize_weights(capsnet)

	train_loader, test_loader = load_mnist(BATCH_SIZE)
	stats = Statistics()
	
	for epoch in range(MAX_EPOCHS):
		capsnet.train()
		for batch, (data, target) in list(enumerate(train_loader)):
			target = torch.eye(10).index_select(dim=0, index=target)
			data, target = Variable(data), Variable(target)
			if USE_GPU:
				data, target = data.cuda(), target.cuda()
			
			optimizer.zero_grad()
			
			output, reconstructions, masked = capsnet(data, target)
			loss = capsnet.loss(data, target, output, reconstructions)
			
			loss.backward()
			optimizer.step()
			
			stats.track_train(loss.data.item())
			
			if batch % DISPLAY_STEP == 0 and batch != 0:
				capsnet.eval()

				for batch_id, (data, target) in enumerate(test_loader):
					target = torch.eye(10).index_select(dim=0, index=target)
					data, target = Variable(data), Variable(target)
					if USE_GPU:
						data,target = data.cuda(), target.cuda()

					output, reconstructions, masked = capsnet(data)
					loss = capsnet.loss(data, target, output, reconstructions)
					
					stats.track_test(loss.data.item(), target, masked)
					
				stats.save_stats(epoch)
				
				# Save model
				model_path = path.join(SAVE_DIR, "model{}.pt".format(epoch))
				torch.save(capsnet.state_dict(), model_path)
				capsnet.train()

if __name__ == '__main__':
	main()