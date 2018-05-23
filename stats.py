import time
import numpy as np
import os

class Statistics:
  
  def __init__(self, LOG_DIR):
    self.TEST_LOSSES = []
    self.TRAIN_LOSSES = []
    self.TEST_ACC = []
    self.RECONSTRUCTION_LOSS = []
    self.RECONSTRUCTION_LOSS_TEST = []
    self.reset_tracking_stats()
    self.initialize_logfile(LOG_DIR)

  def initialize_logfile(self, LOG_DIR):
    logname = "log-{}.txt".format(time.time())
    self.log_file = os.path.join(LOG_DIR, logname)
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    f = open(self.log_file, 'w')
    f.write("epoch, time, test_loss, train_loss, test_accuracy, reconstruction_loss_train, reconstruction_loss_test \n")
    f.close()
  
  def reset_tracking_stats(self):
    self.train_loss = 0
    self.train_steps = 0
    self.test_loss = 0
    self.test_steps = 0
    self.test_correct = 0
    self.test_num_samples = 0
    self.rec_loss = 0
    self.test_rec_loss = 0
    self.time = time.time()    
    
  def track_train(self, train_loss, rec_loss):
    self.train_steps += 1
    self.train_loss += train_loss
    self.rec_loss += rec_loss

  def track_test(self, test_loss, rec_loss, target, prediction):
    # Calculate accuracy
    self.test_correct += (target.max(dim=1)[1] == prediction.max(dim=1)[1]).sum().item()
    self.test_num_samples += target.size(0)
    
    # Track test loss
    self.test_steps += 1
    self.test_loss += test_loss
    self.test_rec_loss += rec_loss
  
  def save_stats(self, epoch):
      
    time_spent = time.time() - self.time
    train_loss = self.train_loss / self.train_steps
    rec_loss = self.rec_loss / self.train_steps
    test_loss = self.test_loss / self.test_steps
    test_acc = self.test_correct / self.test_num_samples
    test_rec_loss = self.test_rec_loss / self.test_steps
    self.TEST_ACC.append(test_acc)
    self.TEST_LOSSES.append(test_loss)
    self.TRAIN_LOSSES.append(train_loss)
    self.RECONSTRUCTION_LOSS.append(rec_loss)
    self.RECONSTRUCTION_LOSS_TEST.append(test_rec_loss)
    #print("Epoch: {:3.0f} \t Time: {:3.0f} \t Test: {:.3f} \t Train: {:.3f} \t Accuracy: {:3.4f} Reconstruction: {:3.4f}".format(epoch, time_spent, test_loss, train_loss, test_acc*100, rec_loss))
    
    f = open(self.log_file, 'a')
    to_write = str([epoch, time_spent, test_loss, train_loss, test_acc*100, rec_loss, test_rec_loss])[1:-1] + "\n"
    f.write(to_write)
    f.close()
    
    self.reset_tracking_stats()
    