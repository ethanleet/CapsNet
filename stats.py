import time
import numpy as np

class Statistics:
  
  def __init__(self):
    self.TEST_LOSSES = []
    self.TRAIN_LOSSES = []
    self.TEST_ACC = []

    self.reset_tracking_stats()
  
  def reset_tracking_stats(self):
    self.train_loss = 0
    self.train_steps = 0
    self.test_loss = 0
    self.test_steps = 0
    self.test_correct = 0
    self.test_num_samples = 0
    self.time = time.time()    
    
  def track_train(self, train_loss):
    self.train_steps += 1
    self.train_loss += train_loss

  def track_test(self, test_loss, target, prediction):
    # Calculate accuracy
    self.test_correct += (target.max(dim=1)[1] == prediction.max(dim=1)[1]).sum().item()
    self.test_num_samples += target.size(0)

    # Track test loss
    self.test_steps += 1
    self.test_loss += test_loss
  
  def save_stats(self, epoch):
    time_spent = time.time() - self.time
    train_loss = self.train_loss / self.train_steps
    test_loss = self.test_loss / self.test_steps
    test_acc = self.test_correct / self.test_num_samples
    self.TEST_ACC.append(test_acc)
    self.TEST_LOSSES.append(test_loss)
    self.TRAIN_LOSSES.append(train_loss)
    print("Epoch: {:3.0f} \t Time: {:3.0f} \t Test: {:.3f} \t Train: {:.3f} \t Accuracy: {:3.4f}".format(epoch, time_spent, test_loss, train_loss, test_acc*100))
    self.reset_tracking_stats()
    