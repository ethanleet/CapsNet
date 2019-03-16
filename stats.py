import time
import numpy as np
import os
import tensorboardX

class Statistics:
  
  def __init__(self, LOG_DIR, model_name):
    self.TEST_LOSSES = []
    self.TRAIN_LOSSES = []
    self.TEST_ACC = []
    self.TRAIN_ACC = []
    self.RECONSTRUCTION_LOSS = []
    self.RECONSTRUCTION_LOSS_TEST = []
    self.MARGIN_LOSS = []
    self.MARGIN_LOSS_TEST = []
    self.reset_tracking_stats()
    #self.initialize_logfile(LOG_DIR, previous_log_file)
    print("logs/{}/train".format(model_name))
    self.logger_train = tensorboardX.SummaryWriter("logs/{}/train".format(model_name))
    self.logger_val = tensorboardX.SummaryWriter("logs/{}/val".format(model_name))
    self.global_step = 0
    self.initialize_logfile(LOG_DIR, model_name)

  def initialize_logfile(self, LOG_DIR, model_name):
    # If the logfile already exists we continue to append to this.
    LOG_DIR = LOG_DIR + "_old"
    self.log_file = os.path.join(LOG_DIR, model_name)
    if os.path.isfile(self.log_file): 
        print("Logfile found and loaded.")
        return
    logname = "log-{}.txt".format(model_name)
    self.log_file = os.path.join(LOG_DIR, logname)
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)

    f = open(self.log_file, 'w')

    f.write("epoch, time, test_loss, train_loss, test_accuracy, train_acc, reconstruction_loss_train, reconstruction_loss_test, margin_loss_train, margin_loss_test\n")
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
    self.marg_loss = 0
    self.test_marg_loss = 0
    self.train_correct = 0
    self.train_num_samples = 0
    self.time = time.time()    
    
  def track_train(self, train_loss, rec_loss, marg_loss, target, prediction):
    self.train_steps += 1
    self.global_step += 1
    self.train_loss += train_loss
    self.rec_loss += rec_loss
    self.marg_loss += marg_loss
    self.train_correct += (target.max(dim=1)[1] == prediction.max(dim=1)[1]).sum().item()
    self.train_num_samples += target.size(0)    
    train_acc = (target.max(dim=1)[1] == prediction.max(dim=1)[1]).float().mean().item()
    self.logger_train.add_scalar("accuracy", train_acc, global_step=self.global_step)
    self.logger_train.add_scalar("loss", train_loss, global_step=self.global_step)
    self.logger_train.add_scalar("reconstruction_loss", rec_loss, global_step=self.global_step)
    self.logger_train.add_scalar("margin_loss", marg_loss, global_step=self.global_step)
   

  def track_test(self, test_loss, rec_loss, marg_loss, target, prediction):
    # Calculate accuracy
    self.test_correct += (target.max(dim=1)[1] == prediction.max(dim=1)[1]).sum().item()
    self.test_num_samples += target.size(0)
    
    # Track test loss
    self.test_steps += 1
    self.test_loss += test_loss
    self.test_rec_loss += rec_loss
    self.test_marg_loss += marg_loss
  
  def save_stats(self, epoch):
      
    time_spent = time.time() - self.time
    train_loss = self.train_loss / self.train_steps
    rec_loss = self.rec_loss / self.train_steps
    marg_loss = self.marg_loss / self.train_steps
    train_acc = self.train_correct / self.train_num_samples

    test_loss = self.test_loss / self.test_steps
    test_acc = self.test_correct / self.test_num_samples
    test_rec_loss = self.test_rec_loss / self.test_steps
    test_marg_loss = self.test_marg_loss / self.test_steps
    #print("Epoch: {:3.0f} \t Time: {:3.0f} \t Test: {:.3f} \t Train: {:.3f} \t Accuracy: {:3.4f} Reconstruction: {:3.4f}".format(epoch, time_spent, test_loss, train_loss, test_acc*100, rec_loss))
    self.logger_val.add_scalar("accuracy", test_acc, global_step=self.global_step)
    self.logger_val.add_scalar("loss", test_loss, global_step=self.global_step)
    self.logger_val.add_scalar("reconstruction_loss", test_rec_loss, global_step=self.global_step)
    self.logger_val.add_scalar("margin_loss", test_marg_loss, global_step=self.global_step)
    self.logger_val.add_scalar("time", time_spent, global_step=self.global_step)

    self.TRAIN_ACC.append(train_acc)
    self.TEST_ACC.append(test_acc)
    self.TEST_LOSSES.append(test_loss)
    self.TRAIN_LOSSES.append(train_loss)
    self.RECONSTRUCTION_LOSS.append(rec_loss)
    self.RECONSTRUCTION_LOSS_TEST.append(test_rec_loss)
    self.MARGIN_LOSS.append(marg_loss)
    self.MARGIN_LOSS_TEST.append(test_marg_loss)

    f = open(self.log_file, 'a')
    to_write = str([epoch, time_spent, test_loss, train_loss, test_acc*100, train_acc*100, rec_loss, test_rec_loss, marg_loss, test_marg_loss])[1:-1] + "\n"
    f.write(to_write)
    f.close()

    self.reset_tracking_stats()
    