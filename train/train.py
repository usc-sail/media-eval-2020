import pickle
import os
import time
import numpy as np
import pandas as pd
from sklearn import metrics
import datetime
import csv
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelBinarizer
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import model as Model
import data_loader as train_loader
from data_loader import read_file
import losses

TAGS = train_loader.TAGS # The classes are specified by the list of tags in the data loader

'''
 Add mixup during training
 The approach used here is modified from the following source: https://github.com/facebookresearch/mixup-cifar10
''' 

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# Modify loss function for mixup
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



class Solver(object):
    def __init__(self, data_loader, train_weights, config):
        # data loader
        self.data_loader = data_loader
        self.data_path = config.data_path
        self.input_length = config.input_length
        self.use_mixup = config.use_mixup
        
        # get train weights for class balancing
        self.train_weights = torch.from_numpy(train_weights).float()

        # training settings
        self.n_epochs = config.n_epochs
        self.lr = config.lr
        self.use_tensorboard = config.use_tensorboard
        self.loss_function = config.loss_function

        # model path and step size
        self.model_save_path = config.model_save_path
        self.model_load_path = config.model_load_path
        self.log_step = config.log_step
        self.batch_size = config.batch_size

        # cuda
        self.is_cuda = torch.cuda.is_available()

        # Build model
        self.get_dataset(config.splits_path)
        self.build_model()

        # Tensorboard
        self.writer = SummaryWriter(log_dir=self.model_save_path)

    def get_dataset(self, splits_path):
        train_file = os.path.join(splits_path, 'jamendo_moodtheme-validation.tsv')
        self.file_dict= read_file(train_file)
        self.valid_list= list(self.file_dict.keys())
        self.mlb = LabelBinarizer().fit(TAGS)


    def get_model(self):

        return Model.ShortChunkCNN_Res()
    

    def build_model(self):
        # model
        self.model = self.get_model()

        # load pretrained model
        if len(self.model_load_path) > 1:
            self.load(self.model_load_path)
        
        '''    
        As noted in the original focal loss paper (https://arxiv.org/pdf/1708.02002.pdf), 
        the final output layer bias should be initialized in such a way to avoid the loss being 
        overwhelmed by the large number of negative labels.
        '''
        
        if (self.loss_function == 'focal_loss') or (self.loss_function == 'cb_focal_loss'):
            pi = 0.01
            fill_val = -np.log((1-pi)/pi)  
            with torch.no_grad():
                self.model.dense_2.bias.fill_(fill_val)

        # cuda
        if self.is_cuda:
            self.model.cuda()


        # optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=1e-4)

    # Load a pretrained model.
    def load(self, filename):
    
        # Load the shared layers of the input pretrained model 
        
        pretrained_dict = torch.load(filename)
        model_dict = self.model.state_dict()
        # Load layers weights with the same layer names as the model
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 

        self.model.load_state_dict(model_dict)

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    # Load the user-specified loss function
    def get_loss_function(self):
        if self.loss_function == 'bce':
            loss = nn.BCEWithLogitsLoss()
        elif self.loss_function == 'focal_loss':
            kwargs = {"alpha": 0.25, "gamma": 2.0}
            loss = losses.FocalLoss(**kwargs)
        elif self.loss_function == 'cb_focal_loss':
            kwargs = {"class_weights": self.train_weights,"beta": 0.995, "gamma": 2.0, "loss_function": 'focal_loss'}
            loss = losses.ClassBalancedLoss(**kwargs)
        elif self.loss_function == 'db_focal_loss':
            kwargs = {"class_weights": self.train_weights,"alpha": 0.25, "gamma": 2.0, "loss_function": 'focal_loss',
                      "rebalance_alpha": 0.1, "rebalance_beta": 10, "rebalance_mu": 0.2,
                      "nt_lambda": 2.} 
            loss = losses.DistributionBalancedLoss(**kwargs)
        return loss

    def train(self):
        # Start training
        start_t = time.time()
        current_optimizer = 'adam'
        reconst_loss = self.get_loss_function()
        best_metric = -10e6
        drop_counter = 0

        # Iterate
        for epoch in range(self.n_epochs):
            ctr = 0
            drop_counter += 1
            self.model = self.model.train()
            for x, y in self.data_loader:
                ctr += 1

                # If mixup is specified, 
             
                if self.use_mixup:

                    x, targets_a, targets_b, lam = mixup_data(x, y,
                                                           1, self.is_cuda)
                    # Forward
                    x = self.to_var(x)
                    y = self.to_var(y)
                    targets_a = self.to_var(targets_a)
                    targets_b = self.to_var(targets_b)
                    out = self.model(x)

                    # Backward
                    loss = mixup_criterion(reconst_loss, out, targets_a, targets_b, lam)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                else:
                
                    # Forward
                    x = self.to_var(x)
                    y = self.to_var(y)
                    out = self.model(x)

                    # Backward
                    loss = reconst_loss(out, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                

                # Log
                self.print_log(epoch, ctr, loss, start_t)
            self.writer.add_scalar('Loss/train', loss.item(), epoch)

            # validation
            best_metric = self.validation(best_metric, epoch)

            # schedule optimizer
            current_optimizer, drop_counter = self.opt_schedule(current_optimizer, drop_counter)

        print("[%s] Train finished. Elapsed: %s"
                % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.timedelta(seconds=time.time() - start_t)))
        
        
    def opt_schedule(self, current_optimizer, drop_counter):
        # adam to sgd
        if current_optimizer == 'adam' and drop_counter == 80:
            self.load(os.path.join(self.model_save_path, 'best_model.pth'))
            self.optimizer = torch.optim.SGD(self.model.parameters(), 0.01,
                                            momentum=0.9, weight_decay=0.0001,
                                            nesterov=True)
            current_optimizer = 'sgd_1'
            drop_counter = 0
            print('sgd 1e-2')    
        # first drop
        if current_optimizer == 'sgd_1' and drop_counter == 30:
            self.load(os.path.join(self.model_save_path, 'best_model.pth'))
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.001
            current_optimizer = 'sgd_2'
            drop_counter = 0
            print('sgd 1e-3')
        # second drop
        if current_optimizer == 'sgd_2' and drop_counter == 30:
            self.load(os.path.join(self.model_save_path, 'best_model.pth'))
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.0001
            current_optimizer = 'sgd_3'
            drop_counter = 0
            print('sgd 1e-4')    
        # third drop
        if current_optimizer == 'sgd_3' and drop_counter == 30:
            self.load(os.path.join(self.model_save_path, 'best_model.pth'))
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.00001
            current_optimizer = 'sgd_4'
            print('sgd 1e-5')
        return current_optimizer, drop_counter

    def save(self, filename):
        model = self.model.state_dict()
        torch.save({'model': model}, filename)

    def get_tensor(self, fn, num_val_chunks):
        # load audio
        filename = self.file_dict[fn]['path']
        npy_path = os.path.join(self.data_path, filename)
        raw = np.load(npy_path, mmap_mode='r')

        # split chunk
        length = len(raw)
        hop = (length - self.input_length) // num_val_chunks
        x = torch.zeros(num_val_chunks, self.input_length)
        for i in range(num_val_chunks):
            x[i] = torch.Tensor(raw[i*hop:i*hop+self.input_length]).unsqueeze(0)
        return x

    def get_auc(self, est_array, gt_array):
        roc_aucs  = metrics.roc_auc_score(gt_array, est_array, average='macro')
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
        print('roc_auc: %.4f' % roc_aucs)
        print('pr_auc: %.4f' % pr_aucs)
        return roc_aucs, pr_aucs

    def print_log(self, epoch, ctr, loss, start_t):
        if (ctr) % self.log_step == 0:
            print("[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" %
                    (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        epoch+1, self.n_epochs, ctr, len(self.data_loader), loss.item(),
                        datetime.timedelta(seconds=time.time()-start_t)))

    def validation(self, best_metric, epoch):
        roc_auc, pr_auc, loss = self.get_validation_score(epoch)
        score = pr_auc # Checkpoint on PR-AUC
        if score > best_metric:
            print('best model!')
            best_metric = score
            torch.save(self.model.state_dict(),
                       os.path.join(self.model_save_path, 'best_model.pth'))
        return best_metric


    def get_validation_score(self, epoch):
        
        num_val_chunks = 16 # The number of segments to split a song into during inference for validation
        
        self.model = self.model.eval()
        est_array = []
        gt_array = []
        losses = []
        reconst_loss = self.get_loss_function()
        index = 0
        for line in tqdm.tqdm(self.valid_list):
            fn = line

            # load and split
            x = self.get_tensor(fn, num_val_chunks)

            # ground truth
            ground_truth = np.sum(self.mlb.transform(self.file_dict[fn]['tags']), axis=0)


            # forward
            x = self.to_var(x)
            y = torch.tensor([ground_truth.astype('float32') for i in range(num_val_chunks)]).cuda()
            out = self.model(x)
            loss = reconst_loss(out, y)
            losses.append(float(loss.data))
            out = out.detach().cpu()

            # estimate
            out = torch.sigmoid(out)
            estimated = np.array(out).mean(axis=0)
            est_array.append(estimated)

            gt_array.append(ground_truth)
            index += 1

        est_array, gt_array = np.array(est_array), np.array(gt_array)
        loss = np.mean(losses)
        print('loss: %.4f' % loss)

        roc_auc, pr_auc = self.get_auc(est_array, gt_array)
        self.writer.add_scalar('Loss/valid', loss, epoch)
        self.writer.add_scalar('AUC/ROC', roc_auc, epoch)
        self.writer.add_scalar('AUC/PR', pr_auc, epoch)
        return roc_auc, pr_auc, loss

