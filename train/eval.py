# coding: utf-8
import os
import time
import numpy as np
import pandas as pd
import datetime
import tqdm
import csv
import fire
import argparse
import pickle
from sklearn import metrics
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import LabelBinarizer

import model as Model
import data_loader
from data_loader import read_file

TAGS = data_loader.TAGS

class Predict(object):
    def __init__(self, config):
  
        self.model_load_path = config.model_load_path
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.is_cuda = torch.cuda.is_available()
        self.use_val_split = config.use_val_split
        self.save_predictions = config.save_predictions
        self.save_path = config.save_path
        self.build_model()
        self.get_dataset()

    def get_model(self):
        
        # Define audio segment length
        self.input_length = 73600
        return Model.ShortChunkCNN_Res()
    

    def build_model(self):
        self.model = self.get_model()
       
        # load model
        self.load(self.model_load_path)

        # cuda
        if self.is_cuda:
            self.model.cuda()


    def get_dataset(self):

        if self.use_val_split:
             test_file = os.path.join(config.splits_path, 'jamendo_moodtheme-validation.tsv')
        else:           
             test_file = os.path.join(config.splits_path, 'jamendo_moodtheme-test.tsv')
        
        self.file_dict= read_file(test_file)
        self.test_list= list(self.file_dict.keys())
        self.mlb = LabelBinarizer().fit(TAGS)

    def load(self, filename):
        S = torch.load(filename)
        if 'spec.mel_scale.fb' in S.keys():
            self.model.spec.mel_scale.fb = S['spec.mel_scale.fb']
        self.model.load_state_dict(S)

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def get_tensor(self, fn):
        # load audio
        filename = self.file_dict[fn]['path']
        npy_path = os.path.join(self.data_path, filename)
        raw = np.load(npy_path,mmap_mode='r')

        # split chunk
        length = len(raw)
        hop = (length - self.input_length) // self.batch_size
        x = torch.zeros(self.batch_size, self.input_length)
        for i in range(self.batch_size):
            x[i] = torch.Tensor(raw[i*hop:i*hop+self.input_length]).unsqueeze(0)
        return x

    def get_auc(self, est_array, gt_array):
        roc_aucs  = metrics.roc_auc_score(gt_array, est_array, average='macro')
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
        return roc_aucs, pr_aucs

    def test(self):
        roc_auc, pr_auc, loss = self.get_test_score()
        print('loss: %.4f' % loss)
        print('roc_auc: %.4f' % roc_auc)
        print('pr_auc: %.4f' % pr_auc)

    def get_test_score(self):
        self.model = self.model.eval()
        est_array = []
        gt_array = []
        losses = []
        reconst_loss = nn.BCELoss()
        for line in tqdm.tqdm(self.test_list):
            fn = line

            # load and split
            x = self.get_tensor(fn)

            ground_truth = np.sum(self.mlb.transform(self.file_dict[fn]['tags']), axis=0)

            # forward
            x = self.to_var(x)
            y = torch.tensor([ground_truth.astype('float32') for i in range(self.batch_size)]).cuda()
            out = self.model(x)
            loss = reconst_loss(out, y)
            losses.append(float(loss.data))
            out = out.detach().cpu()

            # estimate
            estimated = np.array(out).mean(axis=0)
            est_array.append(estimated)
            gt_array.append(ground_truth)

        est_array, gt_array = np.array(est_array), np.array(gt_array)
        loss = np.mean(losses)

        roc_auc, pr_auc = self.get_auc(est_array, gt_array)
        
        if self.save_predictions:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path) 
            np.save(os.path.join(self.save_path,'predictions.npy'),est_array)
            np.save(os.path.join(self.save_path,'ground_truth.npy'),gt_array)
            
        return roc_auc, pr_auc, loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_load_path', type=str, default='.')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--splits_path', type=str, default='./../splits/')

    parser.add_argument('--use_val_split', type=int, default=0)

    parser.add_argument('--save_predictions', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='.')

    config = parser.parse_args()

    p = Predict(config)
    p.test()






