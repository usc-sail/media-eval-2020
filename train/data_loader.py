
'''
The training set here is composed of songs from the Jamendo dataset (https://github.com/MTG/mtg-jamendo-dataset), 
the music4all dataset (https://sites.google.com/view/contact4music4all), 
and the Million Song Dataset (http://millionsongdataset.com/).

Both the validation and test sets are from the following Jamendo split: https://github.com/MTG/mtg-jamendo-dataset/tree/master/data/splits/split-0
'''
# coding: utf-8
import pickle
import os
import csv
import numpy as np
from torch.utils import data
from sklearn.preprocessing import LabelBinarizer
import torch
import random
from torch.utils.data import Sampler

# Tags are the 56 emotions and themes tags from the 2020 MediaEval challenge (https://multimediaeval.github.io/editions/2020/tasks/music/)

TAGS = [
'mood/theme---action',
'mood/theme---adventure',
'mood/theme---advertising',
'mood/theme---background',
'mood/theme---ballad',
'mood/theme---calm',
'mood/theme---children',
'mood/theme---christmas',
'mood/theme---commercial',
'mood/theme---cool',
'mood/theme---corporate',
'mood/theme---dark',
'mood/theme---deep',
'mood/theme---documentary',
'mood/theme---drama',
'mood/theme---dramatic',
'mood/theme---dream',
'mood/theme---emotional',
'mood/theme---energetic',
'mood/theme---epic',
'mood/theme---fast',
'mood/theme---film',
'mood/theme---fun',
'mood/theme---funny',
'mood/theme---game',
'mood/theme---groovy',
'mood/theme---happy',
'mood/theme---heavy',
'mood/theme---holiday',
'mood/theme---hopeful',
'mood/theme---inspiring',
'mood/theme---love',
'mood/theme---meditative',
'mood/theme---melancholic',
'mood/theme---melodic',
'mood/theme---motivational',
'mood/theme---movie',
'mood/theme---nature',
'mood/theme---party',
'mood/theme---positive',
'mood/theme---powerful',
'mood/theme---relaxing',
'mood/theme---retro',
'mood/theme---romantic',
'mood/theme---sad',
'mood/theme---sexy',
'mood/theme---slow',
'mood/theme---soft',
'mood/theme---soundscape',
'mood/theme---space',
'mood/theme---sport',
'mood/theme---summer',
'mood/theme---trailer',
'mood/theme---travel',
'mood/theme---upbeat',
'mood/theme---uplifting']

'''
Below is an implementation of class-aware resampling. This implementation
is modified from the version used by Tu et al 
(https://arxiv.org/abs/2007.09654).
'''

class RandomCycleIter:

    def __init__(self, data_list, test_mode=False):
        self.data_list = list(data_list)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1

        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)

        return self.data_list[self.i]

def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
    i = 0
    j = 0
    while i < n:

        #         yield next(data_iter_list[next(cls_iter)])

        if j >= num_samples_cls:
            j = 0

        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]] * num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]

        i += 1
        j += 1


class ClassAwareSampler(Sampler):

    def __init__(self, data_source, num_samples_cls=3, reduce = 4):
        random.seed(0)
        torch.manual_seed(0)
        num_classes = len(TAGS)
        self.class_counts = data_source.get_class_weights()

        self.epoch = 0

        self.class_iter = RandomCycleIter(range(num_classes))

        self.cls_data_list = data_source.build_cls_data_list()
            
        self.num_classes = len(TAGS)
        self.data_iter_list = [RandomCycleIter(x) for x in self.cls_data_list] # repeated
        self.num_samples = int(max(self.class_counts) * self.num_classes / reduce) # attention, ~ 1500(person) * 80
        self.num_samples_cls = num_samples_cls
        print('>>> Class Aware Sampler Built! Class number: {}, reduce {}'.format(num_classes, reduce))

    def __iter__(self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)

    def __len__(self):
        return self.num_samples

    def set_epoch(self,  epoch):
        self.epoch = epoch

    def get_sample_per_class(self):
        condition_prob = np.zeros([self.num_classes, self.num_classes])
        sample_per_cls = np.asarray([len(x) for x in self.gt_labels])
        rank_idx = np.argsort(-sample_per_cls)

        for i, cls_labels in enumerate(self.gt_labels):
            num = len(cls_labels)
            condition_prob[i] = np.sum(np.asarray(cls_labels), axis=0) / num

        sum_prob = np.sum(condition_prob, axis=0)
        need_sample = sample_per_cls / sum_prob


'''
Below is a function to parse a file that contains train, val, or test splits.
This function assumes that the input tsv file is formatted as follows:

1st column: unique song identifier
2nd column: relative path to the given song, without the file extension
Remaining columns: Tags for the given song, separated by tabs
'''

def read_file(tsv_file):
    tracks = {}
    
    if 'validation' in tsv_file:
        split = 'validation'
    elif 'test' in tsv_file:
        split = 'test'
    else:
        split = 'train'
    
    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            track_id = row[0]
            tracks[track_id] = {
                'path': os.path.join('jamendo', split, row[1]+'.npy'), # location for the given song
                'tags': row[2:], # grab tags
            }
    return tracks


class AudioFolder(data.Dataset):
    def __init__(self, root, split, sampling_type, input_length=None, TSV_PATH = '.'):
        self.root = root
        self.split = split
        self.input_length = input_length
        self.get_songlist(TSV_PATH)
        self.sampling_type = sampling_type

    def __getitem__(self, index):
        npy, tag_binary = self.get_npy(index)
        return npy.astype('float32'), tag_binary.astype('float32')

    def get_songlist(self, TSV_PATH):
        self.mlb = LabelBinarizer().fit(TAGS)
        if self.split == 'TRAIN':
            train_file = os.path.join(TSV_PATH, 'jamendo_moodtheme-train.tsv')
            self.file_dict = read_file(train_file)
            self.fl = list(self.file_dict.keys())
        elif self.split == 'VALID':
            val_file = os.path.join(TSV_PATH,'jamendo_moodtheme-validation.tsv')
            self.file_dict= read_file(val_file)
            self.fl = list(self.file_dict.keys())
        elif self.split == 'TEST':
            test_file = os.path.join(TSV_PATH, 'jamendo_moodtheme-test.tsv')
            self.file_dict= read_file(test_file)
            self.fl = list(self.file_dict.keys())
        else:
            print('Split should be one of [TRAIN, VALID, TEST]')

    '''
    Load an instance from the given npy file by randomly selecting a 
    segment of length specified in main.py. 
    '''
    
    def get_npy(self, index):
        
        if self.sampling_type == 'standard':
            index = self.fl[index]

        filename = self.file_dict[index]['path']
        npy_path = os.path.join(self.root, filename)
        npy = np.load(npy_path, mmap_mode='r')
        random_idx = int(np.floor(np.random.random(1) * (len(npy)-self.input_length)))
        npy = np.array(npy[random_idx:random_idx+self.input_length])
        tag_binary = np.sum(self.mlb.transform(self.file_dict[index]['tags']), axis=0)
        return npy, tag_binary

    def __len__(self):
        return len(self.fl)

  
    def get_gt_labels(self):
        
        all_labels = []
        for key in self.file_dict.keys():
                tag_binary = np.sum(self.mlb.transform(self.file_dict[key]['tags']), axis=0)
                all_labels.append(tag_binary)
        all_labels = np.array(all_labels)
        
        return all_labels
        
    def get_class_weights(self):
        
        all_labels = self.get_gt_labels()
        
        return np.sum(all_labels,axis=0)
        
    def build_cls_data_list(self):
    
        cls_data_list = []
        
        for tag in TAGS:
            tag_list = []
            for key in self.fl:
                if tag in self.file_dict[key]['tags']:
                    tag_list.append(key)
            
            cls_data_list.append(tag_list)

        return cls_data_list
        
def get_audio_loader(root, batch_size, path_to_tsv, sampling_type, split='TRAIN', num_workers=0, input_length=None):
    # Make sure to pass the correct path to the train/val/test split tsv files
    
    dataset = AudioFolder(root, split=split, sampling_type = sampling_type, input_length=input_length, TSV_PATH = path_to_tsv)
    class_weights = dataset.get_class_weights()
    
    '''
    Either build a class-aware sampler, or just use a "standard" sampler
    where the instances are randomly shuffled and visited once per epoch.
    '''
    
    if sampling_type == 'class_aware':
        sampler = ClassAwareSampler(data_source=dataset, reduce=4)
        data_loader = data.DataLoader(dataset,
                                      sampler=sampler,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=num_workers)
     
    else:
        data_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=num_workers)

    return data_loader, class_weights

