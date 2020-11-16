# USC SAIL MediaEval 2020 Submission
MediaEval 2020: Emotions and Themes in Music


This repo is based on the USC SAIL submission for MediaEval 2020 (https://multimediaeval.github.io/editions/2020/tasks/music/), but it is designed to be easy to setup and evaluate for general music tagging problems. 

## Requirements

1. Python >= 3.7
2. PyTorch >= 1.2
3. See requirements.txt

## Usage

1. First run *resample2npy.py* to resample all mp3 files in given a directory to 16 kHz.
	* python -u resample2npy.py run /path/to/mp3s/
2. Create a .tsv file with music tag labels, as specified in *data_loader.py*. 
3. Run *train.py*. The following example uses binary cross-entropy as the loss function, and also uses mixup (https://arxiv.org/pdf/1710.09412).
	* python -u main.py --data_path /path/to/npy_data/ --splits_path /path/to/splits_tsvs/ --model_save_path /output/path/ --use_mixup 1 --loss_function bce --sampling_type standard 
4. Run *eval.py*. The following script will evaluate a trained model on the given test split.
	* python -u eval.py --data_path /path/to/npy_data/ --splits_path /path/to/splits_tsvs/ --model_load_path /path/to/model/best_model.pth --use_val_split 0 --save_predictions 1 --save_path /output/path/

## Loss functions

In this repository, we provide multilabel, multi-class implementations for the following loss functions (see *losses.py*):

- Focal loss (https://arxiv.org/abs/1708.02002)
- Class-balanced loss (https://arxiv.org/abs/1901.05555)
- Distribution-balanced loss (https://arxiv.org/abs/2007.09654)

## Model

Our model is modified from the Short-Chunk CNN with Residual Connections presented by Won et al. (https://arxiv.org/abs/2006.00751)

## Dataset

The low-level feature extract layers for the CNN were pretrained on the Million Song Dataset (http://millionsongdataset.com/). See our paper for implementation details.
We used the training, validation, and test splits as specified by the challenge. We combined our training set with instances with matching tags from the Music4All dataset (https://ieeexplore.ieee.org/document/9145170).
