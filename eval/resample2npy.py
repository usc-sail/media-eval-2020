'''
This script takes a directory of mp3s as input, and for each mp3, resamples to 
the provided sampling rate and outputs in numpy format. To be used for downstream tasks.
'''

import os
import numpy as np
import glob
import librosa
import fire
import tqdm
import sys


class Processor:
	def __init__(self):
		self.fs = 16000 # Sampling rate for resampling

    # Find all mp3s contained in the given directory
	def get_paths(self, data_path, output_path):
		self.files = glob.glob(os.path.join(data_path, '**/*.mp3'), recursive=True)
		self.npy_path = os.path.join(output_path, 'npy')

		if not os.path.exists(output_path):
			os.makedirs(output_path)
    
    # Resample
	def get_npy(self, fn):
		x, sr = librosa.core.load(fn, sr=self.fs)
		return x

	def iterate(self, data_path):
		output_path = './npy_data' # Replace with desired output directory
		self.get_paths(data_path, output_path)
		for fn in tqdm.tqdm(self.files):
			output_fn = os.path.join(output_path, fn.split('/')[-1][:-3]+'npy')
			if not os.path.exists(output_fn):
				try:
					x = self.get_npy(fn)
					np.save(open(output_fn, 'wb'), x) # Save in npy format
				except RuntimeError:
					# some audio files are broken
					print(fn)
					continue

if __name__ == '__main__':
	p = Processor()
	fire.Fire({'run': p.iterate})
