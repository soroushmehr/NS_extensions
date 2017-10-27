from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import pdb, random

""" Generate a data structure to support 1d NS models. Expects:
D - list of tuples (datasets) such that each element consists of:
    Di[0] - string distribution type ('exponential', 'normal', 'uniform', 'laplace')
    Di[1] - np array (1d) with samples from the distribution
    Di[2] - list of distribution parameters ([mu, sigma^2]) 
"""


class simple_data:
    """ Class for appropriate data structures """
    def __init__(self, D):
	self.datasets = []
	for d in D:
	    self.datasets.append(dataset(d))
	self.n_d, self.n_x = len(D), D[0][1].shape[1] 
	## mini batching variables
	self.start, self.epoch = 0,0

    def next_batch(self, batch_size, sample_size=None, shuffle=True):
        """ yield a batch of datasets of size m (each with n samples) """
        start = self.start
        # Shuffle for the first epoch
        if self.epoch == 0 and start == 0 and shuffle:
	    random.shuffle(self.datasets)
        # Go to the next epoch
        if start + batch_size > self.n_d:
            # Finished epoch
            self.epoch += 1
            # Get the rest examples in this epoch
            rest_num_examples = self.n_d - start
            inputs_rest_part = self.datasets[start:self.n_d]
            # Shuffle the data
            if shuffle:
		random.shuffle(self.datasets)
            # Start next epoch
            start = 0
            self.start = batch_size - rest_num_examples
            end = self.start
            inputs_new_part = self.datasets[start:end]
            return np.concatenate((inputs_rest_part, inputs_new_part), axis=0)
        else:
            self.start += batch_size
            end = self.start
            return self.datasets[start:end]
	    

class dataset:
    """ Class for individual datasets """
    def __init__(self, d):
	self.dist_type = d[0]
	self.x = d[1]
	self.mu, self.var = d[2]
	self.n, self.d = self.x.shape
		
