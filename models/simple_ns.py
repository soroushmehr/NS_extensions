from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

from models.model import model

import sys, os, pdb

import numpy as np
import utils.dgm as dgm 

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


""" 
Implementation of basic neural statistician from Harrison and Storkey (2017): p(c) * p(z_i|c) * p(x_i|c,z_i) 
Inference network: q(c,z|D) = q(c|D) * q(z_i|c,x_i) 
"""

class basic_ns(model):
   
    def __init__(self, n_d, n_x, n_z, n_c, n_e, n_hid, x_dist='Gaussian', nonlinearity=tf.nn.relu, batchnorm=False, l2_reg=0.3, mc_samples=1,ckpt=None):
	
	super(basic_ns, self).__init__(n_d, n_x, n_z, n_c, n_e, n_hid, x_dist, nonlinearity, batchnorm, mc_samples, l2_reg, ckpt)

	""" TODO: add any general terms we want to have here """
	self.name = 'neural_statistician'

    def build_model(self):
	""" Define model components and variables """
	self.create_placeholders()
	self.initialize_networks()
	## model variables and relations ##
	# infernce #
        self.qc_mean, self.qc_lv, self.c_ = dgm.samplePassStatistic(self.x, self.qc_x, self.n_hid, self.nonlinearity, self.bn, self.mc_samples, scope='qc_x', reuse=False)
	self.c_ = tf.reshape(self.c_, [-1, self.n_c])
	self.qz_in = tf.concat([self.x, tf.tile(self.c_, [tf.shape(self.x)[0], 1])], axis=-1) 
	self.qz_mean, self.qz_lv, self.qz_ = dgm.samplePassGauss(self.qz_in, self.qz_xc, self.n_hid, self.nonlinearity, self.bn, scope='qz_xc', reuse=False)
	# generative #
	self.c_prior = tf.random_normal([1, self.n_c])
	self.pz_mean, self.pz_lv, self.pz_ = self.sample_pz(self.c_prior, 200, reuse=False)
	self.pz_ = tf.reshape(self.pz_, [-1, self.n_z])
	if self.x_dist == 'Gaussian':
	    self.px_mean, self.px_lv, self.px_ = dgm.samplePassGauss(self.pz_, self.px_z, self.n_hid, self.nonlinearity, self.bn, scope='px_z', reuse=False)
	elif self.x_dist == 'Bernoulli':
	   self.px_ = dgm.forwardPassBernoulli(self.pz_, self.px_z, self.n_hid, self.nonlinearity, self.bn, scope='px_z', reuse=False)	

    def compute_loss(self):
	""" manipulate computed components and compute loss """
	lb_d = tf.map_fn(self.lowerBound,self.D)
	#weight_priors = self.weight_prior()/self.n_d
	return tf.reduce_mean(lb_d, axis=0)# - weight_priors	

    def lowerBound(self, x):
	""" Compute lower bound for one dataset """
	qc_m, qc_lv, c = dgm.samplePassStatistic(x, self.qc_x, self.n_hid, self.nonlinearity, self.bn, self.mc_samples, scope='qc_x')
	qz_m, qz_lv, z = self.sample_qz(x, c)
	pz_m, pz_lv, _ = self.sample_pz(c, 1)
	l_px = self.compute_logpx(x,z)
	l_pz = dgm.gaussianLogDensity(z, pz_m, pz_lv)
	l_qz = dgm.gaussianLogDensity(z, qz_m, qz_lv)
	l_pc = dgm.standardNormalLogDensity(c)
	l_qc = dgm.gaussianLogDensity(c, qc_m, qc_lv)
	return -tf.reduce_mean(tf.reduce_sum(l_px + l_pz - l_qz, axis=1) + l_pc - l_qc, axis=0) 
	
    def sample_pz(self, c, n_samps, reuse=True):
	""" generate n_samps samples of z from p(z|c) """
	pz_in = tf.reshape(c, [-1, self.n_c])
	z_m, z_lv, z_ =  dgm.samplePassGauss(pz_in, self.pz_c, self.n_hid, self.nonlinearity, self.bn, mc_samps=n_samps, scope='pz_c', reuse=reuse)
	return z_m, z_lv, tf.reshape(z_, [self.mc_samples,-1, self.n_z])

    def sample_qz(self, x, c, reuse=True):
	""" sample z from q(z|x,c) """
	x_, c_ = tf.tile(tf.expand_dims(x,0), [self.mc_samples,1,1]), tf.tile(c, [1,tf.shape(x)[0],1])
	qz_in = tf.reshape(tf.concat([x_, c_], axis=-1), [-1, self.n_x+self.n_c])
	z_m, z_lv, z_ = dgm.samplePassGauss(qz_in, self.qz_xc, self.n_hid, self.nonlinearity, self.bn, scope='qz_xc', reuse=reuse)
	return tf.reshape(z_m, [self.mc_samples,-1,self.n_z]), tf.reshape(z_lv, [self.mc_samples,-1,self.n_z]), tf.reshape(z_, [self.mc_samples,-1,self.n_z])

    def compute_logpx(self, x, z):
	px_in = tf.reshape(z, [-1, self.n_z])
	if self.x_dist == 'Gaussian':
            mean, log_var = dgm.forwardPassGauss(px_in, self.px_z, self.n_hid, self.nonlinearity, self.bn, scope='px_z')
	    mean, log_var = tf.reshape(mean, [self.mc_samples, -1, self.n_x]),  tf.reshape(log_var, [self.mc_samples, -1, self.n_x])
            return dgm.gaussianLogDensity(x, mean, log_var)
        elif self.x_dist == 'Bernoulli':
            logits = dgm.forwardPassCatLogits(px_in, self.px_z, self.n_hid, self.nonlinearity, self.bn, scope='px_z')
	    logits = tf.reshape(logits, [self.mc_samples, -1, self.n_x])
            return dgm.bernoulliLogDensity(x, logits) 

    def encode(self, x):
	""" encode a dataset into a context vector """
	_, _, c = dgm.samplePassStatistic(x, self.qc_x, self.n_hid, self.nonlinearity, self.bn, 1, scope='qc_x')
	return tf.reshape(c, [-1, self.n_c])

    def encode_batch(self, D):
	""" encode a batch of datasets into context vectors """
	encoded_batch = tf.map_fn(self.encode, D)
	return encoded_batch

    def initialize_networks(self):
    	""" Initialize all model networks """
	if self.x_dist == 'Gaussian':
      	    self.px_z = dgm.initGaussNet(self.n_z, self.n_hid, self.n_x, 'px_z_')
	elif self.x_dist == 'Bernoulli':
	    self.px_z = dgm.initCatNet(self.n_z, self.n_hid, self.n_x, 'px_y_')
      	self.pz_c = dgm.initGaussNet(self.n_c, self.n_hid, self.n_z, 'pz_c_')
    	self.qc_x = dgm.initStatNet(self.n_x, self.n_hid, self.n_e, self.n_c, 'qc_x_')
    	self.qz_xc = dgm.initGaussNet(self.n_x+self.n_c, self.n_hid, self.n_z, 'qz_xc_')

    def print_verbose1(self, epoch, fd, sess):
	""" printing verbose after every epoch """
	total = sess.run(self.compute_loss(), fd)
	print("Epoch {}: ELBO: {:5.3f}".format(epoch, total[0]))
