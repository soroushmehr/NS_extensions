from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.tensorboard.plugins import projector

import pdb

""" Module containing shared functions and structures for DGMS """

############# (log) Density functions ##############

def gaussianLogDensity(inputs, mu, log_var):
    """ Gaussian log density """
    b_size = tf.cast(tf.shape(mu)[0], tf.float32)
    D = tf.cast(tf.shape(inputs)[-1], tf.float32)
    xc = inputs - mu
    return -0.5*(tf.reduce_sum((xc * xc) / tf.exp(log_var), axis=-1) + tf.reduce_sum(log_var, axis=-1) + D * tf.log(2.0*np.pi))

def standardNormalLogDensity(inputs):
    """ Standard normal log density """
    mu = tf.zeros_like(inputs)
    log_var = tf.log(tf.ones_like(inputs))
    return gaussianLogDensity(inputs, mu, log_var)

def bernoulliLogDensity(inputs, logits):
    """ Bernoulli log density """
    return -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=logits), axis=-1)

def multinoulliLogDensity(inputs, logits):
    """ Categorical log density """
    return -tf.nn.softmax_cross_entropy_with_logits(labels=inputs, logits=logits)

def multinoulliUniformLogDensity(inputs):
    """ Uniform Categorical log density """
    logits = tf.ones_like(inputs)
    return -tf.nn.softmax_cross_entropy_with_logits(labels=inputs, logits=logits)



############## Neural Network modules ##############

def initNetwork(n_in, n_hid, n_out, vname):
    """ Initialize a generic dense network with arbitrary structure """
    weights = {}
    for layer, neurons in enumerate(n_hid):
        weight_name, bias_name = 'W'+str(layer), 'b'+str(layer)
        if layer == 0:
       	    weights[weight_name] = tf.get_variable(shape=[n_in, n_hid[layer]], name=vname+weight_name, initializer=xavier_initializer())
    	else:
    	    weights[weight_name] = tf.get_variable(shape=[n_hid[layer-1], n_hid[layer]], name=vname+weight_name, initializer=xavier_initializer(uniform=False))
    	weights[bias_name] = tf.Variable(tf.zeros(n_hid[layer]) + 1e-1, name=vname+bias_name)
    return weights

def initGaussNet(n_in, n_hid, n_out, vname):
    """ Initialize the weights of a network parameterizeing a Gaussian distribution"""
    weights = initNetwork(n_in, n_hid, n_out, vname)
    weights['Wmean'] = tf.get_variable(shape=[n_hid[-1], n_out], name=vname+'Wmean', initializer=xavier_initializer(uniform=False))
    weights['bmean'] = tf.Variable(tf.zeros(n_out) + 1e-1, name=vname+'bmean')
    weights['Wvar'] = tf.get_variable(shape=[n_hid[-1], n_out], name=vname+'Wvar', initializer=xavier_initializer(uniform=False))
    weights['bvar'] = tf.Variable(tf.zeros(n_out) + 1e-1, name=vname+'bvar')
    return weights

def initCatNet(n_in, n_hid, n_out, vname):
    """ Initialize the weights of a network parameterizeing a Gaussian distribution"""
    weights = initNetwork(n_in, n_hid, n_out, vname)
    weights['Wout'] = tf.get_variable(shape=[n_hid[-1], n_out], name=vname+'Wout', initializer=xavier_initializer(uniform=False))
    weights['bout'] = tf.Variable(tf.zeros([n_out]) + 1e-1, name=vname+'bout')
    return weights

def forwardPass(x, weights, n_h, nonlinearity, bn, training, scope, reuse):
    """ Generic forward pass through a dense network with arbitrary architecture """
    h = x
    for layer, neurons in enumerate(n_h):
	weight_name, bias_name = 'W'+str(layer), 'b'+str(layer)
	h = tf.matmul(h, weights[weight_name]) + weights[bias_name]
	if bn:
	    name = scope+'_bn'+str(layer)
	    h = tf.layers.batch_normalization(h, training=training, name=name, reuse=reuse)
	h = nonlinearity(h)
    return h	

def forwardPassGauss(x, weights, n_h, nonlinearity, bn, training=True, scope='scope', reuse=True):
    """ Forward pass through the network with given weights - Gaussian output """
    h = forwardPass(x, weights, n_h, nonlinearity, bn, training, scope, reuse)
    mean = tf.matmul(h, weights['Wmean']) + weights['bmean']
    log_var = tf.matmul(h, weights['Wvar']) + weights['bvar']
    return mean, log_var

def samplePassGauss(x, weights, n_h, nonlinearity, bn, mc_samps=1, training=True, scope='scope', reuse=True):
    """ Forward pass through the network with given weights - Gaussian sampling """
    mean, log_var = forwardPassGauss(x, weights, n_h, nonlinearity, bn, training, scope, reuse)
    shape = tf.concat([tf.constant([mc_samps,]), tf.shape(mean)], axis=-1)
    epsilon = tf.random_normal(shape, dtype=tf.float32)
    return mean, log_var, mean + tf.sqrt(tf.exp(log_var)) * epsilon

def forwardPassCatLogits(x, weights, n_h, nonlinearity, bn, training=True, scope='scope', reuse=True):
    """ Forward pass through the network with weights as a dictionary """
    h = forwardPass(x, weights, n_h, nonlinearity, bn, training, scope, reuse) 
    logits = tf.matmul(h, weights['Wout']) + weights['bout']
    return logits

def forwardPassCat(x, weights, n_h, nonlinearity, bn=False, training=True, scope='scope', reuse=True):
    """ Forward pass through network with given weights - Categorical output """
    return tf.nn.softmax(forwardPassCatLogits(x, weights, n_h, nonlinearity, bn, training, scope, reuse))

def forwardPassBernoulli(x, weights, n_h, nonlinearity, bn=False, training=True, scope='scope', reuse=True):
    """ Forward pass through the network with given weights - Bernoulli output """
    return tf.nn.sigmoid(forwardPassCatLogits(x, weights, n_h, nonlinearity, bn, training, scope, reuse))


############## Statistic Network modules ############## 

def initStatNet(n_in, n_h, n_e, n_out, vname):
    """ Initialize a simple statistic network for NS models """
    encoder = initNetwork(n_in, n_h, n_e, vname+'_encoder')
    decoder = initGaussNet(n_e, n_h, n_out, vname+'_decoder')
    return {'encoder':encoder, 'decoder':decoder}

def samplePassStatistic(x, statNet, n_h, nonlinearity=tf.nn.relu, bn=True, mc_samps=1, training=True, scope='scope', reuse=True):
    """ Map from a dataset (x) to params of a context vector distribution """
    E = forwardPass(x, statNet['encoder'], n_h, nonlinearity, bn, training, scope+'_encoder', reuse)
    V = tf.expand_dims(tf.reduce_mean(E, axis=0), 0)
    return samplePassGauss(V, statNet['decoder'], n_h, nonlinearity, bn, mc_samps, training, scope+'_decoder', reuse)
