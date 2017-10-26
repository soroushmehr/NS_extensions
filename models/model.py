from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import sys, os, pdb

import numpy as np
import utils.dgm as dgm

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

""" Super class for neural statistician models """

class model(object):

    def __init__(self, n_d, n_x, n_z, n_c, n_e, n_hid=[4], x_dist='Gaussian', nonlinearity=tf.nn.relu, batchnorm=False, mc_samples=1, l2_reg=0.3, ckpt=None):
	
	self.n_d, self.n_x = n_d, n_x      # number of datasets and data dimensions (same for all)
	self.n_z, self.n_c = n_z, n_c      # number of latent/context dimensions
	self.n_e, self.n_hid = n_e, n_hid  # network architectures
	self.nonlinearity = nonlinearity   # activation function
	self.x_dist = x_dist               # likelihood for inputs
	self.bn = batchnorm                # use batch normalization
	self.mc_samples = mc_samples       # MC samples for estimation
	self.l2_reg = l2_reg               # weight regularization scaling constant
	self.name = 'model'                # model name
	self.ckpt = ckpt 		   # preallocated checkpoint dir
	
	# placeholders for necessary values
	self.n, self.n_train = 1,1	 # initialize data size
	self.allocate_directory()  		
	self.build_model()
	self.loss = self.compute_loss()
	self.encoded = self.encode(self.x)
	self.encoded_batch = self.encode_batch(self.D)
	self.session = tf.Session()

    def train(self, Data, n_epochs, batch_size, lr, eval_samps=None, binarize=False, logging=False, verbose=1):
	""" Method for training the models """
	#self.data_init(Data, eval_samps)
	self.lr = self.set_learning_rate(lr)
        ## define optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
	gvs = optimizer.compute_gradients(self.loss)
	capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
	    self.optimizer = optimizer.apply_gradients(capped_gvs, global_step=self.global_step) 
	
        ## initialize session and train
        max_acc, epoch, step = 0, 0, 0
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if logging:
                writer = tf.summary.FileWriter(self.LOGDIR, sess.graph)

            while epoch < n_epochs:
                list_of_datasets = Data.next_batch(batch_size)
		Dbatch = np.array([d.x for d in list_of_datasets])
                _, loss_batch = sess.run([self.optimizer, self.loss], feed_dict={self.D: Dbatch, self.x: list_of_datasets[0].x})
                if logging:
		    writer.add_summary(summary_elbo, global_step=self.global_step)

                if Data.epoch > epoch:
                    epoch += 1
                    fd = self._printing_feed_dict(np.array([d.x for d in Data.datasets]))
		    saver.save(sess, self.ckpt_dir, global_step=step+1)
		    if verbose == 1:
                        self.print_verbose1(epoch, fd, sess)
		    elif verbose == 2:
                        self.print_verbose2(epoch, fd, sess)

            if logging:
                writer.close()


    def encode_new(self, x):
	""" encode a single dataset """
        saver = tf.train.Saver()
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            saver.restore(session, ckpt.model_checkpoint_path)
            self.phase = False
            encoded = session.run(self.encoded, {self.x:x})
        return encoded0

    def encode_batch_new(self, D):
	""" encode a batch of datasets """
        saver = tf.train.Saver()
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            saver.restore(session, ckpt.model_checkpoint_path)
            self.phase = False
            encoded_batch = session.run(self.encoded_batch, {self.D:D})
        return encoded_batch

### Every instance of model must implement these two methods ###

    def encode(self, x):
	pass

    def build_model(self):
	pass

    def compute_loss(self):
	pass

################################################################
    
    def weight_prior(self):
	weights = [V for V in tf.trainable_variables() if 'W' in V.name]
	return np.sum([tf.reduce_sum(dgm.standardNormalLogDensity(w)) for w in weights])

    def weight_regularization(self):
	weights = [V for V in tf.trainable_variables() if 'W' in V.name]
	return np.sum([tf.nn.l2_loss(w) for w in weights])	

    def data_init(self, Data, eval_samps):
	self._process_data(Data, eval_samps)

    def binarize(self, x):
	return np.random.binomial(1,x)

    def set_learning_rate(self, lr):
	""" Set learning rate """
	self.global_step = tf.Variable(0, trainable=False, name='global_step')
	if len(lr) == 1:
	    return lr[0]
	else:
	    start_lr, rate, final_lr = lr
	    return tf.train.polynomial_decay(start_lr, self.global_step, rate, end_learning_rate=final_lr)     

    def create_placeholders(self):
        """ Create input/output placeholders """
        self.x_train = tf.placeholder(tf.float32, shape=[None, self.n_x], name='x_train')
        self.x_test = tf.placeholder(tf.float32, shape=[None, self.n_x], name='x_test')
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_x], name='x')
	self.D = tf.placeholder(tf.float32, shape=[None, None, self.n_x])

    def _printing_feed_dict(self, Data):
	return {self.D:Data}

    def allocate_directory(self):
	if self.ckpt == None:
            self.LOGDIR = './graphs/'+self.name+'-'+str(self.n_d)+'-'+str(self.n_z)+'-'+str(self.n_c)+'/'
            self.ckpt_dir = './ckpt/'+self.name+'-'+str(self.n_d)+'-'+str(self.n_z)+'-'+str(self.n_c) + '/'
	else: 
            self.LOGDIR = 'graphs/'+self.ckpt+'/' 
	    self.ckpt_dir = './ckpt/' + self.ckpt + '/'
        if not os.path.isdir(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        if not os.path.isdir(self.LOGDIR):
            os.mkdir(self.LOGDIR)

