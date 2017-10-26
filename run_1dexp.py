import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from matplotlib import rc
import pickle, sys, pdb, gzip, cPickle
import numpy as np
from sklearn.manifold import TSNE as tsne
import tensorflow as tf
from data.simple_ns_data import simple_data
from models.simple_ns import basic_ns

### Script to run 1d experiments with distribution data ###

# Load and conver data to relevant type
target = './data/simple_data.pickle'
batch_size = 32
with open(target, "rb") as input_file:
    D = cPickle.load(input_file)
data = simple_data(D)
n_D, n_x = data.n_d, data.n_x

# Specify model parameters
lr = (3e-3,)
n_z, n_c, n_e = 32, 3, 128
n_hidden = [128, 128, 128]
n_epochs = 50
x_dist = 'Gaussian'
batchnorm, mc_samps = False, 1
binarize, logging, verbose = True, False, 1

model = basic_ns(n_D, n_x, n_z, n_c, n_e, n_hidden, x_dist=x_dist, batchnorm=batchnorm, mc_samples=mc_samps)
	
# Train model and measure performance on test set
model.train(data, n_epochs, batch_size, lr, binarize=binarize, logging=logging, verbose=verbose)

# Encode datasets into C-space and plot
D = np.array([d.x for d in data.datasets])
encoded = np.squeeze(model.encode_batch_new(D))
dist_type = [d.dist_type for d in data.datasets]

fig = plt.figure()
ax = Axes3D(fig)
colours = ['r','b','g','y']
for (i,cla) in enumerate(set(dist_type)):
    xc = [p for (j,p) in enumerate(encoded[:,0]) if dist_type[j]==cla]
    yc = [p for (j,p) in enumerate(encoded[:,1]) if dist_type[j]==cla]
    zc = [p for (j,p) in enumerate(encoded[:,2]) if dist_type[j]==cla]
    ax.scatter(xc,yc,zc,c=colours[i],label=cla)
plt.legend(loc=4)
plt.show()

