""" Script to generate 1d data for NS experiment as in (Edwards and Storkey 2017) """
import numpy as np
from numpy.random import exponential, normal, uniform, laplace
import pickle, pdb, sys

""" Data generation 
Generation process: 
for i in range(N_D):
  Draw P_i~U{Exponential, Gaussian, Uniform, Laplacian}
  Draw theta: mu~U(-1,1); sigma^2~U(0.5,2)
  for j in range(n_s):
    x_j^(i)~P(theta)
"""

### argv[1] - number of datasets to generate

n_datasets, n_samps = int(sys.argv[1]), 200
distributions = [(exponential, 'exponential'), (normal, 'normal'), 
		 (uniform,'uniform'), (laplace, 'laplace')]
d = []

for i in range(n_datasets):
    choice = np.random.randint(4)
    P, name = distributions[choice]
    mu, var = uniform(-1,1), uniform(0.5,2)
    if name == 'exponential':
	d.append((name, P(np.sqrt(var), (n_samps,1)), [var, var]))
    elif name == 'uniform':
	radius = np.sqrt(12*var)/2.
	a,b = mu-radius, mu+radius
	d.append((name, P(mu, np.sqrt(var), (n_samps,1)), [mu, var]))
    else:
	d.append((name, P(mu, np.sqrt(var), (n_samps,1)), [mu, var]))

with open('data/simple_data.pickle', 'wb') as handle:
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
