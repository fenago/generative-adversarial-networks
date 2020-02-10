# %%
'''
## Use a Gaussian Latent Space
The latent space defines the shape and distribution of the input to the generator model used
to generate new images. The DCGAN recommends sampling from a uniform distribution,
meaning that the shape of the latent space is a hypercube. The more recent best practice is
to sample from a standard Gaussian distribution, meaning that the shape of the latent space
is a hypersphere, with a mean of zero and a standard deviation of one. The example below
demonstrates how to generate 500 random Gaussian examples from a 100-dimensional latent
space that can be used as input to a generator model; each point could be used to generate an
image.
'''

# %%
# example of sampling from a gaussian latent space
from numpy.random import randn

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape((n_samples, latent_dim))
	return x_input

# size of latent space
n_dim = 100
# number of samples to generate
n_samples = 500
# generate samples
samples = generate_latent_points(n_dim, n_samples)
# summarize
print(samples.shape, samples.mean(), samples.std())

# %%
'''
Running the example summarizes the generation of 500 points, each comprised of 100 random
Gaussian values with a mean close to zero and a standard deviation close to 1, e.g. a standard
Gaussian distribution.

(500, 100) -0.004791256735601787 0.9976912528950904
'''
