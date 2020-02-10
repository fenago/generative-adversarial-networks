# %%
'''
## How to Define and Use the Generator Model
The generator model is responsible for creating new, fake, but plausible small photographs of
objects. It does this by taking a point from the latent space as input and outputting a square
color image. The latent space is an arbitrarily defined vector space of Gaussian-distributed
values, e.g. 100 dimensions. It has no meaning, but by drawing points from this space randomly
and providing them to the generator model during training, the generator model will assign
meaning to the latent points and, in turn, the latent space, until, at the end of training, the
latent vector space represents a compressed representation of the output space, CIFAR-10
images, that only the generator knows how to turn into plausible CIFAR-10 images.
❼ Inputs: Point in latent space, e.g. a 100-element vector of Gaussian random numbers.
❼ Outputs: Two-dimensional square color image (3 channels) of 32 × 32 pixels with pixel
values in [-1,1].

We don’t have to use a 100 element vector as input; it is a round number and widely used,
but I would expect that 10, 50, or 500 would work just as well. Developing a generator model
requires that we transform a vector from the latent space with, 100 dimensions to a 2D array
with 32 × 32 × 3, or 3,072 values. There are a number of ways to achieve this, but there is one
approach that has proven effective on deep convolutional generative adversarial networks. It
involves two main elements. The first is a Dense layer as the first hidden layer that has enough
nodes to represent a low-resolution version of the output image. Specifically, an image half the
size (one quarter the area) of the output image would be 16x16x3, or 768 nodes, and an image
one quarter the size (one eighth the area) would be 8 × 8 × 3, or 192 nodes.

With some experimentation, I have found that a smaller low-resolution version of the image
works better. Therefore, we will use 4×4×3, or 48 nodes. We don’t just want one low-resolution
version of the image; we want many parallel versions or interpretations of the input. This is
a pattern in convolutional neural networks where we have many parallel filters resulting in
multiple parallel activation maps, called feature maps, with different interpretations of the input.
We want the same thing in reverse: many parallel versions of our output with different learned
features that can be collapsed in the output layer into a final image. The model needs space
to invent, create, or generate. Therefore, the first hidden layer, the Dense layer, needs enough
nodes for multiple versions of our output image, such as 256.
'''

# %%
# example of defining and using the generator model
from numpy import zeros
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
%matplotlib notebook
from matplotlib import pyplot

# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 4x4 image
	n_nodes = 256 * 4 * 4
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((4, 4, 256)))
	# upsample to 8x8
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 16x16
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 32x32
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# output layer
	model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
	return model

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y

# size of the latent space
latent_dim = 100
# define the discriminator model
model = define_generator(latent_dim)
# generate samples
n_samples = 49
X, _ = generate_fake_samples(model, latent_dim, n_samples)
# scale pixel values from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the generated samples
for i in range(n_samples):
	# define subplot
	pyplot.subplot(7, 7, 1 + i)
	# turn off axis labels
	pyplot.axis('off')
	# plot single image
	pyplot.imshow(X[i])
# show the figure
pyplot.show()