# %%
'''
## How to Train the Generator Model
The weights in the generator model are updated based on the performance of the discriminator
model. When the discriminator is good at detecting fake samples, the generator is updated more,
and when the discriminator model is relatively poor or confused when detecting fake samples,
the generator model is updated less. This defines the zero-sum or adversarial relationship
between these two models. There may be many ways to implement this using the Keras API,
but perhaps the simplest approach is to create a new model that combines the generator and
discriminator models.

Specifically, a new GAN model can be defined that stacks the generator and discriminator
such that the generator receives as input random points in the latent space and generates8.5. How to Train the Generator Model 151
samples that are fed into the discriminator model directly, classified, and the output of this
larger model can be used to update the model weights of the generator. To be clear, we are not
talking about a new third model, just a new logical model that uses the already-defined layers
and weights from the standalone generator and discriminator models. Only the discriminator
is concerned with distinguishing between real and fake examples, therefore the discriminator
model can be trained in a standalone manner on examples of each, as we did in the section on
the discriminator model above.

The generator model is only concerned with the discriminator’s performance on fake examples.
Therefore, we will mark all of the layers in the discriminator as not trainable when it is part
of the GAN model so that they cannot be updated and overtrained on fake examples. When
training the generator via this logical GAN model, there is one more important change. We
want the discriminator to think that the samples output by the generator are real, not fake.
Therefore, when the generator is trained as part of the GAN model, we will mark the generated
samples as real (class = 1).

Why would we want to do this? We can imagine that the discriminator will then classify
the generated samples as not real (class = 0) or a low probability of being real (0.3 or 0.5). The
backpropagation process used to update the model weights will see this as a large error and will
update the model weights (i.e. only the weights in the generator) to correct for this error, in
turn making the generator better at generating good fake samples. Let’s make this concrete.
❼ Inputs: Point in latent space, e.g. a 100-element vector of Gaussian random numbers.
❼ Outputs: Binary classification, likelihood the sample is real (or fake).

The define gan() function below takes as arguments the already-defined generator and
discriminator models and creates the new, logical third model subsuming these two models.
The weights in the discriminator are marked as not trainable, which only affects the weights as
seen by the GAN model and not the standalone discriminator model. The GAN model then
uses the same binary cross-entropy loss function as the discriminator and the efficient Adam
version of stochastic gradient descent with the learning rate of 0.0002 and momentum of 0.5,
recommended when training deep convolutional GANs.
'''

# %%
# demonstrate creating the three models in the gan
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.utils.vis_utils import plot_model

# define the standalone discriminator model
def define_discriminator(in_shape=(32,32,3)):
	model = Sequential()
	# normal
	model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# classifier
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

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

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# summarize gan model
gan_model.summary()
# plot gan model
plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)

# %%
from PIL import Image
from IPython.display import display # to display images

image = Image.open('gan_plot.png')
display(image)