# %%
'''
## How to Develop an InfoGAN for MNIST
In this section, we will take a closer look at the generator (g), discriminator (d), and auxiliary
models (q) and how to implement them in Keras. We will develop an InfoGAN implementation
for the MNIST dataset (described in Section 7.2), as was done in the InfoGAN paper. The
paper explores two versions; the first uses just categorical control codes and allows the model to
map one categorical variable to approximately one digit (although there is no ordering of the
digits by categorical variables)

The paper also explores a version of the InfoGAN architecture with the one hot encoded
categorical variable (c1) and two continuous control variables (c2 and c3). The first continuous
variable is discovered to control the rotation of the digits and the second controls the thickness
of the digits.
'''


# %%
# create and plot the infogan model for mnist
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.initializers import RandomNormal
from keras.utils.vis_utils import plot_model

# %%
'''
Next, we can define the discriminator and auxiliary models. The discriminator model is
trained in a standalone manner on real and fake images, as per a normal GAN. Neither the
generator nor the auxiliary models are fit directly; instead, they are fit as part of a composite
model. Both the discriminator and auxiliary models share the same input and feature extraction
layers but differ in their output layers. Therefore, it makes sense to define them both at the same
time. Again, there are many ways that this architecture could be implemented, but defining the
discriminator and auxiliary models as separate models first allows us later to combine them into
a larger GAN model directly via the functional API.

The define discriminator() function below defines the discriminator and auxiliary models
and takes the cardinality of the categorical variable (e.g. number of values, such as 10) as an
input. The shape of the input image is also parameterized as a function argument and set
to the default value of the size of the MNIST images. The feature extraction layers involve
two downsampling layers, used instead of pooling layers as a best practice. Also following best
practice for DCGAN models, we use the LeakyReLU activation and batch normalization.
The discriminator model (d) has a single output node and predicts the probability of an
input image being real via the sigmoid activation function. The model is compiled as it will be
used in a standalone way, optimizing the binary cross-entropy function via the Adam version
of stochastic gradient descent with best practice learning rate and momentum. The auxiliary
model (q) has one node output for each value in the categorical variable and uses a softmax
activation function. A fully connected layer is added between the feature extraction layers and
the output layer, as was used in the InfoGAN paper. The model is not compiled as it is not for
or used in a standalone manner.
'''

# %%
# define the standalone discriminator model
def define_discriminator(n_cat, in_shape=(28,28,1)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=in_shape)
	# downsample to 14x14
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.1)(d)
	# downsample to 7x7
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = LeakyReLU(alpha=0.1)(d)
	d = BatchNormalization()(d)
	# normal
	d = Conv2D(256, (4,4), padding='same', kernel_initializer=init)(d)
	d = LeakyReLU(alpha=0.1)(d)
	d = BatchNormalization()(d)
	# flatten feature maps
	d = Flatten()(d)
	# real/fake output
	out_classifier = Dense(1, activation='sigmoid')(d)
	# define d model
	d_model = Model(in_image, out_classifier)
	# compile d model
	d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
	# create q model layers
	q = Dense(128)(d)
	q = BatchNormalization()(q)
	q = LeakyReLU(alpha=0.1)(q)
	# q model output
	out_codes = Dense(n_cat, activation='softmax')(q)
	# define q model
	q_model = Model(in_image, out_codes)
	return d_model, q_model


# %%
'''
Let’s start off by developing the generator model as a deep convolutional neural network (e.g.
a DCGAN). The model could take the noise vector (z) and control vector (c) as separate inputs
and concatenate them before using them as the basis for generating the image. Alternately,
the vectors can be concatenated beforehand and provided to a single input layer in the model.
The approaches are equivalent and we will use the latter in this case to keep the model simple.
The define generator() function below defines the generator model and takes the size of the
input vector as an argument.

A fully connected layer takes the input vector and produces a sufficient number of activations
to create 512 7 × 7 feature maps from which the activations are reshaped. These then pass
through a normal convolutional layer with 1×1 stride, then two subsequent upsampling transpose
convolutional layers with a 2 × 2 stride first to 14 × 14 feature maps then to the desired 1
channel 28 × 28 feature map output with pixel values in the range [-1,-1] via a Tanh activation
function. Good generator configuration heuristics are as follows, including a random Gaussian
weight initialization, ReLU activations in the hidden layers, and use of batch normalization.
'''

# %%
# define the standalone generator model
def define_generator(gen_input_size):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image generator input
	in_lat = Input(shape=(gen_input_size,))
	# foundation for 7x7 image
	n_nodes = 512 * 7 * 7
	gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
	gen = Activation('relu')(gen)
	gen = BatchNormalization()(gen)
	gen = Reshape((7, 7, 512))(gen)
	# normal
	gen = Conv2D(128, (4,4), padding='same', kernel_initializer=init)(gen)
	gen = Activation('relu')(gen)
	gen = BatchNormalization()(gen)
	# upsample to 14x14
	gen = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(gen)
	gen = Activation('relu')(gen)
	gen = BatchNormalization()(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(gen)
	# tanh output
	out_layer = Activation('tanh')(gen)
	# define model
	model = Model(in_lat, out_layer)
	return model



# %%
'''
Next, we can define the composite GAN model. This model uses all submodels and is
the basis for training the weights of the generator model. The define gan() function below
implements this and defines and returns the model, taking the three submodels as input. The
discriminator is trained in a standalone manner as mentioned, therefore all weights of the
discriminator are set as not trainable (in this context only). The output of the generator model
is connected to the input of the discriminator model, and to the input of the auxiliary model.

This creates a new composite model that takes a [noise + control] vector as input, that
then passes through the generator to produce an image. The image then passes through the
discriminator model to produce a classification and through the auxiliary model to produce a
prediction of the control variables. The model has two output layers that need to be trained
with different loss functions. Binary cross-entropy loss is used for the discriminator output, as
we did when compiling the discriminator for standalone use, and mutual information loss is
used for the auxiliary model, which, in this case, can be implemented directly as categorical
cross-entropy and achieve the desired result.
'''

# %%
# define the combined discriminator, generator and q network model
def define_gan(g_model, d_model, q_model):
	# make weights in the discriminator (some shared with the q model) as not trainable
	d_model.trainable = False
	# connect g outputs to d inputs
	d_output = d_model(g_model.output)
	# connect g outputs to q inputs
	q_output = q_model(g_model.output)
	# define composite model
	model = Model(g_model.input, [d_output, q_output])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=opt)
	return model

# number of values for the categorical control code
n_cat = 10
# size of the latent space
latent_dim = 62
# create the discriminator
d_model, q_model = define_discriminator(n_cat)
# create the generator
gen_input_size = latent_dim + n_cat
g_model = define_generator(gen_input_size)
# create the gan
gan_model = define_gan(g_model, d_model, q_model)
# plot the model
plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)

# %%
from PIL import Image
from IPython.display import display # to display images

image = Image.open('gan_plot.png')
display(image)

# %%
'''
Running the example creates all three models, then creates the composite GAN model and
saves a plot of the model architecture.

Note: Creating a plot of the model assumes that the pydot and graphviz libraries are
installed. If this is a problem, you can comment out the import statement and the function call
for plot model().

The plot shows all of the detail for the generator model and the compressed description
of the discriminator and auxiliary models. Importantly, note the shape of the output of the
discriminator as a single node for predicting whether the image is real or fake, and the 10
nodes for the auxiliary model to predict the categorical control code. Recall that this composite
model will only be used to update the model weights of the generator and auxiliary models,
and all weights in the discriminator model will remain untrainable, i.e. only updated when the
standalone discriminator model is updated.
'''