# %%
'''
## How to Implement the CycleGAN Generator Model
The CycleGAN Generator model takes an image as input and generates a translated image as
output. The model uses a sequence of downsampling convolutional blocks to encode the input
image, a number of residual network (ResNet) convolutional blocks to transform the image, and
a number of upsampling convolutional blocks to generate the output image.

Let c7s1-k denote a 7 × 7 Convolution-InstanceNormReLU layer with k filters and
stride 1. dk denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters
and stride 2. Reflection padding was used to reduce artifacts. Rk denotes a residual
block that contains two 3 × 3 convolutional layers with the same number of filters on
both layer. uk denotes a 3 × 3 fractional-strided-ConvolutionInstanceNorm-ReLU
layer with k filters and stride 1 2.

First, we need a function to define the ResNet blocks. These are blocks comprised of two
3 × 3 CNN layers where the input to the block is concatenated to the output of the block,
channel-wise. This is implemented in the resnet block() function that creates two ConvInstanceNorm blocks with 3 × 3 filters and 1 × 1 stride and without a ReLU activation after the
second block, matching the official Torch implementation in the build conv block() function.
Same padding is used instead of reflection padded recommended in the paper for simplicity.
# generator a resnet block
'''

# %%
# example of an encoder-decoder generator for the cyclegan
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Activation
from keras.initializers import RandomNormal
from keras.layers import Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.utils.vis_utils import plot_model

# generator a resnet block
def resnet_block(n_filters, input_layer):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# first layer convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# second convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	# concatenate merge channel-wise with input layer
	g = Concatenate()([g, input_layer])
	return g

# define the standalone generator model
def define_generator(image_shape=(256,256,3), n_resnet=9):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# c7s1-64
	g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d128
	g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d256
	g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# R256
	for _ in range(n_resnet):
		g = resnet_block(256, g)
	# u128
	g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# u64
	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# c7s1-3
	g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model

# create the model
model = define_generator()
# summarize the model
model.summary()
# plot the model
plot_model(model, to_file='generator_model_plot.png', show_shapes=True, show_layer_names=True)

# %%
from PIL import Image
from IPython.display import display # to display images

image = Image.open('generator_model_plot.png')
display(image)