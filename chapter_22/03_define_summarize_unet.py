# %%
'''
## How to Implement the U-Net Generator Model
The generator model for the Pix2Pix GAN is implemented as a U-Net. The U-Net model is an
encoder-decoder model for image translation where skip connections are used to connect layers
in the encoder with corresponding layers in the decoder that have the same sized feature maps.
The encoder part of the model is comprised of convolutional layers that use a 2 × 2 stride to
downsample the input source image down to a bottleneck layer. The decoder part of the model
reads the bottleneck output and uses transpose convolutional layers to upsample to the required
output image size
'''

# %%


# %%
'''
Skip connections are added between the layers with the same sized feature maps so that the
first downsampling layer is connected with the last upsampling layer, the second downsampling
layer is connected with the second last upsampling layer, and so on. The connections concatenate
the channels of the feature map in the downsampling layer with the feature map in the upsampling
layer.

Unlike traditional generator models in the GAN architecture, the U-Net generator does not
take a point from the latent space as input. Instead, dropout layers are used as a source of
randomness both during training and when the model is used to make a prediction, e.g. generate
an image at inference time. Similarly, batch normalization is used in the same way during
training and inference, meaning that statistics are calculated for each batch and not fixed at
the end of the training process. This is referred to as instance normalization, specifically when
the batch size is set to 1 as it is with the Pix2Pix model.
'''

# %%
'''
Running the example first summarizes the model. The output of the model summary was
omitted here for brevity. The model has a single input and output, but the skip connections
make the summary difficult to read. A plot of the model is created showing much the same
information in a graphical form. The model is complex, and the plot helps to understand the
skip connections and their impact on the number of filters in the decoder.
'''

# %%
# example of defining a u-net encoder-decoder generator model
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g

# define the standalone generator model
def define_generator(image_shape=(256,256,3)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model: C64-C128-C256-C512-C512-C512-C512-C512
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model

# define image shape
image_shape = (256,256,3)
# create the model
model = define_generator(image_shape)
# summarize the model
model.summary()
# plot the model
plot_model(model, to_file='generator_model_plot.png', show_shapes=True, show_layer_names=True)

# %%
from PIL import Image
from IPython.display import display # to display images

image = Image.open('generator_model_plot.png')
display(image)

# %%
'''
Running the example first summarizes the model. The output of the model summary was
omitted here for brevity. The model has a single input and output, but the skip connections
make the summary difficult to read. A plot of the model is created showing much the same
information in a graphical form. The model is complex, and the plot helps to understand the
skip connections and their impact on the number of filters in the decoder.
'''
