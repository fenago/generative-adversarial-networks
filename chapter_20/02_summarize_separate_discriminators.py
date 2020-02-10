# %%
'''
## Separate Discriminator Models With Shared Weights
Starting with the standard GAN discriminator model, we can update it to create two models
that share feature extraction weights. Specifically, we can define one classifier model that
predicts whether an input image is real or fake, and a second classifier model that predicts the
class of a given model.

❼ Binary Classifier Model. Predicts whether the image is real or fake, sigmoid activation
function in the output layer, and optimized using the binary cross-entropy loss function.
❼ Multiclass Classifier Model. Predicts the class of the image, softmax activation
function in the output layer, and optimized using the categorical cross-entropy loss
function.

Both models have different output layers but share all feature extraction layers. This means
that updates to one of the classifier models will impact both models. The example below creates
the traditional discriminator model with binary output first, then re-uses the feature extraction
layers and creates a new multiclass prediction model, in this case with 10 classes
'''

# %%
# example of defining semi-supervised discriminator model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

# define the standalone supervised and unsupervised discriminator models
def define_discriminator(in_shape=(28,28,1), n_classes=10):
	# image input
	in_image = Input(shape=in_shape)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(in_image)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# unsupervised output
	d_out_layer = Dense(1, activation='sigmoid')(fe)
	# define and compile unsupervised discriminator model
	d_model = Model(in_image, d_out_layer)
	d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
	# supervised output
	c_out_layer = Dense(n_classes, activation='softmax')(fe)
	# define and compile supervised discriminator model
	c_model = Model(in_image, c_out_layer)
	c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
	return d_model, c_model

# create model
d_model, c_model = define_discriminator()
# plot the model
plot_model(d_model, to_file='discriminator1_plot.png', show_shapes=True, show_layer_names=True)
plot_model(c_model, to_file='discriminator2_plot.png', show_shapes=True, show_layer_names=True)

# %%
from PIL import Image
from IPython.display import display # to display images

image = Image.open('discriminator1_plot.png')
display(image)

# %%
from PIL import Image
from IPython.display import display # to display images

image = Image.open('discriminator2_plot.png')
display(image)