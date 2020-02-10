# %%
'''
## Single Discriminator Model With Multiple Outputs
Another approach to implementing the semi-supervised discriminator model is to have a single
model with multiple output layers. Specifically, this is a single model with one output layer for
the unsupervised task and one output layer for the supervised task. This is like having separate
models for the supervised and unsupervised tasks in that they both share the same feature
extraction layers, except that in this case, each input image always has two output predictions,
specifically a real/fake prediction and a supervised class prediction.
A problem with this approach is that when the model is updated with unlabeled and
generated images, there is no supervised class label. In that case, these images must have an
output label of unknown or fake from the supervised output. This means that an additional
class label is required for the supervised output layer. The example below implements the
multi-output single model approach for the discriminator model in the semi-supervised GAN
architecture. We can see that the model is defined with two output layers and that the output
layer for the supervised task is defined with n classes + 1, in this case 11, making room for
the additional unknown class label. We can also see that the model is compiled to two loss
functions, one for each output layer of the model.
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
	# supervised output
	c_out_layer = Dense(n_classes + 1, activation='softmax')(fe)
	# define and compile supervised discriminator model
	model = Model(in_image, [d_out_layer, c_out_layer])
	model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
	return model

# create model
model = define_discriminator()
# plot the model
plot_model(model, to_file='multioutput_discriminator_plot.png', show_shapes=True, show_layer_names=True)

# %%
from PIL import Image
from IPython.display import display # to display images

image = Image.open('multioutput_discriminator_plot.png')
display(image)