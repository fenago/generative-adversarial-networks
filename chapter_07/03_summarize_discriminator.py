# %%
'''
## How to Define and Train the Discriminator Model
The first step is to define the discriminator model. The model must take a sample image from
our dataset as input and output a classification prediction as to whether the sample is real or
fake. This is a binary classification problem:

- Inputs: Image with one channel and 28 × 28 pixels in size.
- Outputs: Binary classification, likelihood the sample is real (or fake).

The discriminator model has two convolutional layers with 64 filters each, a small kernel size
of 3, and larger than normal stride of 2. The model has no pooling layers and a single node in
the output layer with the sigmoid activation function to predict whether the input sample is real7.3. How to Define and Train the Discriminator Model 99
or fake. The model is trained to minimize the binary cross-entropy loss function, appropriate
for binary classification. We will use some best practices in defining the discriminator model,
such as the use of LeakyReLU instead of ReLU, using Dropout, and using the Adam version of
stochastic gradient descent with a learning rate of 0.0002 and a momentum of 0.5. The function
define discriminator() below defines the discriminator model and parametrizes the size of
the input image.
'''

# %%
# example of defining the discriminator model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model

# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define model
model = define_discriminator()
# summarize the model
model.summary()
# plot the model
plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)

# %%
from PIL import Image
from IPython.display import display # to display images

image = Image.open('discriminator_plot.png')
display(image)

# %%
'''
Running the example first summarizes the model architecture, showing the input and output
from each layer. We can see that the aggressive 2 × 2 stride acts to downsample the input
image, first from 28 × 28 to 14 × 14, then to 7 × 7, before the model makes an output prediction.
This pattern is by design as we do not use pooling layers and use the large stride as to achieve
a similar downsampling effect. We will see a similar pattern, but in reverse, in the generator
model in the next section.
'''
