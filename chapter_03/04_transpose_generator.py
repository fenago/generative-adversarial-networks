# %%
'''
## Simple Generator Model With the Conv2DTranspose Layer
The Conv2DTranspose is more complex than the UpSampling2D layer, but it is also effective
when used in GAN models, specifically the generator model. Either approach can be used,
although the Conv2DTranspose layer is preferred, perhaps because of the simpler generator
models and possibly better results, although GAN performance and skill is notoriously difficult
to quantify. We can demonstrate using the Conv2DTranspose layer in a generator model with
another simple example.

In this case, our little GAN generator model must produce a 10 × 10 image and take a
100-element vector from the latent space as input, as in the previous UpSampling2D example.
First, a Dense fully connected layer can be used to interpret the input vector and create a
sufficient number of activations (outputs) that can be reshaped into a low-resolution version of
our output image, in this case, 128 versions of a 5 × 5 image.
'''

# %%
# example of using transpose conv in a simple generator model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2DTranspose
# define model
model = Sequential()
# define input shape, output enough activations for for 128 5x5 image
model.add(Dense(128 * 5 * 5, input_dim=100))
# reshape vector of activations into 128 feature maps with 5x5
model.add(Reshape((5, 5, 128)))
# double input from 128 5x5 to 1 10x10 feature map
model.add(Conv2DTranspose(1, (3,3), strides=(2,2), padding='same'))
# summarize model
model.summary()

# %%
'''
Running the example creates the model and summarizes the output shape of each layer.
We can see that the Dense layer outputs 3,200 activations that are then reshaped into 128
feature maps with the shape 5 × 5. The widths and heights are doubled to 10 × 10 by the
Conv2DTranspose layer resulting in a single feature map with quadruple the area.
'''

