# %%
'''
## Simple Generator Model With the UpSampling2D Layer
The UpSampling2D layer is simple and effective, although does not perform any learning. It
is not able to fill in useful detail in the upsampling operation. To be useful in a GAN, each
UpSampling2D layer must be followed by a Conv2D layer that will learn to interpret the doubled
input and be trained to translate it into meaningful detail. We can demonstrate this with an
example.
In this example, our little GAN generator model must produce a 10 × 10 image as output
and take a 100 element vector of random numbers from the latent space as input. First, a Dense
fully connected layer can be used to interpret the input vector and create a sufficient number of
activations (outputs) that can be reshaped into a low-resolution version of our output image, in
this case, 128 versions of a 5 × 5 image.
'''

# %%
# example of using upsampling in a simple generator model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras.layers import Conv2D
# define model
model = Sequential()
# define input shape, output enough activations for for 128 5x5 image
model.add(Dense(128 * 5 * 5, input_dim=100))
# reshape vector of activations into 128 feature maps with 5x5
model.add(Reshape((5, 5, 128)))
# double input from 128 5x5 to 1 10x10 feature map
model.add(UpSampling2D())

# %%
'''
Finally, the upsampled feature maps can be interpreted and filled in with hopefully useful
detail by a Conv2D layer. The Conv2D has a single feature map as output to create the single
image we require.
'''

# %%
# fill in detail in the upsampled feature maps and output a single image
model.add(Conv2D(1, (3,3), padding='same'))
# summarize model
model.summary()


# %%
'''
Running the example creates the model and summarizes the output shape of each layer. We
can see that the Dense layer outputs 3,200 activations that are then reshaped into 128 feature
maps with the shape 5×5. The widths and heights are doubled to 10×10 by the UpSampling2D
layer, resulting in a feature map with quadruple the area. Finally, the Conv2D processes these
feature maps and adds in detail, outputting a single 10 × 10 image
'''