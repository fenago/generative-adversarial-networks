# %%
'''
## Use Gaussian Weight Initialization
Before a neural network can be trained, the model weights (parameters) must be initialized
to small random variables. The best practice for DCAGAN models reported in the paper is
to initialize all weights using a zero-centered Gaussian distribution (the normal or bell-shaped
distribution) with a standard deviation of 0.02. The example below demonstrates defining a
random Gaussian weight initializer with a mean of 0 and a standard deviation of 0.02 for use in
a transpose convolutional layer in a generator model. The same weight initializer instance could
be used for each layer in a given model.
'''

# %%
# example of gaussian weight initialization in a generator model
from keras.models import Sequential
from keras.layers import Conv2DTranspose
from keras.initializers import RandomNormal
# define model
model = Sequential()
init = RandomNormal(mean=0.0, stddev=0.02)
model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=(64,64,3)))