# %%
'''
## Use Leaky ReLU
The rectified linear activation unit, or ReLU for short, is a simple calculation that returns the
value provided as input directly, or the value 0.0 if the input is 0.0 or less. It has become a best
practice when developing deep convolutional neural networks generally. The best practice for
GANs is to use a variation of the ReLU that allows some values less than zero and learns where
the cut-off should be in each node. This is called the leaky rectified linear activation unit, or
the LeakyReLU layer.

A negative slope can be specified for the LeakyReLU and the default value of 0.2 is recommended. Originally, ReLU was recommend for use in the generator model and LeakyReLU was
recommended for use in the discriminator model, although more recently, the LeakyReLU may
be recommended in both models. The example below demonstrates using the LeakyReLU with
the default slope of 0.2 after a convolutional layer in a discriminator model.
'''

# %%
# example of using leakyrelu in a discriminator model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import LeakyReLU
# define model
model = Sequential()
model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=(64,64,3)))
model.add(LeakyReLU(0.2))
# summarize model
model.summary()

# %%
'''
Running the example demonstrates the structure of the model with a single convolutional
layer followed by the activation layer.
'''
