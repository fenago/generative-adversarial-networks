# %%
'''
## Use Batch Normalization
Batch normalization standardizes the activations from a prior layer to have a zero mean and
unit variance. This has the effect of stabilizing the training process. Batch normalization is used
after the activation of convolution and transpose convolutional layers in the discriminator and
generator models respectively. It is added to the model after the hidden layer, but before the
activation, such as LeakyReLU. The example below demonstrates adding a BatchNormalization
layer after a Conv2D layer in a discriminator model but before the activation.
'''

# %%
# example of using batch norm in a discriminator model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
# define model
model = Sequential()
model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=(64,64,3)))
model.add(BatchNormalization())
model.add(LeakyReLU(0.2))
# summarize model
model.summary()

# %%
'''
Running the example shows the desired usage of batch norm between the outputs of the
convolutional layer and the activation function.
'''