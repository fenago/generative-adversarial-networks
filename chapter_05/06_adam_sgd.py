# %%
'''
## Use Adam Stochastic Gradient Descent
Stochastic gradient descent, or SGD for short, is the standard algorithm used to optimize
the weights of convolutional neural network models. There are many variants of the training
algorithm. The best practice for training DCGAN models is to use the Adam version of
stochastic gradient descent with the learning rate lr of 0.0002 and the beta1 momentum value
of 0.5 instead of the default of 0.9. The Adam optimization algorithm with this configuration
is recommended when both optimizing the discriminator and generator models. The example
below demonstrates configuring the Adam stochastic gradient descent optimization algorithm
for training a discriminator model.
'''

# %%
# example of using adam when training a discriminator model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
# define model
model = Sequential()
model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=(64,64,3)))
# compile model
opt = Adam(lr=0.0002, beta_1=0.5)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])