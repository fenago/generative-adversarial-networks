# %%
'''
## Upsample Using Strided Convolutions
The generator model must generate an output image given a random point from the latent space
as input. The recommended approach for achieving this is to use a transpose convolutional
layer with a strided convolution. This is a special type of layer that performs the convolution
operation in reverse. Intuitively, this means that setting a stride of 2 × 2 will have the opposite
effect, upsampling the input instead of downsampling it in the case of a normal convolutional
layer.

By stacking a transpose convolutional layer with strided convolutions, the generator model is
able to scale a given input to the desired output dimensions. The example below demonstrates
this with a single hidden transpose convolutional layer that uses upsampling strided convolutions
by setting the strides argument to (2,2). The effect is the model will upsample the input from
64 × 64 to 128 × 128.
'''

# %%
# example of upsampling with strided convolutions
from keras.models import Sequential
from keras.layers import Conv2DTranspose
# define model
model = Sequential()
model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', input_shape=(64,64,3)))
# summarize model
model.summary()

# %%
'''
Running the example shows the shape of the output of the convolutional layer, where the
feature maps have quadruple the area.
'''
