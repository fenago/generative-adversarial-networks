# %%
'''
## Downsample Using Strided Convolutions
The discriminator model is a standard convolutional neural network model that takes an image
as input and must output a binary classification as to whether it is real or fake. It is standard
practice with deep convolutional networks to use pooling layers to downsample the input and
feature maps with the depth of the network. This is not recommended for the DCGAN, and
instead, they recommend downsampling using strided convolutions.

This involves defining a convolutional layer as per normal, but instead of using the default
two-dimensional stride of (1,1) to change it to (2,2). This has the effect of downsampling the
input, specifically halving the width and height of the input, resulting in output feature maps with
one quarter the area. The example below demonstrates this with a single hidden convolutional
layer that uses downsampling strided convolutions by setting the strides argument to (2,2).
The effect is the model will downsample the input from 64 × 64 to 32 × 32.
'''

# %%
# example of downsampling with strided convolutions
from keras.models import Sequential
from keras.layers import Conv2D
# define model
model = Sequential()
model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=(64,64,3)))
# summarize model
model.summary()

# %%
'''
Running the example shows the shape of the output of the convolutional layer, where the
feature maps have one quarter of the area.
'''