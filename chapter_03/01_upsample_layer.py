# %%
'''
## Worked Example Using the UpSampling2D Layer
The Keras deep learning library provides this capability in a layer called UpSampling2D. It can
be added to a convolutional neural network and repeats the rows and columns provided as input
in the output. 
'''

# %%
# example of using the upsampling layer
from numpy import asarray
from keras.models import Sequential
from keras.layers import UpSampling2D

# %%
'''
We can demonstrate the behavior of this layer with a simple contrived example. First, we
can define a contrived input image that is 2 × 2 pixels. We can use specific values for each pixel
so that after upsampling, we can see exactly what effect the operation had on the input.

Once the image is defined, we must add a channel dimension (e.g. grayscale) and also a
sample dimension (e.g. we have 1 sample) so that we can pass it as input to the model. The
data dimensions in order are: samples, rows, columns, and channels.
'''

# %%
# define input data
X = asarray([[1, 2],
			 [3, 4]])
# show input data for context
print(X)
# reshape input data into one sample a sample with a channel
X = X.reshape((1, 2, 2, 1))
# define model
model = Sequential()
model.add(UpSampling2D(input_shape=(2, 2, 1)))
# summarize the model
model.summary()

# %%
'''
We can then use the model to make a prediction, that is upsample a provided input image.

The output will have four dimensions, like the input, therefore, we can convert it back to a
2 × 2 array to make it easier to review the result.
'''

# %%
# make a prediction with the model
yhat = model.predict(X)
# reshape output to remove channel to make printing easier
yhat = yhat.reshape((4, 4))
# summarize output
print(yhat)

# %%
'''
Running the example first creates and summarizes our 2 × 2 input data. Next, the model is
summarized. We can see that it will output a 4 × 4 result as we expect, and importantly, the
layer has no parameters or model weights. This is because it is not learning anything; it is just
doubling the input. Finally, the model is used to upsample our input, resulting in a doubling of
each row and column for our input data, as we expected
'''