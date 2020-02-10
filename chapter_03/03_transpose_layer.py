# %%
'''
## Worked Example Using the Conv2DTranspose Layer
Keras provides the transpose convolution capability via the Conv2DTranspose layer. It can be
added to your model directly; for example:
'''

# %%
# example of using the transpose convolutional layer
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2DTranspose
# define input data
X = asarray([[1, 2],
			 [3, 4]])
# show input data for context
print(X)
# reshape input data into one sample a sample with a channel
X = X.reshape((1, 2, 2, 1))

# %%
'''
We can now define our model. The model has only the Conv2DTranspose layer, which
takes 2 × 2 grayscale images as input directly and outputs the result of the operation. The
Conv2DTranspose both upsamples and performs a convolution. As such, we must specify both
the number of filters and the size of the filters as we do for Conv2D layers. Additionally, we
must specify a stride of (2,2) because the upsampling is achieved by the stride behavior of the
convolution on the input. Specifying a stride of (2,2) has the effect of spacing out the input.
Specifically, rows and columns of 0.0 values are inserted to achieve the desired stride. In this
example, we will use one filter, with a 1 × 1 kernel and a stride of 2 × 2 so that the 2 × 2 input
image is upsampled to 4 × 4.
'''

# %%
# define model
model = Sequential()
model.add(Conv2DTranspose(1, (1,1), strides=(2,2), input_shape=(2, 2, 1)))
# summarize the model
model.summary()

# %%
'''
To make it clear what the Conv2DTranspose layer is doing, we will fix the single weight in
the single filter to the value of 1.0 and use a bias value of 0.0. These weights, along with a
kernel size of (1,1) will mean that values in the input will be multiplied by 1 and output as-is,
and the 0 values in the new rows and columns added via the stride of 2 × 2 will be output as 0
(e.g. 1 × 0 in each case).
'''

# %%
# define weights that they do nothing
weights = [asarray([[[[1]]]]), asarray([0])]
# store the weights in the model
model.set_weights(weights)


# %%
'''
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
layer two parameters or model weights. One for the single 1×1 filter and one for the bias. Unlike
the UpSampling2D layer, the Conv2DTranspose will learn during training and will attempt to
fill in detail as part of the upsampling process. Finally, the model is used to upsample our input.
We can see that the calculations of the cells that involve real values as input result in the real
value as output (e.g. 1 × 1, 1 × 2, etc.). We can see that where new rows and columns have
been inserted by the stride of 2 × 2, that their 0.0 values multiplied by the 1.0 values in the
single 1 × 1 filter have resulted in 0 values in the output.
'''
