<img align="right" src="../logo-small.png">

# Lab : Introduction to neural learning: gradient descent

#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

**Note:** Elev8ed Notebooks (powered by Jupyter) will be accessible at the port given to you by your instructor. Password for jupyterLab : `1234`

All Notebooks are present in `work/generative-adversarial-networks` folder. To copy and paste: use **Control-C** and to paste inside of a terminal, use **Control-V**

You can access jupyter lab at `<host-ip>:<port>/lab/workspaces/`


## How to Upsample with Convolutional Neural Networks


Generative Adversarial Networks, or GANs, are an architecture for training generative models,
such as deep convolutional neural networks for generating images. The GAN architecture is
comprised of both a generator and a discriminator model. The generator is responsible for
creating new outputs, such as images, that plausibly could have come from the original dataset.
The generator model is typically implemented using a deep convolutional neural network and
results-specialized layers that learn to fill in features in an image rather than extract features
from an input image.

Two common types of layers that can be used in the generator model are a upsample
layer that simply doubles the dimensions of the input and the transpose convolutional layer
that performs an inverse convolution operation. In this tutorial, you will discover how to use
Upsampling and Transpose Convolutional Layers in Generative Adversarial Networks when
generating images. After completing this tutorial, you will know:

- Generative models in the GAN arch
itecture are required to upsample input data in order
to generate an output image.
- The Upsampling layer is a simple layer with no weights that will double the dimensions of
input and can be used in a generative model when followed by a traditional convolutional
layer.

- The Transpose Convolutional layer is an inverse convolutional layer that will both upsample
input and learn how to fill in details during the model training process.

Let’s get started.

**Tutorial Overview** 

This tutorial is divided into three parts; they are:
1. Need for Upsampling in GANs

2. How to Use the Upsampling Layer

3. How to Use the Transpose Convolutional Layer


## Need for Upsampling in GANs

Generative Adversarial Networks are an architecture for neural networks for training a generative
model. The architecture is comprised of a generator and a discriminator model, each of which
are implemented as a deep convolutional neural network. The discriminator is responsible
for classifying images as either real (from the domain) or fake (generated). The generator is
responsible for generating new plausible examples from the problem domain. The generator
works by taking a random point from the latent space as input and outputting a complete
image, in a one-shot manner.

A traditional convolutional neural network for image classification, and related tasks, will
use pooling layers to downsample input images. For example, an average pooling or max pooling
layer will reduce the feature maps from a convolutional by half on each dimension, resulting
in an output that is one quarter the area of the input. Convolutional layers themselves also
perform a form of downsampling by applying each filter across the input images or feature maps;
the resulting activations are an output feature map that is smaller because of the border effects.
Often padding is used to counter this effect. The generator model in a GAN requires an inverse
operation of a pooling layer in a traditional convolutional layer. It needs a layer to translate
from coarse salient features to a more dense and detailed output.

A simple version of an unpooling or opposite pooling layer is called an upsampling layer.
It works by repeating the rows and columns of the input. A more elaborate approach is to
perform a backwards convolutional operation, originally referred to as a deconvolution, which is
incorrect, but is more commonly referred to as a fractional convolutional layer or a transposed
convolutional layer. Both of these layers can be used on a GAN to perform the required
upsampling operation to transform a small input into a large image output. In the following
sections, we will take a closer look at each and develop an intuition for how they work so that
we can use them effectively in our GAN models.


## How to Use the Upsampling Layer

Perhaps the simplest way to upsample an input is to double each row and column. For example,
an input image with the shape 2 × 2 would be output as 4 × 4.

```
1, 2
Input = 3, 4
1, 1, 2, 2
Output = 1, 1, 2, 2
3, 3, 4, 4
3, 3, 4, 4
```

## Worked Example Using the UpSampling2D Layer

The Keras deep learning library provides this capability in a layer called UpSampling2D. It can
be added to a convolutional neural network and repeats the rows and columns provided as input
in the output. For example:

```
...
# define model
model = Sequential()
model.add(UpSampling2D())

```

We can demonstrate the behavior of this layer with a simple contrived example. First, we
can define a contrived input image that is 2 × 2 pixels. We can use specific values for each pixel
so that after upsampling, we can see exactly what effect the operation had on the input.

```
...
# define input data
X = asarray([[1, 2],
[3, 4]])
# show input data for context
print(X)

```

Once the image is defined, we must add a channel dimension (e.g. grayscale) and also a
sample dimension (e.g. we have 1 sample) so that we can pass it as input to the model. The
data dimensions in order are: samples, rows, columns, and channels.

```
...
# reshape input data into one sample with one channel
X = X.reshape((1, 2, 2, 1))

```

We can now define our model. The model has only the UpSampling2D layer which takes
2 × 2 grayscale images as input directly and outputs the result of the upsampling operation.

```
...
# define model
model = Sequential()
model.add(UpSampling2D(input_shape=(2, 2, 1)))
# summarize the model
model.summary()

```

We can then use the model to make a prediction, that is upsample a provided input image.

```
...
# make a prediction with the model
yhat = model.predict(X)

```

The output will have four dimensions, like the input, therefore, we can convert it back to a
2 × 2 array to make it easier to review the result.

```
...
# reshape output to remove sample and channel to make printing easier
yhat = yhat.reshape((4, 4))
# summarize output
print(yhat)

```

Tying all of this together, the complete example of using the UpSampling2D layer in Keras
is provided below.

```
# example of using the upsampling layer
from numpy import asarray
from keras.models import Sequential
from keras.layers import UpSampling2D
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
# make a prediction with the model
yhat = model.predict(X)
# reshape output to remove channel to make printing easier
yhat = yhat.reshape((4, 4))
# summarize output
print(yhat)

```

Running the example first creates and summarizes our 2 × 2 input data. Next, the model is
summarized. We can see that it will output a 4 × 4 result as we expect, and importantly, the
layer has no parameters or model weights. This is because it is not learning anything; it is just
doubling the input. Finally, the model is used to upsample our input, resulting in a doubling of
each row and column for our input data, as we expected.

```
[[1 2]
[3 4]]
_________________________________________________________________
Layer (type)
Output Shape
Param #
=================================================================
up_sampling2d_1 (UpSampling2 (None, 4, 4, 1)
0
=================================================================
Total params: 0
Trainable params: 0
Non-trainable params: 0
_________________________________________________________________

[[1.
[1.
[3.
[3.

1.
1.
3.
3.

2.
2.
4.
4.

2.]
2.]
4.]
4.]]
```

By default, the UpSampling2D will double each input dimension. This is defined by the
size argument that is set to the tuple (2,2). You may want to use different factors on each
dimension, such as double the width and triple the height. This could be achieved by setting
the size argument to (2, 3). The result of applying this operation to a 2 × 2 image would be
a 4 × 6 output image (e.g. 2 × 2 and 2 × 3). For example:

```
...
# example of using different scale factors for each dimension
model.add(UpSampling2D(size=(2, 3)))

```

Additionally, by default, the UpSampling2D layer will use a nearest neighbor algorithm to
fill in the new rows and columns. This has the effect of simply doubling rows and columns, as
described and is specified by the interpolation argument set to ‘nearest’. Alternately, a
bilinear interpolation method can be used which draws upon multiple surrounding points. This
can be specified via setting the interpolation argument to ‘bilinear’. For example:

```
...
# example of using bilinear interpolation when upsampling
model.add(UpSampling2D(interpolation=✬bilinear✬))

```


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

```
...
# define model
model = Sequential()
# define input shape, output enough activations for 128 5x5 images
model.add(Dense(128 * 5 * 5, input_dim=100))
# reshape vector of activations into 128 feature maps with 5x5
model.add(Reshape((5, 5, 128)))

```

Next, the 5 × 5 feature maps can be upsampled to a 10 × 10 feature map.

```
...
# quadruple input from 128 5x5 to 1 10x10 feature map
model.add(UpSampling2D())
```

Finally, the upsampled feature maps can be interpreted and filled in with hopefully useful
detail by a Conv2D layer. The Conv2D has a single feature map as output to create the single
image we require.

```
...
# fill in detail in the upsampled feature maps
model.add(Conv2D(1, (3,3), padding=✬same✬))

```

Tying this together, the complete example is listed below.

```
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
# fill in detail in the upsampled feature maps and output a single image
model.add(Conv2D(1, (3,3), padding=✬same✬))
# summarize model
model.summary()

```

Running the example creates the model and summarizes the output shape of each layer. We
can see that the Dense layer outputs 3,200 activations that are then reshaped into 128 feature
maps with the shape 5 × 5. The widths and heights are doubled to 10 × 10 by the UpSampling2D
layer, resulting in a feature map with quadruple the area. Finally, the Conv2D processes these
feature maps and adds in detail, outputting a single 10 × 10 image.

```
_________________________________________________________________
Layer (type)
Output Shape
Param #
=================================================================
dense_1 (Dense)
(None, 3200)
323200
_________________________________________________________________
reshape_1 (Reshape)
(None, 5, 5, 128)
0
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 10, 10, 128) 0
_________________________________________________________________
conv2d_1 (Conv2D)
(None, 10, 10, 1)
1153
=================================================================


Total params: 324,353
Trainable params: 324,353
Non-trainable params: 0
_________________________________________________________________
```




## How to Use the Transpose Convolutional Layer

The transpose convolutional layer is more complex than a simple upsampling layer. A simple
way to think about it is that it both performs the upsample operation and interprets the coarse
input data to fill in the detail while it is upsampling. It is like a layer that combines the
UpSampling2D and Conv2D layers into one layer. This is a crude understanding, but a practical
starting point.

The need for transposed convolutions generally arises from the desire to use a
transformation going in the opposite direction of a normal convolution, i.e., from
something that has the shape of the output of some convolution to something that
has the shape of its input while maintaining a connectivity pattern that is compatible
with said convolution

— A Guide To Convolution Arithmetic For Deep Learning, 2016.

In fact, the transpose convolutional layer performs an inverse convolution operation. Specifically, the forward and backward passes of the convolutional layer are reversed.

One way to put it is to note that the kernel defines a convolution, but whether it’s a
direct convolution or a transposed convolution is determined by how the forward
and backward passes are computed.

— A Guide To Convolution Arithmetic For Deep Learning, 2016.

It is sometimes called a deconvolution or deconvolutional layer and models that use these
layers can be referred to as deconvolutional networks, or deconvnets.

A deconvnet can be thought of as a convnet model that uses the same components
(filtering, pooling) but in reverse, so instead of mapping pixels to features does the
opposite.

— Visualizing and Understanding Convolutional Networks, 2013.

Referring to this operation as a deconvolution is technically incorrect as a deconvolution is a
specific mathematical operation not performed by this layer. In fact, the traditional convolutional
layer does not technically perform a convolutional operation, it performs a cross-correlation.


The deconvolution layer, to which people commonly refer, first appears in Zeiler’s
paper as part of the deconvolutional network but does not have a specific name. [...]
It also has many names including (but not limited to) subpixel or fractional convolutional layer, transposed convolutional layer, inverse, up or backward convolutional
layer.

— Is The Deconvolution Layer The Same As A Convolutional Layer?, 2016.

It is a very flexible layer, although we will focus on its use in generative models for upsampling
an input image. The transpose convolutional layer is much like a normal convolutional layer.
It requires that you specify the number of filters and the kernel size of each filter. The key to
the layer is the stride. Typically, the stride of a convolutional layer is (1 × 1), that is a filter is
moved along one pixel horizontally for each read from left-to-right, then down pixel for the next
row of reads. A stride of 2 × 2 on a normal convolutional layer has the effect of downsampling
the input, much like a pooling layer. In fact, a 2 × 2 stride can be used instead of a pooling
layer in the discriminator model.

The transpose convolutional layer is like an inverse convolutional layer. As such, you would
intuitively think that a 2 × 2 stride would upsample the input instead of downsample, which is
exactly what happens. Stride or strides refers to the manner of a filter scanning across an input
in a traditional convolutional layer. Whereas, in a transpose convolutional layer, stride refers to
the manner in which outputs in the feature map are laid down. This effect can be implemented
with a normal convolutional layer using a fractional input stride (f ), e.g. with a stride of f = 12 .
When inverted, the output stride is set to the numerator of this fraction, e.g. f = 2.

In a sense, upsampling with factor f is convolution with a fractional input stride
of f1 . So long as f is integral, a natural way to upsample is therefore backwards
convolution (sometimes called deconvolution) with an output stride of f .

— Fully Convolutional Networks for Semantic Segmentation, 2014.

One way that this effect can be achieved with a normal convolutional layer is by inserting
new rows and columns of 0.0 values in the input data.

Finally note that it is always possible to emulate a transposed convolution with
a direct convolution. The disadvantage is that it usually involves adding many
columns and rows of zeros to the input ...

— A Guide To Convolution Arithmetic For Deep Learning, 2016.

Let’s make this concrete with an example. Consider an input image with the size 2 × 2 as
follows:

```
1, 2
Input = 3, 4

```

Assuming a single filter with a 1 × 1 kernel and model weights that result in no changes to
the inputs when output (e.g. a model weight of 1.0 and a bias of 0.0), a transpose convolutional
operation with an output stride of 1 × 1 will reproduce the output as-is:

```
1, 2
Output = 3, 4

```

With an output stride of (2,2), the 1 × 1 convolution requires the insertion of additional
rows and columns into the input image so that the reads of the operation can be performed.
Therefore, the input looks as follows:


```
1,
Input = 0,
3,
0,

0,
0,
0,
0,

2,
0,
4,
0,

0
0
0
0

```

The model can then read across this input using an output stride of (2,2) and will output a
4 × 4 image, in this case with no change as our model weights have no effect by design:

```
1,
Output = 0,
3,
0,

0,
0,
0,
0,

2,
0,
4,
0,

0
0
0
0

```


## Worked Example Using the Conv2DTranspose Layer

Keras provides the transpose convolution capability via the Conv2DTranspose layer. It can be
added to your model directly; for example:

```
...
# define model
model = Sequential()
model.add(Conv2DTranspose(

...))

```

We can demonstrate the behavior of this layer with a simple contrived example. First, we
can define a contrived input image that is 2 × 2 pixels, as we did in the previous section. We
can use specific values for each pixel so that after the transpose convolutional operation, we can
see exactly what effect the operation had on the input.

```
...
# define input data
X = asarray([[1, 2],
[3, 4]])
# show input data for context
print(X)

```

Once the image is defined, we must add a channel dimension (e.g. grayscale) and also a
sample dimension (e.g. we have 1 sample) so that we can pass it as input to the model.


```
...
# reshape input data into one sample a sample with a channel
X = X.reshape((1, 2, 2, 1))

```

We can now define our model. The model has only the Conv2DTranspose layer, which
takes 2 × 2 grayscale images as input directly and outputs the result of the operation. The
Conv2DTranspose both upsamples and performs a convolution. As such, we must specify both
the number of filters and the size of the filters as we do for Conv2D layers. Additionally, we
must specify a stride of (2,2) because the upsampling is achieved by the stride behavior of the
convolution on the input. Specifying a stride of (2,2) has the effect of spacing out the input.
Specifically, rows and columns of 0.0 values are inserted to achieve the desired stride. In this
example, we will use one filter, with a 1 × 1 kernel and a stride of 2 × 2 so that the 2 × 2 input
image is upsampled to 4 × 4.

```
...
# define model
model = Sequential()
model.add(Conv2DTranspose(1, (1,1), strides=(2,2), input_shape=(2, 2, 1)))
# summarize the model
model.summary()

```

To make it clear what the Conv2DTranspose layer is doing, we will fix the single weight in
the single filter to the value of 1.0 and use a bias value of 0.0. These weights, along with a
kernel size of (1,1) will mean that values in the input will be multiplied by 1 and output as-is,
and the 0 values in the new rows and columns added via the stride of 2 × 2 will be output as 0
(e.g. 1 × 0 in each case).

```
...
# define weights that do nothing
weights = [asarray([[[[1]]]]), asarray([0])]
# store the weights in the model
model.set_weights(weights)

```

We can then use the model to make a prediction, that is upsample a provided input image.

```
...
# make a prediction with the model
yhat = model.predict(X)

```

The output will have four dimensions, like the input, therefore, we can convert it back to a
2 × 2 array to make it easier to review the result.

```
...
# reshape output to remove channel to make printing easier
yhat = yhat.reshape((4, 4))
# summarize output
print(yhat)

```


Tying all of this together, the complete example of using the Conv2DTranspose layer in
Keras is provided below.

```
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
# define model
model = Sequential()
model.add(Conv2DTranspose(1, (1,1), strides=(2,2), input_shape=(2, 2, 1)))
# summarize the model
model.summary()
# define weights that they do nothing
weights = [asarray([[[[1]]]]), asarray([0])]
# store the weights in the model
model.set_weights(weights)
# make a prediction with the model
yhat = model.predict(X)
# reshape output to remove channel to make printing easier
yhat = yhat.reshape((4, 4))
# summarize output
print(yhat)

```

Running the example first creates and summarizes our 2 × 2 input data. Next, the model is
summarized. We can see that it will output a 4 × 4 result as we expect, and importantly, the
layer two parameters or model weights. One for the single 1 × 1 filter and one for the bias. Unlike
the UpSampling2D layer, the Conv2DTranspose will learn during training and will attempt to
fill in detail as part of the upsampling process. Finally, the model is used to upsample our input.
We can see that the calculations of the cells that involve real values as input result in the real
value as output (e.g. 1 × 1, 1 × 2, etc.). We can see that where new rows and columns have
been inserted by the stride of 2 × 2, that their 0.0 values multiplied by the 1.0 values in the
single 1 × 1 filter have resulted in 0 values in the output.

```
[[1 2]
[3 4]]
_________________________________________________________________
Layer (type)
Output Shape
Param #
=================================================================
conv2d_transpose_1 (Conv2DTr (None, 4, 4, 1)
2
=================================================================
Total params: 2
Trainable params: 2
Non-trainable params: 0
_________________________________________________________________
```

```

[[1.
[0.
[3.
[0.

0.
0.
0.
0.

2.
0.
4.
0.

41

0.]
0.]
0.]
0.]]

```

Recall that this is a contrived case where we artificially specified the model weights so that
we could see the effect of the transpose convolutional operation. In practice, we will use a
large number of filters (e.g. 64 or 128), a larger kernel (e.g. 3 × 3, 5 × 5, etc.), and the layer
will be initialized with random weights that will learn how to effectively upsample with detail
during training. In fact, you might imagine how different sized kernels will result in different
sized outputs, more than doubling the width and height of the input. In this case, the padding
argument of the layer can be set to ‘same’ to force the output to have the desired (doubled)
output shape; for example:

```
...
# example of using padding to ensure that the output are only doubled
model.add(Conv2DTranspose(1, (3,3), strides=(2,2), padding=✬same✬, input_shape=(2, 2, 1)))

```

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

```
...
# define model
model = Sequential()
# define input shape, output enough activations for 128 5x5 images
model.add(Dense(128 * 5 * 5, input_dim=100))
# reshape vector of activations into 128 feature maps with 5x5
model.add(Reshape((5, 5, 128)))

```

Next, the 5 × 5 feature maps can be upsampled to a 10 × 10 feature map. We will use a
3 × 3 kernel size for the single filter, which will result in a slightly larger than doubled width
and height in the output feature map (11 × 11). Therefore, we will set the padding argument
to ‘same’ to ensure the output dimensions are 10 × 10 as required.


```
...
# double input from 128 5x5 to 1 10x10 feature map
model.add(Conv2DTranspose(1, (3,3), strides=(2,2), padding=✬same✬))

```

Tying this together, the complete example is listed below.

```
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
model.add(Conv2DTranspose(1, (3,3), strides=(2,2), padding=✬same✬))
# summarize model
model.summary()

```

Running the example creates the model and summarizes the output shape of each layer.
We can see that the Dense layer outputs 3,200 activations that are then reshaped into 128
feature maps with the shape 5 × 5. The widths and heights are doubled to 10 × 10 by the
Conv2DTranspose layer resulting in a single feature map with quadruple the area.

```
_________________________________________________________________
Layer (type)
Output Shape
Param #
=================================================================
dense_1 (Dense)
(None, 3200)
323200
_________________________________________________________________
reshape_1 (Reshape)
(None, 5, 5, 128)
0
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 10, 10, 1)
1153
=================================================================
Total params: 324,353
Trainable params: 324,353
Non-trainable params: 0
_________________________________________________________________

```


## Further Reading

This section provides more resources on the topic if you are looking to go deeper.


## Papers

- A Guide To Convolution Arithmetic For Deep Learning, 2016.
https://arxiv.org/abs/1603.07285

- Deconvolutional Networks, 2010.
https://ieeexplore.ieee.org/document/5539957

- Is The Deconvolution Layer The Same As A Convolutional Layer?, 2016.
https://arxiv.org/abs/1609.07009

- Visualizing and Understanding Convolutional Networks, 2013.
https://arxiv.org/abs/1311.2901

- Fully Convolutional Networks for Semantic Segmentation, 2014.
https://arxiv.org/abs/1411.4038



## API

- Keras Convolutional Layers API.
https://keras.io/layers/convolutional/


## Articles

- Convolution Arithmetic Project, GitHub.
https://github.com/vdumoulin/conv_arithmetic

- What Are Deconvolutional Layers?, Data Science Stack Exchange.
https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers


## Summary

In this tutorial, you discovered how to use Upsampling and Transpose Convolutional Layers in
Generative Adversarial Networks when generating images. Specifically, you learned:

- Generative models in the GAN architecture are required to upsample input data in order
to generate an output image.

- The Upsampling layer is a simple layer with no weights that will double the dimensions of
input and can be used in a generative model when followed by a traditional convolutional
layer.

- The Transpose Convolutional layer is an inverse convolutional layer that will both upsample
input and learn how to fill in details during the model training process.


## Next

In the next tutorial, you will discover the algorithm for training generative adversarial network
models.