<img align="right" src="../logo-small.png">

# Lab : Introduction to neural learning: gradient descent

#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

**Note:** Elev8ed Notebooks (powered by Jupyter) will be accessible at the port given to you by your instructor. Password for jupyterLab : `1234`

All Notebooks are present in `work/generative-adversarial-networks` folder. To copy and paste: use **Control-C** and to paste inside of a terminal, use **Control-V**

You can access jupyter lab at `<host-ip>:<port>/lab/workspaces/`



### Chapter 12
How to Implement the Inception Score
Generative Adversarial Networks, or GANs for short, is a deep learning neural network architecture for training a generator model for generating synthetic images. A problem with generative
models is that there is no objective way to evaluate the quality of the generated images. As
such, it is common to periodically generate and save images during the model training process
and use subjective human evaluation of the generated images in order to both evaluate the
quality of the generated images and to select a final generator model. Many attempts have been
made to establish an objective measure of generated image quality. An early and somewhat
widely adopted example of an objective evaluation method for generated images is the Inception
Score, or IS. In this tutorial, you will discover the inception score for evaluating the quality of
generated images. After completing this tutorial, you will know:
- How to calculate the inception score and the intuition behind what it measures.
- How to implement the inception score in Python with NumPy and the Keras deep learning
library.
- How to calculate the inception score for small images such as those in the CIFAR-10
dataset.

Let’s get started.

12.1

Tutorial Overview

This tutorial is divided into five parts; they are:
1. What Is the Inception Score?
2. How to Calculate the Inception Score
3. How to Implement the Inception Score With NumPy
4. How to Implement the Inception Score With Keras
5. Problems With the Inception Score

253

### 12.2. What Is the Inception Score?

12.2

254

What Is the Inception Score?

The Inception Score, or IS for short, is an objective metric for evaluating the quality of
generated images, specifically synthetic images output by generative adversarial network models.
The inception score was proposed by Tim Salimans, et al. in their 2016 paper titled Improved
Techniques for Training GANs. In the paper, the authors use a crowd-sourcing platform (Amazon
Mechanical Turk) to evaluate a large number of GAN generated images. They developed the
inception score as an attempt to remove the subjective human evaluation of images. The authors
discover that their scores correlated well with the subjective evaluation.
As an alternative to human annotators, we propose an automatic method to evaluate
samples, which we find to correlate well with human evaluation ...
— Improved Techniques for Training GANs, 2016.
The inception score involves using a pre-trained deep learning neural network model for image
classification to classify the generated images. Specifically, the Inception v3 model described by
Christian Szegedy, et al. in their 2015 paper titled Rethinking the Inception Architecture for
Computer Vision. The reliance on the inception model gives the inception score its name. A
large number of generated images are classified using the model. Specifically, the probability of
the image belonging to each class is predicted. These predictions are then summarized into the
inception score. The score seeks to capture two properties of a collection of generated images:
- Image Quality. Do images look like a specific object?
- Image Diversity. Is a wide range of objects generated?

The inception score has a lowest value of 1.0 and a highest value of the number of classes
supported by the classification model; in this case, the Inception v3 model supports the 1,000
classes of the ILSVRC 2012 dataset, and as such, the highest inception score on this dataset is
1,000. The CIFAR-10 dataset is a collection of 50,000 images divided into 10 classes of objects.
The original paper that introduces the inception calculated the score on the real CIFAR-10
training dataset, achieving a result of 11.24 +/- 0.12. Using the GAN model also introduced in
their paper, they achieved an inception score of 8.09 +/- .07 when generating synthetic images
for this dataset.

12.3

How to Calculate the Inception Score

The inception score is calculated by first using a pre-trained Inception v3 model to predict the
class probabilities for each generated image. These are conditional probabilities, e.g. class label
conditional on the generated image. Images that are classified strongly as one class over all
other classes indicate a high quality. As such, the conditional probability of all generated images
in the collection should have a low entropy.
Images that contain meaningful objects should have a conditional label distribution
p(y|x) with low entropy.
— Improved Techniques for Training GANs, 2016.

### 12.3. How to Calculate the Inception Score

255

The entropy is calculated as the negative sum of each observed probability multiplied by the
log of the probability. The intuition here is that large probabilities have less information than
small probabilities.
X
entropy = −
pi × log(pi )
(12.1)
The conditional probability captures our interest in image quality. To capture our interest
in a variety of images, we use the marginal probability. This is the probability distribution
of all generated images. We, therefore, would prefer the integral of the marginal probability
distribution to have a high entropy.
Moreover, we expect the model to generate varied images, so the marginal integral
p(y|x = G(z))dz should have high entropy.
— Improved Techniques for Training GANs, 2016.
These elements are combined by calculating the Kullback-Leibler divergence, or KL divergence
(relative entropy), between the conditional and marginal probability distributions. Calculating
the divergence between two distributions is written using the || operator, therefore we can say we
are interested in the KL divergence between C for conditional and M for marginal distributions
or:
KL(C||M )

(12.2)

Specifically, we are interested in the average of the KL divergence for all generated images.
Combining these two requirements, the metric that we propose is: exp(Ex KL(p(y|x)||p(y))).
— Improved Techniques for Training GANs, 2016.
We don’t need to translate the calculation of the inception score. Thankfully, the authors of
the paper also provide source code on GitHub that includes an implementation of the inception
score. The calculation of the score assumes a large number of images for a range of objects, such
as 50,000. The images are split into 10 groups, e.g 5,000 images per group, and the inception
score is calculated on each group of images, then the average and standard deviation of the
score is reported.
The calculation of the inception score on a group of images involves first using the inception v3
model to calculate the conditional probability for each image (p(y|x)). The marginal probability
is then calculated as the average of the conditional probabilities for the images in the group
(p(y)). The KL divergence is then calculated for each image as the conditional probability
multiplied by the log of the conditional probability minus the log of the marginal probability.
KL divergence = p(y|x) × (log(p(y|x)) − log(p(y)))

(12.3)

The KL divergence is then summed over all images and averaged over all classes and the
exponent of the result is calculated to give the final score. This defines the official inception
score implementation used when reported in most papers that use the score, although variations
on how to calculate the score do exist.

### 12.4. How to Implement the Inception Score With NumPy

12.4

256

How to Implement the Inception Score With NumPy

Implementing the calculation of the inception score in Python with NumPy arrays is straightforward. First, let’s define a function that will take a collection of conditional probabilities
and calculate the inception score. The calculate inception score() function listed below
implements the procedure. One small change is the introduction of an epsilon (a tiny number
close to zero) when calculating the log probabilities to avoid blowing up when trying to calculate
the log of a zero probability. This is probably not needed in practice (e.g. with real generated
images) but is useful here and good practice when working with log probabilities.
# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=1E-16):
# calculate p(y)
p_y = expand_dims(p_yx.mean(axis=0), 0)
# kl divergence for each image
kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
# sum over classes
sum_kl_d = kl_d.sum(axis=1)
# average over images
avg_kl_d = mean(sum_kl_d)
# undo the logs
is_score = exp(avg_kl_d)
return is_score

```

We can then test out this function to calculate the inception score for some contrived
conditional probabilities. We can imagine the case of three classes of image and a perfect
confident prediction for each class for three images.
# conditional probabilities for high quality images
p_yx = asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

```

We would expect the inception score for this case to be 3.0 (or very close to it). This is
because we have the same number of images for each image class (one image for each of the
three classes) and each conditional probability is maximally confident. The complete example
for calculating the inception score for these probabilities is listed below.
# calculate inception score in numpy
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp
# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=1E-16):
# calculate p(y)
p_y = expand_dims(p_yx.mean(axis=0), 0)
# kl divergence for each image
kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
# sum over classes
sum_kl_d = kl_d.sum(axis=1)
# average over images

### 12.4. How to Implement the Inception Score With NumPy

257

avg_kl_d = mean(sum_kl_d)
# undo the logs
is_score = exp(avg_kl_d)
return is_score
# conditional probabilities for high quality images
p_yx = asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
score = calculate_inception_score(p_yx)
print(score)

```

Running the example gives the expected score of 3.0 (or a number extremely close).
2.999999999999999

```

We can also try the worst case. This is where we still have the same number of images for
each class (one for each of the three classes), but the objects are unknown, giving a uniform
predicted probability distribution across each class.
# conditional probabilities for low quality images
p_yx = asarray([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
score = calculate_inception_score(p_yx)
print(score)

```

In this case, we would expect the inception score to be the worst possible where there is no
difference between the conditional and marginal distributions, e.g. an inception score of 1.0.
Tying this together, the complete example is listed below.
# calculate inception score in numpy
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp
# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=1E-16):
# calculate p(y)
p_y = expand_dims(p_yx.mean(axis=0), 0)
# kl divergence for each image
kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
# sum over classes
sum_kl_d = kl_d.sum(axis=1)
# average over images
avg_kl_d = mean(sum_kl_d)
# undo the logs
is_score = exp(avg_kl_d)
return is_score
# conditional probabilities for low quality images
p_yx = asarray([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
score = calculate_inception_score(p_yx)

### 12.5. How to Implement the Inception Score With Keras

258

print(score)

```

Running the example reports the expected inception score of 1.0.
1.0

```

You may want to experiment with the calculation of the inception score and test other
pathological cases.

12.5

How to Implement the Inception Score With Keras

Now that we know how to calculate the inception score and to implement it in Python, we can
develop an implementation in Keras. This involves using the real Inception v3 model to classify
images and to average the calculation of the score across multiple splits of a collection of images.
First, we can load the Inception v3 model in Keras directly.
...
# load inception v3 model
model = InceptionV3()

```

The model expects images to be color and to have the shape 299 × 299 pixels. Additionally,
the pixel values must be scaled in the same way as the training data images, before they can be
classified. This can be achieved by converting the pixel values from integers to floating point
values and then calling the preprocess input() function for the images.
...
# convert from uint8 to float32
processed = images.astype('float32')
# pre-process raw images for inception v3 model
processed = preprocess_input(processed)

```

Then the conditional probabilities for each of the 1,000 image classes can be predicted for
the images.
...
# predict class probabilities for images
yhat = model.predict(images)

```

The inception score can then be calculated directly on the NumPy array of probabilities as
we did in the previous section. Before we do that, we must split the conditional probabilities
into groups, controlled by a n split argument and set to the default of 10 as was used in the
original paper.
...
n_part = floor(images.shape[0] / n_split)

```


### 12.5. How to Implement the Inception Score With Keras

259

We can then enumerate over the conditional probabilities in blocks of n part images or
predictions and calculate the inception score.
...
# retrieve p(y|x)
ix_start, ix_end = i * n_part, (i+1) * n_part
p_yx = yhat[ix_start:ix_end]

```

After calculating the scores for each split of conditional probabilities, we can calculate and
return the average and standard deviation inception scores.
...
# average across images
is_avg, is_std = mean(scores), std(scores)

```

across splits.
Tying all of this together, the calculate inception score() function below takes an array
of images with the expected size and pixel values in [0,255] and calculates the average and
standard deviation inception scores using the inception v3 model in Keras.
# assumes images have the shape 299x299x3, pixels in [0,255]
def calculate_inception_score(images, n_split=10, eps=1E-16):
# load inception v3 model
model = InceptionV3()
# convert from uint8 to float32
processed = images.astype('float32')
# pre-process raw images for inception v3 model
processed = preprocess_input(processed)
# predict class probabilities for images
yhat = model.predict(processed)
# enumerate splits of images/predictions
scores = list()
n_part = floor(images.shape[0] / n_split)
for i in range(n_split):
# retrieve p(y|x)
ix_start, ix_end = i * n_part, i * n_part + n_part
p_yx = yhat[ix_start:ix_end]
# calculate p(y)
p_y = expand_dims(p_yx.mean(axis=0), 0)
# calculate KL divergence using log probabilities
kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
# sum over classes
sum_kl_d = kl_d.sum(axis=1)
# average over images
avg_kl_d = mean(sum_kl_d)
# undo the log
is_score = exp(avg_kl_d)
# store
scores.append(is_score)
# average across images
is_avg, is_std = mean(scores), std(scores)
return is_avg, is_std

### 12.5. How to Implement the Inception Score With Keras

260

```

calculating the inception score.
We can test this function with 50 artificial images with the value 1.0 for all pixels.
...
# pretend to load images
images = ones((50, 299, 299, 3))
print('loaded', images.shape)

```

This will calculate the score for each group of five images and the low quality would suggest
that an average inception score of 1.0 will be reported. The complete example is listed below.
# calculate inception score with Keras
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
# assumes images have the shape 299x299x3, pixels in [0,255]
def calculate_inception_score(images, n_split=10, eps=1E-16):
# load inception v3 model
model = InceptionV3()
# convert from uint8 to float32
processed = images.astype('float32')
# pre-process raw images for inception v3 model
processed = preprocess_input(processed)
# predict class probabilities for images
yhat = model.predict(processed)
# enumerate splits of images/predictions
scores = list()
n_part = floor(images.shape[0] / n_split)
for i in range(n_split):
# retrieve p(y|x)
ix_start, ix_end = i * n_part, i * n_part + n_part
p_yx = yhat[ix_start:ix_end]
# calculate p(y)
p_y = expand_dims(p_yx.mean(axis=0), 0)
# calculate KL divergence using log probabilities
kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
# sum over classes
sum_kl_d = kl_d.sum(axis=1)
# average over images
avg_kl_d = mean(sum_kl_d)
# undo the log
is_score = exp(avg_kl_d)
# store
scores.append(is_score)

### 12.5. How to Implement the Inception Score With Keras

261

# average across images
is_avg, is_std = mean(scores), std(scores)
return is_avg, is_std
# pretend to load images
images = ones((50, 299, 299, 3))
print('loaded', images.shape)
# calculate inception score
is_avg, is_std = calculate_inception_score(images)
print('score', is_avg, is_std)

```

calculating the inception score.
Running the example first defines the 50 fake images, then calculates the inception score on
each batch and reports the expected inception score of 1.0, with a standard deviation of 0.0.
Note: the first time the InceptionV3 model is used, Keras will download the model weights
and save them into the ∼/.keras/models/ directory on your workstation. The weights are
about 100 megabytes and may take a moment to download depending on the speed of your
internet connection.
loaded (50, 299, 299, 3)
score 1.0 0.0

```

model and calculating the inception score.
We can test the calculation of the inception score on some real images. The Keras API
provides access to the CIFAR-10 dataset (described in Section 8.2). These are color photos
with the small size of 32 × 32 pixels. First, we can split the images into groups, then upsample
the images to the expected size of 299 × 299, pre-process the pixel values, predict the class
probabilities, then calculate the inception score. This will be a useful example if you intend to
calculate the inception score on your own generated images, as you may have to either scale
the images to the expected size for the inception v3 model or change the model to perform the
upsampling for you. First, the images can be loaded and shuffled to ensure each split covers a
diverse set of classes.
...
# load cifar10 images
(images, _), (_, _) = cifar10.load_data()
# shuffle images
shuffle(images)

```

Next, we need a way to scale the images. We will use the scikit-image library to resize
the NumPy array of pixel values to the required size. The scale images() function below
implements this.
# scale an array of images to a new size
def scale_images(images, new_shape):
images_list = list()
for image in images:

### 12.5. How to Implement the Inception Score With Keras

262

# resize with nearest neighbor interpolation
new_image = resize(image, new_shape, 0)
# store
images_list.append(new_image)
return asarray(images_list)

```

You may have to install the scikit-image library if it is not already installed. This can be
achieved as follows:
sudo pip install scikit-image

```

We can then enumerate the number of splits, select a subset of the images, scale them,
pre-process them, and use the model to predict the conditional class probabilities.
...
# retrieve images
ix_start, ix_end = i * n_part, (i+1) * n_part
subset = images[ix_start:ix_end]
# convert from uint8 to float32
subset = subset.astype('float32')
# scale images to the required size
subset = scale_images(subset, (299,299,3))
# pre-process images, scale to [-1,1]
subset = preprocess_input(subset)
# predict p(y|x)
p_yx = model.predict(subset)

```

probabilities.
The rest of the calculation of the inception score is the same. Tying this all together, the
complete example for calculating the inception score on the real CIFAR-10 training dataset is
listed below. Based on the similar calculation reported in the original inception score paper, we
would expect the reported score on this dataset to be approximately 11. Interestingly, the best
inception score for CIFAR-10 with generated images is about 8.8 at the time of writing using a
progressive growing GAN.
# calculate inception score for cifar-10 in Keras
from math import floor
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets import cifar10
from skimage.transform import resize
from numpy import asarray
# scale an array of images to a new size
def scale_images(images, new_shape):

### 12.5. How to Implement the Inception Score With Keras

263

images_list = list()
for image in images:
# resize with nearest neighbor interpolation
new_image = resize(image, new_shape, 0)
# store
images_list.append(new_image)
return asarray(images_list)
# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images, n_split=10, eps=1E-16):
# load inception v3 model
model = InceptionV3()
# enumerate splits of images/predictions
scores = list()
n_part = floor(images.shape[0] / n_split)
for i in range(n_split):
# retrieve images
ix_start, ix_end = i * n_part, (i+1) * n_part
subset = images[ix_start:ix_end]
# convert from uint8 to float32
subset = subset.astype('float32')
# scale images to the required size
subset = scale_images(subset, (299,299,3))
# pre-process images, scale to [-1,1]
subset = preprocess_input(subset)
# predict p(y|x)
p_yx = model.predict(subset)
# calculate p(y)
p_y = expand_dims(p_yx.mean(axis=0), 0)
# calculate KL divergence using log probabilities
kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
# sum over classes
sum_kl_d = kl_d.sum(axis=1)
# average over images
avg_kl_d = mean(sum_kl_d)
# undo the log
is_score = exp(avg_kl_d)
# store
scores.append(is_score)
# average across images
is_avg, is_std = mean(scores), std(scores)
return is_avg, is_std
# load cifar10 images
(images, _), (_, _) = cifar10.load_data()
# shuffle images
shuffle(images)
print('loaded', images.shape)
# calculate inception score
is_avg, is_std = calculate_inception_score(images)
print('score', is_avg, is_std)

```

Running the example may take some time depending on the speed of your workstation. It
may also require a large amount of RAM. If you have trouble with the example, try reducing

### 12.6. Problems With the Inception Score

264

the number of images in the training dataset. Running the example loads the dataset, prepares
the model, and calculates the inception score on the CIFAR-10 training dataset. We can see
that the score is 11.3, which is close to the expected score of 11.24.
loaded (50000, 32, 32, 3)
score 11.317895 0.14821531

```

dataset.

12.6

Problems With the Inception Score

The inception score is effective, but it is not perfect. Generally, the inception score is appropriate
for generated images of objects known to the model used to calculate the conditional class
probabilities. In this case, because the inception v3 model is used, this means that it is most
suitable for 1,000 object types used in the ILSVRC 2012 dataset1 . This is a lot of classes, but
not all objects that may interest us.
It also requires that the images are square and have the relatively small size of about
300 × 300 pixels, including any scaling required to get your generated images to that size. A
good score also requires having a good distribution of generated images across the possible
objects supported by the model, and close to an even number of examples for each class. This
can be hard to control for many GAN models that don’t offer controls over the types of objects
generated. Shane Barratt and Rishi Sharma take a closer look at the inception score and list a
number of technical issues and edge cases in there 2018 paper titled A Note on the Inception
Score. This is a good reference if you wish to dive deeper.

12.7

Further Reading

This section provides more resources on the topic if you are looking to go deeper.

12.7.1

Papers

- Improved Techniques for Training GANs, 2016.
https://arxiv.org/abs/1606.03498
- A Note on the Inception Score, 2018.
https://arxiv.org/abs/1801.01973
- Rethinking the Inception Architecture for Computer Vision, 2015.
https://arxiv.org/abs/1512.00567

12.7.2

Projects

- Code for the paper Improved Techniques for Training GANs.
https://github.com/openai/improved-gan
1

http://image-net.org/challenges/LSVRC/2012/browse-synsets

### 12.8. Summary

265

- Large Scale Visual Recognition Challenge 2012 (ILSVRC2012).
http://image-net.org/challenges/LSVRC/2012/

12.7.3

API

- Keras Inception v3 Model.
https://keras.io/applications/#inceptionv3
- scikit-image Library.
https://scikit-image.org/

12.7.4

Articles

- Image Generation on CIFAR-10
https://paperswithcode.com/sota/image-generation-on-cifar-10
- Inception Score calculation, GitHub, 2017.
https://github.com/openai/improved-gan/issues/29
- Kullback-Leibler divergence, Wikipedia.
https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
- Entropy (information theory), Wikipedia.
https://en.wikipedia.org/wiki/Entropy_(information_theory)

12.8

Summary

In this tutorial, you discovered the inception score for evaluating the quality of generated images.
Specifically, you learned:
- How to calculate the inception score and the intuition behind what it measures.
- How to implement the inception score in Python with NumPy and the Keras deep learning
library.
- How to calculate the inception score for small images such as those in the CIFAR-10
dataset.

12.8.1

Next

In the next tutorial, you will discover the FID score and how to implement it from scratch and
interpret the results.
