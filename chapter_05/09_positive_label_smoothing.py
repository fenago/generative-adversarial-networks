# %%
'''
## Use Label Smoothing
It is common to use the class label 1 to represent real images and class label 0 to represent fake
images when training the discriminator model. These are called hard labels, as the label values
are precise or crisp. It is a good practice to use soft labels, such as values slightly more or less
than 1.0 or slightly more than 0.0 for real and fake images respectively, where the variation for
each image is random. This is often referred to as label smoothing and can have a regularizing
effect when training the model. The example below demonstrates defining 1,000 labels for the
positive class (class = 1) and smoothing the label values uniformly into the range [0.7,1.2] as
recommended.
'''

# %%
# example of positive label smoothing
from numpy import ones
from numpy.random import random

# example of smoothing class=1 to [0.7, 1.2]
def smooth_positive_labels(y):
	return y - 0.3 + (random(y.shape) * 0.5)

# generate 'real' class labels (1)
n_samples = 1000
y = ones((n_samples, 1))
# smooth labels
y = smooth_positive_labels(y)
# summarize smooth labels
print(y.shape, y.min(), y.max())

# %%
'''
Running the example summarizes the min and max values for the smooth values, showing
they are close to the expected values.

(1000, 1) 0.7003103006957805 1.1997858934066357
'''