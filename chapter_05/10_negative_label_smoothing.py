# %%
'''
## Use Label Smoothing (Negative)
There have been some suggestions that only positive-class label smoothing is required and
to values less than 1.0. Nevertheless, you can also smooth negative class labels. The example
below demonstrates generating 1,000 labels for the negative class (class = 0) and smoothing
the label values uniformly into the range [0.0, 0.3] as recommended.
'''

# %%
# example of negative label smoothing
from numpy import zeros
from numpy.random import random

# example of smoothing class=0 to [0.0, 0.3]
def smooth_negative_labels(y):
	return y + random(y.shape) * 0.3

# generate 'fake' class labels (0)
n_samples = 1000
y = zeros((n_samples, 1))
# smooth labels
y = smooth_negative_labels(y)
# summarize smooth labels
print(y.shape, y.min(), y.max())