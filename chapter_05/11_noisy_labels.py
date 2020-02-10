# %%
'''
## Use Noisy Labels
The labels used when training the discriminator model are always correct. This means that fake
images are always labeled with class 0 and real images are always labeled with class 1. It is
recommended to introduce some errors to these labels where some fake images are marked as
real, and some real images are marked as fake. If you are using separate batches to update the
discriminator for real and fake images, this may mean randomly adding some fake images to the
batch of real images, or randomly adding some real images to the batch of fake images. If you
are updating the discriminator with a combined batch of real and fake images, then this may
involve randomly flipping the labels on some images. The example below demonstrates this by
creating 1,000 samples of real (class = 1) labels and flipping them with a 5% probability, then
doing the same with 1,000 samples of fake (class = 0) labels.
'''

# %%
# example of noisy labels
from numpy import ones
from numpy import zeros
from numpy.random import choice

# randomly flip some labels
def noisy_labels(y, p_flip):
	# determine the number of labels to flip
	n_select = int(p_flip * y.shape[0])
	# choose labels to flip
	flip_ix = choice([i for i in range(y.shape[0])], size=n_select)
	# invert the labels in place
	y[flip_ix] = 1 - y[flip_ix]
	return y

# generate 'real' class labels (1)
n_samples = 1000
y = ones((n_samples, 1))
# flip labels with 5% probability
y = noisy_labels(y, 0.05)
# summarize labels
print(y.sum())

# generate 'fake' class labels (0)
y = zeros((n_samples, 1))
# flip labels with 5% probability
y = noisy_labels(y, 0.05)
# summarize labels
print(y.sum())

# %%
'''
Try running the example a few times. The results show that approximately 50 of the 1s are
flipped to 0s for the positive labels (e.g. 5% of 1,000) and approximately 50 0s are flopped to 1s
in for the negative labels.

951.0

49.0
'''
