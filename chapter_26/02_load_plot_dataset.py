# %%
'''
## What Is the CycleGAN?
The CycleGAN model was described by Jun-Yan Zhu, et al. in their 2017 paper titled Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. The benefit of the CycleGAN model is that it can be trained without paired
examples. That is, it does not require examples of photographs before and after the translation
in order to train the model, e.g. photos of the same city landscape during the day and at night.
Instead, the model is able to use a collection of photographs from each domain and extract
and harness the underlying style of images in the collection in order to perform the translation.
The paper provides a good description of the models and training process, although the official
Torch implementation was used as the definitive description for each model and training process
and provides the basis for the model implementations described below.
'''

# %%
'''
We can load the dataset and plot some of the photos to confirm that we are handling
the image data correctly. The complete example is listed below.
'''

# %%
# load and plot the prepared dataset
from numpy import load
%matplotlib notebook
from matplotlib import pyplot
# load the face dataset
data = load('horse2zebra_256.npz')
dataA, dataB = data['arr_0'], data['arr_1']
print('Loaded: ', dataA.shape, dataB.shape)
# plot source images
n_samples = 3
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + i)
	pyplot.axis('off')
	pyplot.imshow(dataA[i].astype('uint8'))
# plot target image
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + n_samples + i)
	pyplot.axis('off')
	pyplot.imshow(dataB[i].astype('uint8'))
pyplot.show()