# %%
'''
## How to Implement the PatchGAN Discriminator Model
The discriminator model in the Pix2Pix GAN is implemented as a PatchGAN. The PatchGAN
is designed based on the size of the receptive field, sometimes called the effective receptive field.
The receptive field is the relationship between one output activation of the model to an area on
the input image (actually volume as it proceeded down the input channels). A PatchGAN with
the size 70 × 70 is used, which means that the output (or each output) of the model maps to a
70 × 70 square of the input image. In effect, a 70 × 70 PatchGAN will classify 70 × 70 patches
of the input image as real or fake.
'''

# %%
'''
Before we dive into the configuration details of the PatchGAN, it is important to get a handle
on the calculation of the receptive field. The receptive field is not the size of the output of the
discriminator model, e.g. it does not refer to the shape of the activation map output by the22.3. How to Implement the PatchGAN Discriminator Model 466
model. It is a definition of the model in terms of one pixel in the output activation map to the
input image. The output of the model may be a single value or a square activation map of values
that predict whether each patch of the input image is real or fake. Traditionally, the receptive
field refers to the size of the activation map of a single convolutional layer with regards to the
input of the layer, the size of the filter, and the size of the stride. The effective receptive field
generalizes this idea and calculates the receptive field for the output of a stack of convolutional
layers with regard to the raw image input. The terms are often used interchangeably.
The authors of the Pix2Pix GAN provide a Matlab script to calculate the effective receptive
field size for different model configurations in a script called receptive field sizes.m1. It
can be helpful to work through an example for the 70 × 70 PatchGAN receptive field calculation.
The 70 × 70 PatchGAN has a fixed number of three layers (excluding the output and second
last layers), regardless of the size of the input image. The calculation of the receptive field in
one dimension is calculated as:

receptive field = (output size − 1) × stride + kernel size

Where output size is the size of the prior layers activation map, stride is the number of
pixels the filter is moved when applied to the activation, and kernel size is the size of the filter
to be applied. The PatchGAN uses a fixed stride of 2 × 2 (except in the output and second
last layers) and a fixed kernel size of 4 × 4. We can, therefore, calculate the receptive field size
starting with one pixel in the output of the model and working backward to the input image.
We can develop a Python function called receptive field() to calculate the receptive field,
then calculate and print the receptive field for each layer in the Pix2Pix PatchGAN model. The
complete example is listed below.
'''

# %%
# example of calculating the receptive field for the PatchGAN

# calculate the effective receptive field size
def receptive_field(output_size, kernel_size, stride_size):
    return (output_size - 1) * stride_size + kernel_size

# output layer 1x1 pixel with 4x4 kernel and 1x1 stride
rf = receptive_field(1, 4, 1)
print(rf)
# second last layer with 4x4 kernel and 1x1 stride
rf = receptive_field(rf, 4, 1)
print(rf)
# 3 PatchGAN layers with 4x4 kernel and 2x2 stride
rf = receptive_field(rf, 4, 2)
print(rf)
rf = receptive_field(rf, 4, 2)
print(rf)
rf = receptive_field(rf, 4, 2)
print(rf)