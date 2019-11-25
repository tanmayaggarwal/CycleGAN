# loss function definition
# import standard libraries

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import warnings
import torch.nn as nn
import torch.nn.functional as F

# To overcome the vanishing gradient problem, we use a least squares loss function for the discriminator
# Generator loss is similar to the discriminator loss but with flipped labels
# Cycle consistency loss will also be included in the generator loss as a measure of how good a reconstructed image is, when compared to the original image
# The total generator loss is the sum of the generator losses and the forward and backward cycle consistency losses

# Three loss functions are defined below:
# 1. real_mse_loss: looks at the output of a discriminator and returns the error based on how close that output is to being classified as real. This is a mean squared error.
# 2. fake_mse_loss: looks at the output of a discriminator and returns the error based on how close that output is to being classified as fake. This is a mean squared error.
# 3. cycle_consistency_loss: looks at a set of real image and a set of reconstructed/generated images, and returns the mean absolute error between them. This has a lambda_weight parameter that weights the mean absolute error in a batch.

def real_mse_loss(D_out):
    # how close is the produced output from being "real"?
    real_err = torch.mean((D_out-1)**2)
    return real_err

def fake_mse_loss(D_out):
    # how close is the produced output from being "fake"?
    fake_err = torch.mean(D_out**2)
    return fake_err

def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight):
    # calculate reconstruction loss
    # return weighted loss
    reconstruction_err = torch.mean(torch.abs(real_im - reconstructed_im))
    return lambda_weight * reconstruction_err
