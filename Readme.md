# Overview

This repository defines and trains a CycleGAN to read in an image from a set X and transform it so that it looks as if it belongs in set Y.

## Data set

This model has been trained on a set of images of Yosemite National Park taken either during the summer or winter. The seasons are the two domains.

These images do not contain labels. The CycleGAN learns the mapping between one image domain and another using an unsupervised approach.

## Approach

The following steps were taken to define and train the CycleGAN:
1. Load and visualize the input data
2. Define the CycleGAN architecture
3. Calculate the adversarial and cycle consistency losses and train the model
4. Evaluate the model by looking at the loss over time and looking at sample, generated images

## Model architecture

The CycleGAN is made of two types of networks: discriminators, and generators.

The discriminators are responsible for classifying images as real or fake (for both X and Y sets of images).

The generators are responsible for generating convincing, fake images for both kinds of images.
