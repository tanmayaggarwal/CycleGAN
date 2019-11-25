# main application file
# import standard libraries

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import warnings
import torch.optim as optim

# loading the data
from load_data import get_data_loader

# Create train and test dataloaders for images from the two domains X and Y
# image_type = directory names for our data
dataloader_X, test_dataloader_X = get_data_loader(image_type='summer')
dataloader_Y, test_dataloader_Y = get_data_loader(image_type='winter')

# display sample of training images
from visualize_data import visualize_data
images = visualize_data(dataloader_X, dataloader_Y)

# pre-processing the images
from preprocess import scale

# current range
img = images[0]

print('Min: ', img.min())
print('Max: ', img.max())

# scaled range
scaled_img = scale(img)

print('Scaled min: ', scaled_img.min())
print('Scaled max: ', scaled_img.max())

# define the CycleGAN model
from model import create_model, print_models
G_XtoY, G_YtoX, D_X, D_Y = create_model(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6)

# print all of the models
print_models(G_XtoY, G_YtoX, D_X, D_Y)

# computing the discriminator and generator losses
from loss_functions import real_mse_loss, fake_mse_loss, cycle_consistency_loss

# set hyperparameters for the Adam optimizer
lr = 0.0002
beta1 = 0.5
beta2 = 0.999

g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters

# Create optimizers for the generators and discriminators
g_optimizer = optim.Adam(g_params, lr, [beta1, beta2])
d_x_optimizer = optim.Adam(D_X.parameters(), lr, [beta1, beta2])
d_y_optimizer = optim.Adam(D_Y.parameters(), lr, [beta1, beta2])

# training the network
from train import training_loop
n_epochs = 1000
G_XtoY, G_YtoX, D_X, D_Y, d_x_optimizer, d_y_optimizer, g_optimizer, losses = training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_x_optimizer, d_y_optimizer, n_epochs=n_epochs)

# visualize the losses

fig, ax = plt.subplots(figsize=(12,8))
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator, X', alpha=0.5)
plt.plot(losses.T[1], label='Discriminator, Y', alpha=0.5)
plt.plot(losses.T[2], label='Generators', alpha=0.5)
plt.title("Training Losses")
plt.legend()

# display generated samples
from view_samples import view_samples
# view samples at iteration 100
view_samples(100, 'samples_cyclegan')

# view samples at iteration 1000
view_samples(1000, 'samples_cyclegan')

# saving the models
torch.save({'Discriminator_X_state_dict': D_X.state_dict(), 'Discriminator_Y_state_dict': D_Y.state_dict(), 'Generator_XtoY_state_dict': G_XtoY.state_dict(), 'Generator_YtoX_state_dict': G_YtoX.state_dict(), 'd_y_optimizer_state_dict': d_y_optimizer.state_dict(), 'd_x_optimizer_state_dict': d_x_optimizer.state_dict(), 'g_optimizer_state_dict': g_optimizer.state_dict()}, "saved_model.pth")
