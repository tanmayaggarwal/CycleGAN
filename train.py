# training function definition
# import standard libraries

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loss_functions import real_mse_loss, fake_mse_loss, cycle_consistency_loss
from preprocess import scale

# import save code
from helpers import save_samples, checkpoint

# train the network
def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_x_optimizer, d_y_optimizer, n_epochs=1000):

    print_every=10

    # keep track of losses over time
    losses = []

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, to help inspect the model's performance.
    fixed_X = test_iter_X.next()[0]
    fixed_Y = test_iter_Y.next()[0]
    fixed_X = scale(fixed_X) # scale to a range -1 to 1
    fixed_Y = scale(fixed_Y)

    # batches per epoch
    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)
    batches_per_epoch = min(len(iter_X), len(iter_Y))

    for epoch in range(1, n_epochs+1):

        # Reset iterators for each epoch
        if epoch % batches_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)

        images_X, _ = iter_X.next()
        images_X = scale(images_X) # scale to a range -1 to 1

        images_Y, _ = iter_Y.next()
        images_Y = scale(images_Y)

        # move images to GPU if available (otherwise stay on CPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images_X = images_X.to(device)
        images_Y = images_Y.to(device)

        # ============================================
        #            TRAIN THE DISCRIMINATORS
        # ============================================

        ##   First: D_X, real and fake loss components   ##
        d_x_optimizer.zero_grad()

        # 1. Compute the discriminator losses on real images
        D_out = D_X(images_X)
        D_x_real_err = real_mse_loss(D_out)

        # 2. Generate fake images that look like domain X based on real images in domain Y
        G_out = G_YtoX(images_Y)

        # 3. Compute the fake loss for D_X
        D_out = D_X(G_out)
        D_x_fake_err = fake_mse_loss(D_out)

        # 4. Compute the total loss and perform backprop
        d_x_loss = D_x_real_err + D_x_fake_err
        d_x_loss.backward()
        d_x_optimizer.step()

        ##   Second: D_Y, real and fake loss components   ##
        d_y_optimizer.zero_grad()

        # 1. Compute the discriminator losses on real images
        D_out = D_Y(images_Y)
        D_y_real_err = real_mse_loss(D_out)

        # 2. Generate fake images that look like domain Y based on real images in domain X
        G_out = G_XtoY(images_X)

        # 3. Compute the fake loss for D_Y
        D_out = D_Y(G_out)
        D_y_fake_err = fake_mse_loss(D_out)

        # 4. Compute the total loss and perform backprop
        d_y_loss = D_y_real_err + D_y_fake_err
        d_y_loss.backward()
        d_y_optimizer.step()

        # =========================================
        #            TRAIN THE GENERATORS
        # =========================================

        ##    First: generate fake X images and reconstructed Y images    ##
        g_optimizer.zero_grad()
        # 1. Generate fake images that look like domain X based on real images in domain Y
        G_out = G_YtoX(images_Y)

        # 2. Compute the generator loss based on domain X
        D_out = D_X(G_out)
        g_YtoX_loss = real_mse_loss(D_out)

        # 3. Create a reconstructed y
        reconstructed_y = G_XtoY(G_out)

        # 4. Compute the cycle consistency loss (the reconstruction loss)
        reconstructed_y_loss = cycle_consistency_loss(images_Y, reconstructed_y, lambda_weight=10)

        ##    Second: generate fake Y images and reconstructed X images    ##

        # 1. Generate fake images that look like domain Y based on real images in domain X
        G_out = G_XtoY(images_X)

        # 2. Compute the generator loss based on domain Y
        D_out = D_Y(G_out)
        g_XtoY_loss = real_mse_loss(D_out)

        # 3. Create a reconstructed x
        reconstructed_x = G_YtoX(G_out)

        # 4. Compute the cycle consistency loss (the reconstruction loss)
        reconstructed_x_loss = cycle_consistency_loss(images_X, reconstructed_x, lambda_weight=10)

        # 5. Add up all generator and reconstructed losses and perform backprop
        g_total_loss = g_YtoX_loss + g_XtoY_loss + reconstructed_y_loss + reconstructed_x_loss
        g_total_loss.backward()
        g_optimizer.step()

        # Print the log info
        if epoch % print_every == 0:
            # append real and fake discriminator losses and the generator loss
            losses.append((d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))
            print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(epoch, n_epochs, d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))


        sample_every=100
        # Save the generated samples
        if epoch % sample_every == 0:
            G_YtoX.eval() # set generators to eval mode for sample generation
            G_XtoY.eval()
            save_samples(epoch, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=16)
            G_YtoX.train()
            G_XtoY.train()

        checkpoint_every=1000
        # Save the model parameters
        if epoch % checkpoint_every == 0:
            checkpoint(epoch, G_XtoY, G_YtoX, D_X, D_Y)

    return G_XtoY, G_YtoX, D_X, D_Y, d_x_optimizer, d_y_optimizer, g_optimizer, losses
