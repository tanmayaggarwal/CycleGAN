# model definition
# import standard libraries

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import warnings
import torch.nn as nn
import torch.nn.functional as F

# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    #Creates a convolutional layer, with optional batch normalization.

    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)

class Discriminator(nn.Module):

    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim

        # define all convolutional layers - 5 layers that downsample the image by a factor of 2
        # 128x128 input
        self.conv1 = conv(3, conv_dim, 4, 2, 1, batch_norm=False) # input image is 128x128x3
        # 64x64 out
        self.conv2 = conv(conv_dim, conv_dim*2, 4, 2, 1, batch_norm=True)
        # 32x32 out
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4, 2, 1, batch_norm=True)
        # 16x16 out
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4, 2, 1, batch_norm=True)
        # 8x8 out
        # classification layer
        self.conv5 = conv(conv_dim*8, 1, 4, 1, batch_norm=False)

    def forward(self, x):
        # feedforward behavior
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # last, classification layer
        x = self.conv5(x)

        return x

# defining the ResidualBlock Class for the generator
class ResidualBlock(nn.Module):
    # defines a residual block
    # this adds an input x to a convolutional layer with the same input and output
    # These blocks allow a model to learn an effective transformation from one domain to another

    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        self.conv_dim = conv_dim
        self.conv1 = conv(conv_dim, conv_dim, 3, 1, 1, batch_norm=True)
        self.conv2 = conv(conv_dim, conv_dim, 3, 1, 1, batch_norm=True)

    def forward(self, x):
        # apply a Relu activation to the output of the first layer
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        # return a summed output
        x = x + out

        return x

# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    #Creates a transpose convolutional layer, with optional batch normalization.
    layers = []
    # append transpose conv layer
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    # optional batch norm layer
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)

class CycleGenerator(nn.Module):
    def __init__(self, conv_dim=64, n_res_blocks=6):
        super(CycleGenerator, self).__init__()

        self.conv_dim = conv_dim
        self.n_res_blocks = n_res_blocks

        # 1. define the encoder part of the generator
        # 128x128 input
        self.conv1 = conv(3, conv_dim, 4, 2, 1, batch_norm=True)
        # 64x64 out
        self.conv2 = conv(conv_dim, conv_dim*2, 4, 2, 1, batch_norm=True)
        # 32x32 out
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4, 2, 1, batch_norm=True)
        #16x16 out

        # 2. define the resnet part of the generator
        res_layers = []
        for layer in range(n_res_blocks):
            res_layers.append(ResidualBlock(conv_dim*4))
        self.res_blocks = nn.Sequential(*res_layers)

        # 3. define the decoder part of the generator
        self.deconv1 = deconv(conv_dim*4, conv_dim*2, 4)
        self.deconv2 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv3 = deconv(conv_dim, 3, 4, batch_norm=False)

    def forward(self, x):
        # given an image x, returns a transformed image
        # feedforward behavior
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        out = self.res_blocks(out)

        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = F.tanh(self.deconv3(out))

        return out

def create_model(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6):
    # builds the generators and discriminators

    # instantiate generators
    G_XtoY = CycleGenerator(conv_dim=g_conv_dim, n_res_blocks=n_res_blocks)
    G_YtoX = CycleGenerator(conv_dim=g_conv_dim, n_res_blocks=n_res_blocks)
    # instantiate discriminators
    D_X = Discriminator(conv_dim=d_conv_dim)
    D_Y = Discriminator(conv_dim=d_conv_dim)

    # move models to GPU, if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        G_XtoY.to(device)
        G_YtoX.to(device)
        D_X.to(device)
        D_Y.to(device)
        print('Models moved to GPU.')
    else:
        print('Only CPU available.')

    return G_XtoY, G_YtoX, D_X, D_Y

# helper function for printing the model architecture
def print_models(G_XtoY, G_YtoX, D_X, D_Y):
    """Prints model information for the generators and discriminators.
    """
    print("                     G_XtoY                    ")
    print("-----------------------------------------------")
    print(G_XtoY)
    print()

    print("                     G_YtoX                    ")
    print("-----------------------------------------------")
    print(G_YtoX)
    print()

    print("                      D_X                      ")
    print("-----------------------------------------------")
    print(D_X)
    print()

    print("                      D_Y                      ")
    print("-----------------------------------------------")
    print(D_Y)
    print()
