# visualizing data
# import standard libraries
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import warnings

# %matplotlib inline
# helper imshow function
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    return

def visualize_data(dataloader_X, dataloader_Y):
    # get some images from X
    dataiter = iter(dataloader_X)
    # the "_" is a placeholder for no labels
    images, _ = dataiter.next()

    # show images
    fig = plt.figure(figsize=(12, 8))
    imshow(torchvision.utils.make_grid(images))

    # get some images from Y
    dataiter = iter(dataloader_Y)
    images, _ = dataiter.next()

    # show images
    fig = plt.figure(figsize=(12,8))
    imshow(torchvision.utils.make_grid(images))

    return images
