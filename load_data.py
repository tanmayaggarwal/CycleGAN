# loading in and transforming data
# import standard libraries

import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def get_data_loader(image_type, image_dir='summer2winter_yosemite', image_size=128, batch_size=16, num_workers=0):
    # Returns training and test data loaders for a given image type, either 'summer' or 'winter'.
    # These images will be resized to 128x128x3, by default, converted into Tensors, and normalized.

    # resize and normalize the images
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()]) # resize to 128x128

    # get training and test directories
    image_path = './' + image_dir
    train_path = os.path.join(image_path, image_type)
    test_path = os.path.join(image_path, 'test_{}'.format(image_type))

    # define datasets using ImageFolder
    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    # create and return DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
