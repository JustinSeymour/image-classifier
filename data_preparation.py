"""
Author: Justin Seymour
Date: 12 March 2020
"""

#Declare libraries for import
from torchvision import datasets, transforms, models
import torch

# Define defaults for directories storing training data
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


def create_transforms(rand_rotation, crop_size, resize, normalize):

   data_transforms = {
    
    'training': transforms.Compose([transforms.RandomRotation(rand_rotation),
                                          transforms.RandomResizedCrop(crop_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize]),
    
    'validation': transforms.Compose([transforms.Resize(resize),
                                          transforms.CenterCrop(crop_size),     
                                          transforms.ToTensor(),
                                          normalize]),
    
    'testing': transforms.Compose([transforms.Resize(resize),
                                          transforms.CenterCrop(crop_size),     
                                          transforms.ToTensor(),
                                          normalize])
   }

   return data_transforms


def image_datasets(dt):

   image_datasets = {
      'training': datasets.ImageFolder(train_dir, transform=dt['training']),
      'validation': datasets.ImageFolder(valid_dir, transform=dt['validation']),
      'testing': datasets.ImageFolder(test_dir, transform=dt['testing'])
   }

   return image_datasets


def data_loaders(image_datasets, batch_size): 

   data_loaders = {
    'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=batch_size),
    'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=batch_size),
    'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size=batch_size)
   }  

   return data_loaders