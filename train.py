"""
Author: Justin Seymour
Date: 12 March 2020
"""

#Declare libraries for import
import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from get_input_args import arg_parser
from data_preparation import create_transforms, image_datasets, data_loaders
from classifier import create_model, create_classifier, check_gpu, train, validate_model, validate
from checkpoint import initial_checkpoint

def main():

   # Instantiate the console arguments function
   args = arg_parser()

   # Define normalization for transforms
   normalize = transforms.Normalize(
      mean = [0.485, 0.456, 0.406],
      std = [0.229, 0.224, 0.225],
   )

   # Define transformations for training, validation and test sets
   data_transforms = create_transforms(30, 224 , 256, normalize)

   # Load the datasets from the image folders
   datasets = image_datasets(data_transforms)

   # Define the dataloaders using the image datasets
   loaders = data_loaders(datasets, 32)

   # Instantiate a new model
   model = create_model(arch=args.arch)

   # Create new classifier
   model.classifier = create_classifier(model, args.hidden_layers, args.output, args.dropout)

   device = check_gpu("gpu")
   model.to(device)

   learning_rate = args.learning_rate
   criterion = nn.NLLLoss()
   optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
   epochs = args.epochs
   print_every = args.print_every
   steps = 0
   trainloader = loaders['training']
   validloader = loaders['validation']

   # trained_model = train(model, epochs, learning_rate, criterion, optimizer, loaders['training'], loaders['validation'], device)
   trained_model = train(model, trainloader, validloader, device, criterion, optimizer, epochs, print_every, steps)

   print("Training has completed")
   
   validate_model(trained_model, loaders['testing'], device)

   initial_checkpoint(trained_model, args.checkpoint_dir, datasets['training'])


if __name__ == '__main__': main()


