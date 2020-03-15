"""
Author: Justin Seymour
Date: 12 March 2020
"""

from os.path import isdir
from torchvision import datasets, transforms, models
import torch

def initial_checkpoint(model, directory, training_data):
       
    # Save model at checkpoint
    if type(directory) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if isdir(directory):
            # Create `class_to_idx` attribute in model
            model.class_to_idx = training_data.class_to_idx
            
            # Create checkpoint dictionary
            checkpoint = {'architecture': model.name,
                          'classifier': model.classifier,
                          'class_to_idx': model.class_to_idx,
                          'state_dict': model.state_dict()}
            
            # Save checkpoint
            torch.save(checkpoint, './checkpoints/checkpoint.pth')

        else: 
            print("Directory not found, model will not be saved.")

def reload_checkpoint(path):

    checkpoint = torch.load(path)
    
    # Load Defaults if none specified
    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else: 
        exec("model = models.{}(pretrained=True)".checkpoint['architecture'])
        model.name = checkpoint['architecture']
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): param.requires_grad = False
    
    # Load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model