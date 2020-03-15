"""
Author: Justin Seymour
Date: 12 March 2020
"""

#Declare libraries for import
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

def create_model(arch):

   model = None
 
   if arch == 'resnet':
      model = models.resnet18(pretrained=True)
      model.name = "resnet18"
   elif arch == 'alexnet':
      model = models.alexnet(pretrained=True)
      model.name = "alexnet"
   elif arch == 'vgg':
      model = models.vgg16(pretrained=True)
      model.name = "vgg16"

   if type(model) == type(None): 
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"

   # Freeze parameters to prevent back propagation
   for param in model.parameters():
    param.requires_grad = False

   return model

def check_gpu(gpu_arg):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu_arg else "cpu")

    # Print result
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    print("Device: {}".format(device))
    return device

def create_classifier(model, hidden_units, output_units, dropout):

   input_features = model.classifier[0].in_features
   print("Number of input features for model: {}".format(input_features))

   classifier = nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(input_features, hidden_units, bias=True)),
                           ('reLU1', nn.ReLU()),
                           ('dropout1', nn.Dropout(p=dropout)),
                           ('fc2', nn.Linear(hidden_units, output_units, bias=True)),
                           ('output', nn.LogSoftmax(dim=1))
                           ]))

   return classifier


def test_model(model, test_loader, device):
   # Do validation on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of network based on test images is: %d%%' % (100 * correct / total))

def train(model, training_loader, validation_loader, device, 
                  criterion, optimizer, epochs, print_every, steps):
  
    print("Model beginning training .....\n")

    running_loss = 0
    steps = 0

    for e in range(epochs):
        
        model.train() 
        
        for inputs, labels in training_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validate(model, validation_loader, criterion, device)
            
                print("Epoch number: {}/{} | ".format(e+1, epochs),
                     "Training Loss: {:.3f} | ".format(running_loss/print_every),
                     "Validation Loss: {:.3f} | ".format(valid_loss/len(validation_loader)),
                     "Validation Accuracy: {:.3f}".format(accuracy/len(validation_loader)))
            
                running_loss = 0
                model.train()

     
    return model

def validate(model, validation_loader, criterion, device):
    loss = 0
    accuracy = 0
    
    for inputs, labels in validation_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return loss, accuracy