
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision import datasets, transforms, models
from math import ceil

import torch

normalize = transforms.Normalize(
   mean = [0.485, 0.456, 0.406],
   std = [0.229, 0.224, 0.225],
)

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    print(image_path)
    img = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),     
                                          transforms.ToTensor(),
                                          normalize])
    
    img_tensor = transform(img)
    
    return img_tensor


def predict(image_tensor, model, topk, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    processed_image = image_tensor
    processed_image.unsqueeze_(0)
    
    model=model.cpu()
    
    probs = torch.exp(model.forward(processed_image))
    top_probs, top_labs = probs.topk(topk)
    
    idx_to_class = {}
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key
        
    np_top_labs = top_labs[0].numpy()
    
    top_labels = []
    
    for label in np_top_labs:
        top_labels.append(int(idx_to_class[label]))
        
    top_flowers = [cat_to_name[str(lab)] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers

