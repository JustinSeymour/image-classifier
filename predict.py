"""
Author: Justin Seymour
Date: 12 March 2020
"""

#Declare libraries for import
import argparse
import json

from predictions_helpers import process_image, predict
from get_input_args import predict_arg_parser
from checkpoint import reload_checkpoint

def main():
    
    args = predict_arg_parser()
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
        
    model = reload_checkpoint(args.checkpoint_path)
    
    image_tensor = process_image(args.image)
    
    top_probs, top_labels, top_flowers = predict(image_tensor, model, 5, cat_to_name, 'gpu')
    
    print(top_probs)
    
    print(top_labels)
    
    print(top_flowers)
    

if __name__ == '__main__': main()