"""
Author: Justin Seymour
Date: 12 March 2020
"""

#Declare libraries for import
import argparse

def arg_parser():

   arg_parse = argparse.ArgumentParser("Configuring neural network settings")

   arg_parse.add_argument('--arch', type=str, help='Configure architecture for model')
   arg_parse.add_argument('--checkpoint_dir', type=str, help='Configure directory to save checkpoints to preserve models')
   arg_parse.add_argument('--learning_rate', type=float, help='Configure learning rate for the training model')
   arg_parse.add_argument('--hidden_layers', type=int, help='Hidden layer units for network')
   arg_parse.add_argument('--epochs', type=int, help='Number of epochs for training as int')
   arg_parse.add_argument('--gpu', action='store_true', help='Set GPU or CPU')
   arg_parse.add_argument('--output', type=int, help='Set the number of output units')
   arg_parse.add_argument('--dropout', type=float, help='Set the dropout for the model')
   arg_parse.add_argument('--print_every', type=int, help='Set the number of times to print out results')
    
   args = arg_parse.parse_args()
   return args

def predict_arg_parser():

   arg_parse = argparse.ArgumentParser("Configuring neural network settings for predict.py")

   arg_parse.add_argument('--image', type=str, help='Path of image file for prediction')
   arg_parse.add_argument('--checkpoint_path', type=str, help='Location of the previously saved checkpoint')
   arg_parse.add_argument('--top_k', type=int, help='top K matches for prediction')
   arg_parse.add_argument('--category_names', type=str, help='Mappings for category names')
   arg_parse.add_argument('--gpu', action='store_true', help='Set GPU or CPU')


   args = arg_parse.parse_args()
   return args