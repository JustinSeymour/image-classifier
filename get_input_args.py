"""
Author: Justin Seymour
Date: 12 March 2020
"""

#Declare libraries for import
import argparse

def arg_parser():

   arg_parse = argparse.ArgumentParser("Configuring neural network settings")
   arg_parse.add_argument('data_dir', type=str, help='Folder of data images')
   arg_parse.add_argument('--arch', type=str, help='Configure architecture for model, you can choose resnet, vgg or alexnet', default='vgg')
   arg_parse.add_argument('--checkpoint_dir', type=str, help='Configure directory to save checkpoints to preserve models', default='checkpoints')
   arg_parse.add_argument('--learning_rate', type=float, help='Configure learning rate for the training model', default=0.001)
   arg_parse.add_argument('--hidden_layers', type=int, help='Hidden layer units for network', default=4096)
   arg_parse.add_argument('--epochs', type=int, help='Number of epochs for training as int', default=9)
   arg_parse.add_argument('--gpu', action='store_false', help='Set GPU or CPU')
   arg_parse.add_argument('--dropout', type=float, help='Set the dropout for the model', default=0.2)
   arg_parse.add_argument('--print_every', type=int, help='Set the number of times to print out results', default=40)
    
   args = arg_parse.parse_args()
   return args

def predict_arg_parser():

   arg_parse = argparse.ArgumentParser("Configuring neural network settings for predict.py")

   arg_parse.add_argument('image', type=str, help='Path of image file for prediction')
   arg_parse.add_argument('checkpoint_path', type=str, help='Location of the previously saved checkpoint')
   arg_parse.add_argument('--top_k', type=int, help='top K matches for prediction', default=5)
   arg_parse.add_argument('--category_names', type=str, help='Mappings for category names', default='./cat_to_name.json')
   arg_parse.add_argument('--gpu', action='store_true', help='Set GPU or CPU')


   args = arg_parse.parse_args()
   return args