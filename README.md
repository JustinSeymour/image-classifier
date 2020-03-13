# Image Classifier to predict species of 102 flowers.

This is an image classifier that has been built with PyTorch. It uses a deep learning neural network to predict the image of flower based on the input of an image into the network.

### Training the model

I have set up a bash script that can be used to configure the inputs and run the train.py script. Use the following command to run the training script:

````bash
sh run_train.sh
````

You can configure the input variable for the network in the run_train.sh file:

````bash
ARCH='vgg'
HIDDEN_LAYERS=4096
EPOCHS=9
LEARNING_RATE=0.001
CHECKPOINT_DIR='checkpoints'
OUTPUT=102
DROPOUT=0.2
PRINT_EVERY=40
FILENAME='training-log-'$DATE_WITH_TIME'.txt'
````

### Making a prediction

I have set up a bash script to run the prediction script. Use the following command to run the prediction script:

````bash
sh run_predict.sh
````

You can configure the input variable for the network in the run_train.sh file:

````bash
IMAGE='flowers/test/34/image_06961.jpg'
CHECKPOINT_PATH='./checkpoints/checkpoint.pth'
CATEGORY_NAMES='./cat_to_name.json'
FILENAME='prediction-log-'$DATE_WITH_TIME'.txt'
````

### Other

This project was made while taking the Intro to AI course in Python Programming by Udacity.
