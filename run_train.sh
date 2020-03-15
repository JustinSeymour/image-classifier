#!/bin/sh

echo "Configuring network settings...\n"

DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`

ARCH='vgg'
HIDDEN_LAYERS=4096
EPOCHS=4
GPU=$TRUE
LEARNING_RATE=0.001
CHECKPOINT_DIR='checkpoints'
DROPOUT=0.2
PRINT_EVERY=40
FILENAME='training-log-'$DATE_WITH_TIME'.txt'

echo "=============================================="
echo "Network will train using the following config:"
echo "============================================== \n"

echo "Architecture: 	| $ARCH"
echo "Hidden layers: 	| $HIDDEN_LAYERS"
echo "Epochs:       	| $EPOCHS"
echo "Learning rate: 	| $LEARNING_RATE"
echo "Checkpoint dir: 	| $CHECKPOINT_DIR"
echo "Dropout rate: 	| $DROPOUT"
echo "Print every:  	| $PRINT_EVERY\n\n"


echo "Training has started...\n\n*** All output will be logged in the folder training_logs under $FILENAME. \n    When the model is complete, output will be sent to this console to notify you.***\n\n"

python3 train.py data_dir='flowers' --arch=${ARCH} --hidden_layers=${HIDDEN_LAYERS} --epochs=${EPOCHS} --learning_rate=${LEARNING_RATE} --checkpoint_dir=${CHECKPOINT_DIR} --dropout=${DROPOUT} --print_every=${PRINT_EVERY}
                
echo "Training model is now complete!"

