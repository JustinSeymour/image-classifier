#!/bin/sh


echo "Configuring prediction settings...\n"

DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`

IMAGE='flowers/test/34/image_06961.jpg'
CHECKPOINT_PATH='./checkpoints/checkpoint.pth'
CATEGORY_NAMES='./cat_to_name.json'
FILENAME='prediction-log-'$DATE_WITH_TIME'.txt'

echo "Network is running prediction..."

python predict.py $IMAGE $CHECKPOINT_PATH --category_names=$CATEGORY_NAMES > prediction_logs/$FILENAME

echo "Check the results in the prediction_logs folder under the file called: $FILENAME"