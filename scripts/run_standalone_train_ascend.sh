#!/bin/bash

if [ $# != 1 ]
then
  echo "Usage: bash run_standalone_train_ascend [DEVICE_ID]"
  exit 1
fi

DEVICE_ID=$1
python ../train.py --device_id=$DEVICE_ID > train.log 2>&1 &