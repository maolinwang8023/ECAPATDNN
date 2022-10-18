#!/bin/bash

if [ $# != 2 ]
then
  echo "Usage: bash run_eval_asecnd.sh [DEVICE_ID] [PATH_CHECKPOINT]"
  exit 1
fi

DEVICE_ID=$1
PATH_CHECKPOINT=$2

if [ ! -f $2 ];
then echo "PATH_CHECKPOINT Does Not Exist!"
fi
python3 ../eval.py  --device_id=$DEVICE_ID --model_path=$PATH_CHECKPOINT > eval.log 2>&1 &