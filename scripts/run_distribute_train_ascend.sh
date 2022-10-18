#!/bin/bash

if [ $# != 1 ]
then
    echo "Usage: bash run_distribute_train.sh [RANK_TABLE_FILE]."
    exit 1
fi

if [ ! -f $1 ]
then
    echo "RANK_TABLE_FILE Does Not Exist!"
    exit 1
fi


export RANK_TABLE_FILE=$1
export DEVICE_NUM=8
export RANK_SIZE=8

for((i=0; i<${RANK_SIZE}; i++))
do
    export DEVICE_ID=${i}
    export RANK_ID=${i}
    rm -rf ../train_parallel$i
    mkdir ../train_parallel$i
    cp ../*.py ../train_parallel$i
    cp -r ../src ../train_parallel$i
    cd ../train_parallel$i || exit
    echo "start distributed training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    python train.py > train$i.log 2>&1 &
done
