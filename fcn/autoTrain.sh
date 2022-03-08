#!/bin/bash
# Input data
RAW_PATH="data/raw/"
FMT_PATH="data/clean/"

# parameters
EPOCHS=2000
TEST=150

GPU_NAME="GTX1080"
BROWSER="chrome"

DATA_FOLDER=$GPU_NAME"-"$BROWSER"-"$EPOCHS"epo-"$TEST"tst/"
DATA_ID="result"

mkdir -p "$FMT_PATH$DATA_FOLDER"
mkdir -p "Results/"$DATA_FOLDER
mkdir -p "Models/"
mkdir -p "log/"

# Config
echo "GPU:"$GPU_NAME" Browser:"$BROWSER" ."
echo "Test:"$TEST" Epochs:"$EPOCHS" ."

# Convert and train a model
# ORIGINAL_DATA_PATH="$RAW_PATH$DATA_FOLDER"
FORMATTED_DATA_PATH="$FMT_PATH$DATA_FOLDER$DATA_ID"
echo "Starting Convert..."
./convert.py $RAW_PATH $FORMATTED_DATA_PATH $TEST
echo "Starting FCN..."
./FCN.py $FORMATTED_DATA_PATH $EPOCHS $DATA_FOLDER$DATA_ID >> "log/fcn.log"
echo "FCN Finished!"

# Evaluate the module
ModelName="Models/"$DATA_FOLDER$DATA_ID
TestData=$FORMATTED_DATA_PATH"_TEST"
ResName="Results/"$DATA_FOLDER$DATA_ID
echo "Starting Predict..."
./predict.py $ModelName $TestData $ResName
