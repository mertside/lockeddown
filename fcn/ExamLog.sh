#!/bin/bash

cat $1 | grep -o 'val_accuracy: [0-9].[0-9]*' | grep -o [0-9].[0-9]*
