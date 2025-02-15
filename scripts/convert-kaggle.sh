#!/usr/bin/env bash

genai-detection dataset convert \
     --csv-file data/datasets/kaggle/Training_Essay_Data.csv \
     --label-col generated \
     --val-split-size 3000 \
     --test-split-size 3000 \
     --output data/datasets/kaggle-converted
