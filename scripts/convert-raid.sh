#!/usr/bin/env bash

genai-detection dataset convert \
    --csv-file data/datasets/raid/train-good-models.csv \
    --text-col generation \
    --label-col model \
    --output data/datasets/raid-good-models-converted \
    --val-split-size 5200 \
    --test-split-size 15000
