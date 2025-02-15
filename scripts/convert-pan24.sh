#!/usr/bin/env bash

# Excludes german, high-temperature, alpaca, gpt2
genai-detection dataset convert \
    -h data/datasets/pan24/human \
    -m data/datasets/pan24/machines \
    --test-ids data/datasets/pan24/ids-test.txt \
    --recursive \
    --model-name-parent 1 \
    --val-split-size 2000 \
    --output data/datasets/pan24-converted
