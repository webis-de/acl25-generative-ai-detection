#!/usr/bin/env bash

# Excludes german, high-temperature, alpaca, gpt2
genai-detection dataset convert \
    -h data/datasets/pan24-extended/human \
    -m data/datasets/pan24-extended/machines \
    --test-ids data/datasets/pan24-extended/ids-test.txt \
    --recursive \
    --model-name-parent 1 \
    --val-split-size 2000 \
    --output data/datasets/pan24-extended-converted
