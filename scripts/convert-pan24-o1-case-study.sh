#!/usr/bin/env bash

genai-detection dataset convert \
    -h data/datasets/pan24-extended/human \
    -m data/datasets/pan24-extended/machines/openai-o1 \
    --test-ids data/datasets/pan24-extended/ids-test.txt \
    --model-name-parent 1 \
    --recursive \
    --val-split-size 0.1 \
    --output data/datasets/pan24-o1-case-study-converted
