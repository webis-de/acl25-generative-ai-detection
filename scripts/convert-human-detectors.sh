#!/usr/bin/env bash

genai-detection dataset convert \
    --json-file-single data/datasets/human-detectors/human_detectors.json \
    --text-col article \
    --label-col ground_truth \
    --model-col generation_model \
    --human-label Human-written \
    --output data/datasets/human-detectors-converted \
    --val-split-size 0.15 \
    --test-split-size 0.1
