#!/usr/bin/env bash

 genai-detection dataset adapt-hf-dataset yaful/MAGE \
    --model-col src \
    --output data/datasets/mage-converted \
    --human-label 1
