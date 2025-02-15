#!/usr/bin/env bash

genai-detection dataset convert \
    -h data/datasets/ghostbuster-data/essay/human \
    -m data/datasets/ghostbuster-data/essay/claude \
    -m data/datasets/ghostbuster-data/essay/gpt \
    -m data/datasets/ghostbuster-data/essay/gpt_prompt1 \
    -m data/datasets/ghostbuster-data/essay/gpt_prompt2 \
    -m data/datasets/ghostbuster-data/essay/gpt_writing \
    -m data/datasets/ghostbuster-data/essay/gpt_semantic \
    -o data/datasets/ghostbuster-essay-converted \
    --val-split-size 600 \
    --test-split-size 600 \

# shellcheck disable=SC2046
# We cannot use --recursive, because folders have logprobs subfolders
genai-detection dataset convert \
    $(for i in data/datasets/ghostbuster-data/reuter/human/*; do \
        echo "-h data/datasets/ghostbuster-data/reuter/human/$(basename $i)"
        echo "-m data/datasets/ghostbuster-data/reuter/claude/$(basename $i)"
        echo "-m data/datasets/ghostbuster-data/reuter/gpt/$(basename $i)"
        echo "-m data/datasets/ghostbuster-data/reuter/gpt_prompt1/$(basename $i)"
        echo "-m data/datasets/ghostbuster-data/reuter/gpt_prompt2/$(basename $i)"
        echo "-m data/datasets/ghostbuster-data/reuter/gpt_writing/$(basename $i)"
        echo "-m data/datasets/ghostbuster-data/reuter/gpt_semantic/$(basename $i)"
    done) \
    --model-name-parent 1 \
    --output data/datasets/ghostbuster-reuters-converted \
    --val-split-size 600 \
    --test-split-size 600
