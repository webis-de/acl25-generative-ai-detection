#!/usr/bin/env bash

set -e
docker build -t registry.webis.de/code-research/authorship/generative-ai-detection "$@" .
docker push registry.webis.de/code-research/authorship/generative-ai-detection
