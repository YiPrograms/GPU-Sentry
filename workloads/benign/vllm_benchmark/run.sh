#! /bin/bash

source .venv/bin/activate
vllm bench throughput \
	--model /home/yi/models/Llama-3.2-3B-Instruct \
	--dataset-name sharegpt \
	--dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
	--num-prompts 100
