# Using LLMs for Relevance Judgment Assessment
This repository contains the code used in the following papers:

(1) <a href="https://arxiv.org/pdf/2405.05600">Can We Use Large Language Models to Fill Relevance Judgment Holes?</a>

(2) <a href="">Improving the Reusability of Conversational Search Test Collections.</a>

This repo provides the scripts for using LLMs to generate relevance labels of TREC CAsT 2022 and TREC iKAT 2023 datasets using zero-, one-, and two-shot prompts.

## Instructions
Run the following code for doing relevance judgment with one-shot labeling approach using open source LLMs:

```bash
python one-shot.py \
--dataset_name "cast22" \
--from_checkpoint False \
--model_id "gpt-3.5-turbo-0125" \
--api_key  "YOUR_API_KEY" \
--use_context False
```
