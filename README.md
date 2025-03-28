# Using LLMs for Relevance Judgment Assessment
This repository contains the code used in the following papers:

(1) <a href="https://arxiv.org/pdf/2405.05600">Can We Use Large Language Models to Fill Relevance Judgment Holes?</a>

(2) <a href="">Improving the Reusability of Conversational Search Test Collections.</a>

This repo provides the scripts for using LLMs to generate relevance labels of TREC CAsT 2022 and TREC iKAT 2023 datasets using zero-, one-, and two-shot prompts.

## Instructions
Run the following code for doing relevance judgment with different prompts using open source LLMs:

One-shot prompting:

```bash
python gpt-one-shot.py \
--dataset_name "cast22" \
--from_checkpoint False \
--model_id "gpt-3.5-turbo-0125" \
--api_key  "YOUR_API_KEY" \
--use_context False
```

Zero-shot prompting:

```bash
python gpt-zero-shot.py \
--dataset_name "cast22" \
--from_checkpoint False \
--model_id "gpt-3.5-turbo-0125" \
--api_key  "YOUR_API_KEY" 
```


Two-shot prompting:

```bash
python gpt-two-shot.py \
--dataset_name "cast22" \
--from_checkpoint False \
--model_id "gpt-3.5-turbo-0125" \
--api_key  "YOUR_API_KEY" 
```
Note that the ``dataset_name`` can be either "cast22" or "ikat23".
