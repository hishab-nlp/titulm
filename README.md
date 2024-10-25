# TituLM
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-TituLM-blue)](https://huggingface.co/hishab)


TituLM is a collection of open source LLMs trained by Hishab for the purpose of better language understanding and generation capabilities. At Hishab, we are pushing the boundaries of what is possible with LLMs for product development. According to this pipeline we are pretraining and finetuning models on a variety of tasks to improve the capabilities of the models. Although TituLM is not bound to any specific language but it mostly focuses on Bangla language.

## Models

### TituLM Llama Family
We have trained multiple variants of Llama 3.2 family models with different sizes and configurations. Our released models are:

__3B Family__
- [TituLM-Llama-3.2-3B-v2.0](https://huggingface.co/hishab/titulm-llama-3.2-3b-v2.0): Model trained with 37B Bangla tokens and the tokenizer is extended with 42k Bangla tokens.
- [TituLM-Llama-3.2-3B-v1.1](https://huggingface.co/hishab/titulm-llama-3.2-3b-v1.1): Model trained with 8.5B Bangla tokens with original llama 3.2 tokenizer.
- [TituLM-Llama-3.2-3B-v1.0](https://huggingface.co/hishab/titulm-llama-3.2-3b-v1.0): Model trained with 6B Bangla tokens with original llama 3.2 tokenizer.

__1B Family__
- [TituLM-Llama-3.2-1B-v2.0](https://huggingface.co/hishab/titulm-llama-3.2-1b-v2.0): Model trained with 37B Bangla tokens and the tokenizer extended with 42k Bangla tokens.
- [TituLM-Llama-3.2-1B-v1.1](https://huggingface.co/hishab/titulm-llama-3.2-1b-v1.1): Model trained with 8.5B Bangla tokens with original llama 3.2 tokenizer.
- [TituLM-Llama-3.2-1B-v1.0](https://huggingface.co/hishab/titulm-llama-3.2-1b-v1.0): Model trained with 6B Bangla tokens with original llama 3.2 tokenizer.

### TituLM Gemma Family
We have trained multiple variants of Gemma family models with different sizes and configurations. Our released models are:

__Gemma 2 2B__
- [TituLM-Gemma-2-2B-v1.1](https://huggingface.co/hishab/titulm-gemma-2-2b-v1.1): Model trained with 4.4B Bangla tokens with original gemma 2 tokenizer.
- [TituLM-Gemma-2-2B-v1.0](https://huggingface.co/hishab/titulm-gemma-2-2b-v1.0): Model trained with 3B Bangla tokens with original gemma 2 tokenizer.

### TituLM MPT Family
- [titulm-mpt-1b-v1.0](https://huggingface.co/hishab/titulm-mpt-1b-v1.0): Trained with 4.51B Bangla tokens with custom Bangla tokenizer.
- [titulm-mpt-1b-v2.0](https://huggingface.co/hishab/titulm-mpt-1b-v2.0): Trained with 43B Bangla and English tokens with custom Bangla+English tokenizer.

## Usage
### Generation using transformers

```python
# pip install transformers
import torch
from transformers import pipeline

model_id = "hishab/titulm-llama-3.2-3b-v2.0"

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

pipe("আমাদের দেশের নাম")
```

## Benchmark
- Clone the forked version of [lm-evaluation-harness](https://github.com/hishab-nlp/lm-evaluation-harness) repository.
- Now run the following commands to evaluate the models.
- Pass required arguments according to your needs.

```bash
git clone https://github.com/hishab-nlp/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .


# Running the benchmark
# https://github.com/hishab-nlp/lm-evaluation-harness/blob/main/scripts/bangla_lm_benchmark.py
cd scripts
# pass arguemnt according to your needs
python bangla_lm_benchmark.py

```