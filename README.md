# SuperTransformer

Suptertransformer that auto loads Huggingface models 

# Introduction
This is a single line transformer for easy to load models from Huggingface.  It is not to replace Huggingface Transformer process.  It simplifies it and speed up the loading the process of the HuggingFace models

# Usage
SuperTransformers download the model locally.  The super class uses AutoTokenizer and AutoModelForCausalLM.from_pretrained.

# Installation
``` bash
pip install bitsandbytes>=0.39.0
pip install --upgrade accelerate transformers
```
# Setup Virtual Environment
Create a virtual environment with uv (refer to Installation for installation instructions), a fast Rust-based Python package and project manager.
``` bash
uv venv my-env
source my-env/bin/activate
```

# How to run
```python
python SuperTransformer.py
```

# Example of usage:
 
```python
import SuperTransformer
# Load SuperTransformer Class,  (1) Loads Huggingface model, (2) System Prompt (3) Text/prompt (4)Max tokens
SuperTransformers = SuperTransformers("EpistemeAI/ReasoningCore-3B-RE1-V2","You are a highly knowledgeable assistant with expertise in chemistry and physics. <reasoning>","What is the area of a circle, radius=16, reason step by step", 2026)
# 8-bit quantization
SuperTransformers.HuggingFaceTransformer8bit()
# or 4-bit quantization
SuperTransformers.HuggingFaceTransformer4bit()
```

## Returns model and tokenizer
```python
import SuperTransformer
SuperTransformers = SuperTransformers("EpistemeAI/ReasoningCore-3B-RE1-V2")
model, tokenizer = HuggingfaceTransfomer()  #returns the model and tokenizer
```
## returns pipline as higher helper
```python
import SuperTransformer
SuperTransformers = SuperTransformers("EpistemeAI/ReasoningCore-3B-RE1-V2")
pipe = HuggingfacePipeline()  #returns the pipeline only
output = pipe(self.text, max_new_tokens=self.max_new_tokens)  # Limit output length to save memory
# Print the generated output
print(output)
```

## example
Example in Colab- [supertransformer.ipynb](supertransformer.ipynb)

