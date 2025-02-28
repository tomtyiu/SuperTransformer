# SuperTransformer

Suptertransformer for Huggingface Transformers class

# Introduction
This is a single line transformer for easy to load models from Huggingface.

# Usage
SuperTransformers download the model locally.  The super class usea AutoTokenizer and AutoModelForCausalLM.from_pretrained.

# Example of usage:
```python
# (1) Loads Huggingface model, (2) System Prompt (3) Text 
SuperTransformers = SuperTransformers("EpistemeAI/ReasoningCore-3B-RE1-V2","You are a highly knowledgeable assistant with expertise in chemistry and physics. <reasoning>","What is the area of a circle, radius=16, reason step by step", 2026)
SuperTransformers.HuggingFaceTransformer8bit()
```
