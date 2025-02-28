# SuperTransformer

Suptertransformer that auto loads Huggingface models 

# Introduction
This is a single line transformer for easy to load models from Huggingface.

# Usage
SuperTransformers download the model locally.  The super class uses AutoTokenizer and AutoModelForCausalLM.from_pretrained.

# Example of usage:
 
```python
# (1) Loads Huggingface model, (2) System Prompt (3) Text (4)Max tokens
SuperTransformers = SuperTransformers("EpistemeAI/ReasoningCore-3B-RE1-V2","You are a highly knowledgeable assistant with expertise in chemistry and physics. <reasoning>","What is the area of a circle, radius=16, reason step by step", 2026)
# 8-bit quantization
SuperTransformers.HuggingFaceTransformer8bit()
# or 4-bit quantization
SuperTransformers.HuggingFaceTransformer4bit()
```

## Returns model and tokenizer
```python
SuperTransformers = SuperTransformers("EpistemeAI/ReasoningCore-3B-RE1-V2")
model, tokenizer = HuggingfaceTransfomer()  #returns the model and tokenizer
```
## returns pipline as higher helper
```python
SuperTransformers = SuperTransformers("EpistemeAI/ReasoningCore-3B-RE1-V2")
pipe = HuggingfacePipeline()  #returns the pipeline only
output = pipe(self.text, max_new_tokens=self.max_new_tokens)  # Limit output length to save memory
# Print the generated output
print(output)
  
```

