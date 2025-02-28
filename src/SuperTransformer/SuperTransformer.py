# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
import torch
from accelerate import Accelerator

class SuperTransformers:
  def __init__(self, model, system_prompt, text, max_new_tokens):
    self.model = model
    self.system_prompt = system_prompt
    self.text = text
    self.max_new_tokens = max_new_tokens

  def HuggingfaceTransfomer(self):
    tokenizer = AutoTokenizer.from_pretrained(self.model)
    model = AutoModelForCausalLM.from_pretrained(self.model)
    return model, tokenizer

  def HuggingfacePipeline(self):
    tokenizer = AutoTokenizer.from_pretrained(self.model)
    model = AutoModelForCausalLM.from_pretrained(self.model)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

  def HuggingFaceTransformer4bit(self):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",  # Reduces precision, saving memory
        bnb_4bit_use_double_quant=True  # Further memory savings
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(self.model)  # Replace with your actual model name

    # Load quantized model
    model_4bit = AutoModelForCausalLM.from_pretrained(
        self.model,  # Replace with your actual model name
        quantization_config=quantization_config,
        device_map="auto"  # Ensures model is loaded on available GPU
    )

    # Use pipeline with the correct model
    pipe = pipeline("text-generation", model=model_4bit, tokenizer=tokenizer)  # Set device=0 for GPU
    
    #Define the system prompt and the user prompt
    #Combine the system prompt with the user prompt. The format here follows a common convention for chat-like interactions.
    full_prompt = f"System: {self.system_prompt}\nUser: {self.text}\nAssistant:"

    # Generate output with memory optimization
    output = pipe(self.text, max_new_tokens=self.max_new_tokens)  # Limit output length to save memory

    # Print the generated output
    print(output)
  
  def HuggingFaceTransformer8bit(self):
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(self.model)  # Replace with your actual model name

    # Load quantized model
    model_8bit = AutoModelForCausalLM.from_pretrained(
        self.model,  # Replace with your actual model name
        quantization_config=quantization_config,
        device_map="auto"  # Ensures model is loaded on available GPU
    )

    # Use pipeline with the correct model
    pipe = pipeline("text-generation", model=model_8bit, tokenizer=tokenizer)  # Set device=0 for GPU
    
    #Define the system prompt and the user prompt
    #Combine the system prompt with the user prompt. The format here follows a common convention for chat-like interactions.
    full_prompt = f"System: {self.system_prompt}\nUser: {self.text}\nAssistant:"

    # Generate output with memory optimization
    output = pipe(self.text, max_new_tokens=self.max_new_tokens)  # Limit output length to save memory

    # Print the generated output
    print(output)
