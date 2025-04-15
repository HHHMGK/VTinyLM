from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
# import bitsandbytes as bnb
import torch
import torch.nn as nn
import copy

def load_model(model_path, bnb='none', peft_path=None):
    bnb_config = None
    if bnb == '4bit':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif bnb == '8bit':
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, quantization_config=bnb_config)
    if peft_path is not None:
        model = PeftModel.from_pretrained(model, peft_path)
        model = model.merge_and_unload()
    add_modeltype(model, model_path)
    return model

def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def add_modeltype(model, name):
    """
    Add a name to the model.
    """
    if name == 'vinai/PhoGPT-4B-Chat':
        name = 'phogpt'
    elif name == 'meta-llama/Llama-3.2-3B-Instruct':
        name = 'llama'
    setattr(model, 'model_type', name)
