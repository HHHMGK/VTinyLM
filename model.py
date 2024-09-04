from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import copy

def load_model(model_name):
    return AutoModel.from_pretrained(model_name, trust_remote_code=True)

def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def clone_model(model):
    return copy.deepcopy(model)

    
    