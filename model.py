from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import copy

def load_model(model_name):
    return AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def clone_model(model):
    return copy.deepcopy(model)

def layer_reduction(model, num_layers = None):
    if num_layers is None:
        num_layers = [1,2,4,10]
    max_len = len(model.transformer.blocks)
    for n in num_layers:
        i = 0
        while i + n - 1 < max_len:
            # Memory intensive
            # new_model = clone_model(model)
            # del new_model.transformer.blocks[i:i+n-1]   
            # yield model, i, i+n-1
            # i+=n
            
            # Memory efficient
            del_blocks = list(copy.deepcopy(model.transformer.blocks[i:i+n]))
            del model.transformer.blocks[i:i+n]
            yield model, i, i+n-1
            model.transformer.blocks[i:i+n] = del_blocks
            del del_blocks
            i+=n

    return None
    