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
    return model

def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def clone_model(model):
    return copy.deepcopy(model)

def layer_removal(model, start_layer, end_layer):
    del model.transformer.blocks[start_layer:end_layer]
    return model

def layer_reduction_model_generator(model, num_layers = None, step = None):
    if num_layers is None:
        num_layers = [1,2,4,8]
    max_len = len(model.transformer.blocks)
    for n in num_layers:
        i = 0
        if step is None:
            step = n
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
            # model.transformer.blocks[i:i+n] = del_blocks
            # del del_blocks
            for j, block in enumerate(del_blocks):
                if i + j < len(model.transformer.blocks):
                    model.transformer.blocks.insert(i + j, block)
                else:
                    model.transformer.blocks.append(block)
            i+=step

    return None
    
def ranking_layers_importance(model, intput_data):
    model.eval()
    model.zero_grad()
    # criterion = nn.CrossEntropyLoss()
    input_ids = intput_data['input_ids']
    labels = intput_data['labels']
    
    # outputs = model(input_ids)
    # logits = outputs.logits
    # loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss = model(input_ids, labels=labels).loss
    model.zero_grad()
    loss.backward()
    
    # for name, param in model.named_parameters():
    #     if param.requires_grad and param.grad is not None:
    #         grads.append((name, param.grad.clone()))
    layer_importance = []
    for layer in model.transformer.blocks:
        importance = []
        for name, param in layer.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad.clone()
                # for each weight, multiply it by the gradient
                importance.append((grad * param).abs().mean().item())
        layer_importance.append(sum(importance) / len(importance))
        
    del loss
    print('Layer importance:', layer_importance, sep='\n')
    
    return layer_importance
