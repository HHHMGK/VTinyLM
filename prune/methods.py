import torch
import gc
from tqdm import tqdm

def ranking_by_grads(model, input, avg=False, T_order=1, batch_size=0):
    model.eval()
    model.zero_grad(set_to_none=True)

    input_size = input.shape[0]
    batch_size = batch_size if batch_size > 0 else input_size
    num_steps = (input_size + batch_size - 1) // batch_size

    if model.model_type == 'llama':
        sequential = model.model.layers
    else:
        sequential = model.transformer.blocks
    importance_list = [0.0] * len(sequential)

    for i in tqdm(range(0, input_size, batch_size), desc="Processing batches"):
        batch_input = input[i:min(i+batch_size, input_size)]  # Process a smaller batch
        # print('doing batch', i, i+batch_size)
        # print('batch shape', batch_input.shape)
        loss = model(batch_input, labels=batch_input).loss
        # loss = loss / input_size  # Normalize the loss
        loss.backward()
        
        for id, layer in enumerate(sequential):
            importance = 0
            for name, param in layer.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # for each weight, multiply it by the gradient for first-order
                    layer_importance = (param.grad * param).abs().sum().item()
                    if T_order >= 2:
                        # second order
                        # !!! not correct for now
                        layer_importance += -1.0/2 * layer_importance * layer_importance
                    if avg:
                        layer_importance /= param.numel()
                    importance += layer_importance
            importance_list[id] += importance
        del loss, batch_input
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    # normalize from multibatch -> 1 
    importance_list = [imp / num_steps for imp in importance_list]
    return importance_list

def ranking_by_magnitude(model, norm='l1', avg=False, target=None):
    importance_list = []
    if model.model_type == 'llama':
        sequential = model.model.layers
    else:
        sequential = model.transformer.blocks
    for id, layer in enumerate(sequential):
        layer_importance = 0.0
        for name, param in layer.named_parameters():
            if(target is not None and target not in name):
                continue
            if(norm == 'l1'):
                imp = param.detach().abs().sum(dtype=torch.float32).item()
            elif(norm == 'l2'):
                imp = param.detach().pow(2).sum(dtype=torch.float32).sqrt().item()
            if avg:
                imp /= param.numel()
            layer_importance += imp
        # print('-=-=-=-=-=-=-=-=-=- layer ',id,': ',layer_importance)
        importance_list.append(layer_importance)
    return importance_list


def ranking_by_activation(model, batch_input, avg=False):
    model.eval()
    # importance_list = [0.0] * len(sequential)
    activations = {}
    hooks = []
    def get_activation(name):
        def hook(module, input, output):
            act_val = output[0].detach().abs().mean().item()
            if name in activations:
                activations[name].append(act_val)
            else:
                activations[name] = [act_val]
        return hook
    if model.model_type == 'llama':
        sequential = model.model.layers
    else:
        sequential = model.transformer.blocks
    for i, layer in enumerate(sequential):
        hook = layer.register_forward_hook(get_activation(f"layer_{i}"))
        hooks.append(hook)

    try:
        with torch.no_grad():
            model(batch_input)
        for hook in hooks:
            hook.remove()
    except Exception as e:
        print(f"Error during forward pass: {e}")
        for hook in hooks:
            hook.remove()
        raise e

    importance_list = []
    prev = 0
    for i in range(len(sequential)):
        acts = activations.get(f"layer_{i}", [0])
        layer_importance = sum(acts)
        if avg:
            layer_importance /= len(acts)
        importance_list.append(layer_importance - prev)
        prev = layer_importance

    # print(sample_output)
    # for k,v in activations.items():
    #     print(k,v)
    return importance_list
