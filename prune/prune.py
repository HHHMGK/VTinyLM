from .methods import ranking_by_grads, ranking_by_magnitude, ranking_by_activation
import copy
from math import isnan

def get_transformer_sequential(model):
    """
    Get the transformer sequential layers of the model.
    """
    if (hasattr(model, 'model_type') and model.model_type == 'phogpt') or hasattr(model, 'transformer'):
        return model.transformer.blocks
    elif (hasattr(model, 'model_type') and model.model_type == 'llama') or hasattr(model, 'model'):
        return model.model.layers
    else:
        raise ValueError("Model does not have transformer or model attribute.")
def normalize(arr):
    mi = min(arr)
    ma = max(arr)
    print(f"Min: {mi}, Max: {ma}")
    print(arr)
    for x in arr:
        if isnan(x):
            x = 0
        else:
            x = (x - mi)/(ma - mi)
    print(arr)
    return arr
def estimate_importance(model, method='magnitude', prune_data=None, avg=False, 
                        norm='l1', target=None, T_order=1, batch_size=16):
    """
    Estimate the importance of model layers using different methods.
    """
    if method == 'magnitude':
        return ranking_by_magnitude(model, norm=norm, avg=avg, target=target)
    elif method == 'grads':
        return ranking_by_grads(model, prune_data, avg=avg, T_order=T_order, batch_size=batch_size)
    elif method == 'activation':
        return ranking_by_activation(model, prune_data, avg=avg)
    elif method == 'combine':
        mag = ranking_by_magnitude(model, norm=norm, avg=avg, target=target)
        mag = normalize(mag)
        print(f"Mag: {mag}")
        grads = ranking_by_grads(model, prune_data, avg=avg, T_order=T_order, batch_size=batch_size)
        grads = normalize(grads)
        print(f"Grads: {grads}")
        act = ranking_by_activation(model, prune_data, avg=avg)
        act = normalize(act)
        print(f"Act: {act}")
        return [m*0.45 + g*0.45 + a*0.1 / 3 for m, g, a in zip(mag, grads, act)]

    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'magnitude', 'grads', or 'activation'")
    
    return rankings

def prune_model(model, rankings, pruning_rate=0.2, targets=[]):
    """
    Prune the targets module of the model based on the given rankings and pruning rate.
    Returns the pruned layers.
    """
    num_layers = int(len(rankings) * pruning_rate)
    layers = range(len(rankings))
    layers_to_prune = sorted(layers, key=rankings.__getitem__)[:num_layers]
    layers_to_prune.sort(reverse=True) # Sort in descending order for safe indexing later in removal (remove from the end)
    print(f"Layers to prune: {layers_to_prune}")
    # Prune the model
    sequential = get_transformer_sequential(model)
    for layer in layers_to_prune:
        if not targets:
            del sequential[layer]
        else:
            pass
            # for target in targets:
            #     if target in sequential[layer].named_modules():
            #         sequential[layer].remove_module(target)
    
    
    return layers_to_prune

def serial_pruning_model_generator(model, num_layers = None, step = None):
    """
    Generate models with layers removed in a serial manner.
    """
    if num_layers is None:
        num_layers = [1,2,4,8]
    sequential = get_transformer_sequential(model)
    max_len = len(sequential)
    for n in num_layers:
        i = 0
        if step is None:
            step = n
        while i + n - 1 < max_len:
            # Memory intensive
            # new_model = clone_model(model)
            # del new_sequential[i:i+n-1]   
            # yield model, i, i+n-1
            # i+=n
            
            # Memory efficient
            del_blocks = list(copy.deepcopy(sequential[i:i+n]))
            del sequential[i:i+n]
            yield model, i, i+n-1
            for j, block in enumerate(del_blocks):
                if i + j < len(sequential):
                    sequential.insert(i + j, block)
                else:
                    sequential.append(block)
            i+=step

    return None
