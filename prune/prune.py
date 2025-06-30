from .methods import ranking_by_gradient, ranking_by_magnitude, ranking_by_activation
import copy
from math import isnan

def get_transformer_sequential(model):
    """
    Get the transformer sequential layers of the model.
    """
    if (hasattr(model, 'model_type') and model.model_type == 'phogpt') or hasattr(model, 'transformer'):
        return model.transformer.blocks
    elif (hasattr(model, 'model_type') and model.model_type in ['llama','qwen']) or hasattr(model, 'model'):
        return model.model.layers
    else:
        raise ValueError("Model does not have transformer or model attribute.")
    
def normalize(arr):
    mi = min(arr)
    ma = max(arr)
    print(f"Min: {mi}, Max: {ma}")
    print(arr)
    for i in range(len(arr)):
        if isnan(arr[i]):
            arr[i] = 0
        else:
            arr[i] = (arr[i] - mi)/(ma - mi)
    print(arr)
    return arr

def estimate_importance(model, method='magnitude', prune_data=None, avg=False, 
                        norm='l1', target=None, T_order=1, batch_size=16):
    """
    Estimate the importance of model layers using different methods.
    """
    if method == 'magnitude':
        return ranking_by_magnitude(model, norm=norm, avg=avg, target=target)
    elif method == 'gradient':
        return ranking_by_gradient(model, prune_data, avg=avg, T_order=T_order, batch_size=batch_size)
    elif method == 'activation':
        return ranking_by_activation(model, prune_data, avg=avg)
    elif method == 'combine':
        mag = ranking_by_magnitude(model, norm=norm, avg=avg, target=target)
        mag = normalize(mag)
        print(f"Mag: {mag}")
        grad = ranking_by_gradient(model, prune_data, avg=avg, T_order=T_order, batch_size=batch_size)
        grad = normalize(grad)
        print(f"gradient: {grad}")
        act = ranking_by_activation(model, prune_data, avg=avg)
        act = normalize(act)
        print(f"Act: {act}")
        return [m*0.45 + g*0.45 + a*0.1 for m, g, a in zip(mag, grad, act)]

    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'magnitude', 'gradient', or 'activation'")
    
    return rankings

def prune_model_generator(model, rankings, pruning_rate=[], pruning_layer_num=[]):
    """
    Generate (yield) the pruned model based on the given rankings and pruning rate or number of layers.
    Returns the pruned layers.
    """
    if pruning_layer_num:
        num_layers = pruning_layer_num
    elif pruning_rate:
        num_layers = [int(len(rankings) * x) for x in pruning_rate]
    else:
        num_layers = [1]
    num_layers.sort()
    sequential = get_transformer_sequential(model)
    layers = range(len(rankings))
    layers_to_prune = sorted(layers, key=rankings.__getitem__)[:num_layers[-1]]
        
    prev = 0
    for num_to_prune in num_layers:
        for i in range(prev, num_to_prune):    
            prune_layer = layers_to_prune[i]
            for j in range(i):
                if layers_to_prune[j] < layers_to_prune[i]:
                    prune_layer -= 1
            del sequential[prune_layer]
        yield model, layers_to_prune[:num_to_prune]
        prev = num_to_prune
    
    return None

def serial_pruning_model_generator(model, num_layers = None, step = None):
    """
    Generate (yield) models with layers removed in a serial manner.
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
            yield model, range(i, i+n)
            for j, block in enumerate(del_blocks):
                if i + j < len(sequential):
                    sequential.insert(i + j, block)
                else:
                    sequential.append(block)
            i += step

    return None
