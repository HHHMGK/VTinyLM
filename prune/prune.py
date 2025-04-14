from .methods import ranking_by_grads, ranking_by_magnitude, ranking
_by_activation
import copy

def estimate_importance(model, method='magnitude', input_data=None, avg=False, 
                        norm='l1', target=None, T_order=1, batch_size=0):
    """
    Estimate the importance of model layers using different methods.
    """
    if method == 'magnitude':
        return ranking_by_magnitude(model, norm=norm, avg=avg, target=target)
    
    elif method == 'grads':
        if input_data is None:
            raise ValueError("Input data is required for gradient-based importance estimation")
        return ranking_by_grads(model, input_data, avg=avg, T_order=T_order, batch_size=batch_size)
    
    elif method == 'activation':
        if input_data is None:
            raise ValueError("Input data is required for activation-based importance estimation")
        return ranking
        _by_activation(model, input_data, avg=avg)
    elif method == 'combine':
        if input_data is None:
            raise ValueError("Input data is required for gradient-based and activation-based importance estimation")
        grads = ranking_by_grads(model, input_data, avg=avg, T_order=T_order, batch_size=batch_size)
        mag = ranking_by_magnitude(model, norm=norm, avg=avg, target=target)
        act = ranking
        _by_activation(model, input_data, avg=avg)
        return [(g + m + a) / 3 for g, m, a in zip(grads, mag, act)]
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'magnitude', 'grads', or 'activation'")

def prune_model(model, rankings, pruning_rate=0.2, targets=[], model_type='phogpt'):
    """
    Prune the targets module of the model based on the given rankings and pruning rate.
    """
    num_layers = int(len(rankings) * pruning_rate)
    layers = range(len(rankings))
    layers_to_prune = sorted(layers, key=rankings.__getitem__)[:num_layers]
    layers_to_prune.sort(reverse=True) # Sort in descending order for safe indexing later in removal (remove from the end)
    # Prune the model
    sequential = []
    if model_type == 'phogpt':
        sequential = model.transformer.blocks
    elif model_type == 'llama':
        sequential = model.model.layers
    for layer in layers_to_prune:
        if not targets:
            del sequential[layer]
        else:
            pass
            # for target in targets:
            #     if target in sequential[layer].named_modules():
            #         sequential[layer].remove_module(target)
    
    
    return model

def serial_pruning_model_generator(model, num_layers = None, step = None):
    """
    Generate models with layers removed in a serial manner.
    """
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
            for j, block in enumerate(del_blocks):
                if i + j < len(model.transformer.blocks):
                    model.transformer.blocks.insert(i + j, block)
                else:
                    model.transformer.blocks.append(block)
            i+=step

    return None
