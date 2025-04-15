from .methods import ranking_by_grads, ranking_by_magnitude, ranking_by_activation
import copy

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
    
def estimate_importance(model, method='magnitude', prune_data=None, avg=False, 
                        norm='l1', target=None, T_order=1, batch_size=16):
    """
    Estimate the importance of model layers using different methods.
    """
    rankings = [0] * len(get_transformer_sequential(model))
    if method == 'magnitude' or method == 'combine':
        r = ranking_by_magnitude(model, norm=norm, avg=avg, target=target)
        rankings = [rankings[i] + r[i] for i in range(len(rankings))]
        print("Ranking by magnitude: ",rankings)
    
    if method == 'grads' or method == 'combine':
        r = ranking_by_grads(model, prune_data, avg=avg, T_order=T_order, batch_size=batch_size)
        rankings = [rankings[i] + r[i] for i in range(len(rankings))]
        print("Ranking by grads: ",rankings)
    
    if method == 'activation' or method == 'combine':
        r = ranking_by_activation(model, prune_data, avg=avg)
        rankings = [rankings[i] + r[i] for i in range(len(rankings))]
        print("Ranking by activation: ",rankings)
    
    if method == 'combine':
        rankings = [rankings[i] / 3 for i in range(len(rankings))]
    
    return rankings

def prune_model(model, rankings, pruning_rate=0.2, targets=[]):
    """
    Prune the targets module of the model based on the given rankings and pruning rate.
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
    
    
    # return model

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
