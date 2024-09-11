import argparse
import os, gc
from model import load_model, load_tokenizer, layer_reduction
from eval import eval_perplexity
import torch
import json

# Take arg from command line
parser = argparse.ArgumentParser(description='')
parser.add_argument('run_mode', type=str, default='train', choices=['train','eval'], help='Mode run mode')
parser.add_argument('--config', type=str, default='config.json', help='Path to config file')

# For EVALuating mode
parser.add_argument('--benchmark', type=str, default='perplexity-vn', choices=['perplexity-vn','perplexity-en','villm-eval'], help='Benchmark to evaluate')
parser.add_argument('--model', type=str, default='', help='Base model name')
parser.add_argument('--modification', type=str, default='layer_reduction', choices=['layer_reduction'], help='Model modification method')
# Import config from config.json

args = parser.parse_args()

if args.run_mode == 'train':
    print('Training')
    print('Config path:', args.config)
    
if args.run_mode == 'eval':
    print('Evaluating')
    print('Benchmark:', args.benchmark)
    
    base_model = load_model(args.model)
    tokenizer = load_tokenizer(args.model)
    results = []
    if args.benchmark == 'perplexity-vn':
        eval_perplexity(None, None, None, lang='vn')
        if args.modification == 'layer_reduction':
            while True:
                model, layer_start, layer_end = layer_reduction(base_model)
                if model is None:
                    break
                perplexity = eval_perplexity(model, tokenizer, None, lang='vn')
                results.append([f'Removed {layer_start} to {layer_end}', perplexity])
                
                del model
                gc.collect()
                torch.cuda.empty_cache()

    with open('results.json','w') as f:
        json.dump(results, f)
