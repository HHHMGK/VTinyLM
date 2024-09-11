import argparse
import os, gc
from model import load_model, load_tokenizer, layer_reduction
from eval import eval_perplexity
import torch
import json, csv

# Take arg from command line
parser = argparse.ArgumentParser(description='')
parser.add_argument('run_mode', type=str, default='train', choices=['train','eval'], help='Mode run mode')
parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
# parser.add_argument('--verbose', type=int, default=0, choices=[0,1,2], help='Verbose mode, 0 = silent, 1 = print log, 2 = print all')

# For EVALuating mode
parser.add_argument('--benchmark', type=str, default='perplexity-vn', choices=['perplexity-vn','perplexity-en','villm-eval'], help='Benchmark to evaluate')
parser.add_argument('--model', type=str, default='', help='Base model name')
parser.add_argument('--modification', type=str, default='layer_reduction', choices=['layer_reduction'], help='Model modification method')

# Import config from config.json

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.run_mode == 'train':
    print('Training')
    print('Config path:', args.config)
    
if args.run_mode == 'eval':
    print('Evaluating with benchmark:', args.benchmark)
    
    print('Loading model:', args.model)
    base_model = load_model(args.model)
    tokenizer = load_tokenizer(args.model)
    print('Model loaded')
    results = []
    if args.benchmark == 'perplexity-vn':
        results.append(['Modification', 'Perplexity'])
        if args.modification == 'layer_reduction':
            while True:
                model, layer_start, layer_end = layer_reduction(base_model)
                if model is None:
                    break
                print(f'Evaluating model with layers from {layer_start} to {layer_end} removed')
                perplexity = eval_perplexity(model, tokenizer, device, lang='vn')
                print(f'Perplexity: {perplexity}')
                results.append([f'Removed {layer_start} to {layer_end}', perplexity])
                
                del model
                gc.collect()
                torch.cuda.empty_cache()

    with open('results.csv','w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(results)
