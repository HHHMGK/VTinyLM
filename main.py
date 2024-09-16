import argparse
import os, gc, time
from model import load_model, load_tokenizer, layer_reduction
from eval import eval_perplexity
import torch
import json, csv

# Take arg from command line
parser = argparse.ArgumentParser(description='')
parser.add_argument('run_mode', type=str, default='train', choices=['train','eval','infer'], help='Mode run mode')
parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
# parser.add_argument('--verbose', type=int, default=0, choices=[0,1,2], help='Verbose mode, 0 = silent, 1 = print log, 2 = print all')
parser.add_argument('--model', type=str, default='', help='Base model name')

# For TRAINing mode

# For EVALuating mode
parser.add_argument('--benchmark', type=str, default='perplexity-vn', choices=['perplexity-vn','perplexity-en','villm-eval'], help='Benchmark to evaluate')
parser.add_argument('--repeat', type=int, default=1, help='Number of evaluation to repeat')
parser.add_argument('--modification', type=str, default='layer_reduction', choices=['layer_reduction'], help='Model modification method')
parser.add_argument('--eval_base', type=bool, default=True, help='Evaluate base model or not')
parser.add_argument('--output', type=str, default='results.csv', help='Output file for results')

# For INFERing mode
parser.add_argument('--prompt', type=str, default='', help='Input prompt for inference')
parser.add_argument('--file', type=str, default='', help='Input file for inference')
# parser.add_argument('--output', type=str, default='output.txt', help='Output file for results')
# Import config from config.json

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_time = time.time()

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
        # results.append(['Modification', 'Perplexity'])

        if args.eval_base:
            print('Evaluating base model')
            perplexities = eval_perplexity(base_model, tokenizer, device, lang='vn', repeat=args.repeat)
            print('Perplexity:', perplexities)
            results['base'] = perplexities
        
        if args.modification == 'layer_reduction':
            new_model_generator = layer_reduction(base_model)
            while True:
                model, layer_start, layer_end = next(new_model_generator, (None, None, None))
                if model is None:
                    break
                print(f'Evaluating model with layers from {layer_start} to {layer_end} removed')
                perplexities = eval_perplexity(model, tokenizer, device, lang='vn', repeat=args.repeat)
                print('Perplexity:', perplexities)
                # results.append([f'Removed {layer_start} to {layer_end}', perplexity])
                results[f'Removed {layer_start}-{layer_end}'] = perplexities
                
                del model
                gc.collect()
                torch.cuda.empty_cache()
        
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

        if args.repeat > 1:
            for k,v in results.items():
                v_mean = sum(v)/len(v)
                v_var = sum([(x-v_mean)**2 for x in v])/(len(v)-1)
                v_std = v_var**0.5
                results[k].extend([v_mean, v_var, v_std, f'{v_mean}±{v_std}'])

    with open(args.output,'w') as f:
        csv_writer = csv.writer(f)
        for k,v in results.items():
            csv_writer.writerow([k]+v)


if args.run_mode == 'infer':
    print('Infering')
    
    input = []
    if args.prompt:
        input.append(args.prompt)

    if args.file:
        with open(args.file,'r') as f:
            input.extend(f.readlines())
    
    model = load_model(args.model)
    tokenizer = load_tokenizer(args.model)
    
print('Time elapsed:', time.time()-start_time)