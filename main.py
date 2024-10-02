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

# Common arguments
parser.add_argument('--base_model', type=str, default='', help='Base model name')
parser.add_argument('--output', type=str, default='results.csv', help='Output file for results')
parser.add_argument('--measure_time', type=bool, default=False, help='Measure run time or not')
# parser.add_argument('--run_dummy', type=bool, default=False, help='Run dummy mode (system testing) or not')

# For TRAINing mode

# For EVALuating mode
parser.add_argument('--benchmark', type=str, default='perplexity-vn', choices=['perplexity-vn','perplexity-en','villm-eval'], help='Benchmark to evaluate')
parser.add_argument('--repeat', type=int, default=1, help='Number of evaluation to repeat')
parser.add_argument('--modification', type=str, default='layer_reduction', choices=['layer_reduction'], help='Model modification method')
parser.add_argument('--eval_base', type=bool, default=True, help='Evaluate base model or not')
parser.add_argument('--layer_step', type=int, default=0, help='Step for layer modification')

# For INFERing mode
parser.add_argument('--prompt', type=str, default='', help='Input prompt for inference')
parser.add_argument('--file', type=str, default='', help='Input file for inference')
# Import config from config.json

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.measure_time:
    start_time = time.time()

if args.run_mode == 'train':
    print('Training')
    print('Config path:', args.config)
    
if args.run_mode == 'eval':
    print('Evaluating with benchmark:', args.benchmark)
    
    print('Loading as base model:', args.base_model)
    base_model = load_model(args.base_model)
    tokenizer = load_tokenizer(args.base_model)
    print('Model loaded')
    results = []
    if args.benchmark == 'perplexity-vn':

        if args.eval_base:
            print('Evaluating base model')
            eval_results = eval_perplexity(base_model, tokenizer, device, lang='vn', repeat=args.repeat, measure_time=args.measure_time)
            print('Perplexity:', eval_results['perplexity'])
            # _perplexity = f'{round(eval_results["perplexity"][0],3)} 0xC2 {round(round(eval_results["perplexity"][1],3))}'
            # _time = f'{round(eval_results["time"][0],3)} 0xC2 {round(round(eval_results["time"][1],3))}' if args.measure_time else None
            results.append({'Modification':'Base model', 
                            'Perplexity_mean':eval_results['perplexity'][0], 'Perplexity_stddev':eval_results['perplexity'][1], 
                            'Time_mean':eval_results['time'][0], 'Time_stddev':eval_results['time'][1]})
            
        if args.modification == 'layer_reduction':
            new_model_generator = layer_reduction(base_model, num_layers = None, step=args.layer_step)
            while True:
                model, layer_start, layer_end = next(new_model_generator, (None, None, None))
                if model is None:
                    break
                print(f'Evaluating model with layers from {layer_start} to {layer_end} removed')
                eval_results = eval_perplexity(model, tokenizer, device, lang='vn', repeat=args.repeat, measure_time=args.measure_time)
                print('Perplexity:', eval_results['perplexity'])
                results.append({'Modification':f'Removed {layer_start} to {layer_end}', 
                            'Perplexity_mean':eval_results['perplexity'][0], 'Perplexity_stddev':eval_results['perplexity'][1], 
                            'Time_mean':eval_results['time'][0], 'Time_stddev':eval_results['time'][1]})
                
                del model
                gc.collect()
                torch.cuda.empty_cache()
        
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

    with open(args.output,'w') as f:
        header = ['Modification', 'Perplexity_mean', 'Perplexity_stddev', 'Time_mean', 'Time_stddev']
        csv_writer = csv.DictWriter(f, fieldnames=header)
        csv_writer.writeheader()
        csv_writer.writerows(results)


if args.run_mode == 'infer':
    print('Infering')
    
    input = []
    if args.prompt:
        input.append(args.prompt)

    if args.file:
        with open(args.file,'r') as f:
            input.extend(f.readlines())
    
    model = load_model(args.base_model)
    tokenizer = load_tokenizer(args.base_model)
    
if args.measure_time:
    print('Time elapsed:', time.time()-start_time)