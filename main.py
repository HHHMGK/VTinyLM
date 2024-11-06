import argparse
import os, gc, time
import torch
import json, csv
from train import train_with_hf_dataset
from model import load_model, load_tokenizer, layer_reduction_model_generator, layer_removal
from eval import eval_perplexity

# Take arg from command line
parser = argparse.ArgumentParser(description='')
parser.add_argument('run_mode', type=str, default='train', choices=['train','eval','infer'], help='Mode run mode')
parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
# parser.add_argument('--verbose', type=int, default=0, choices=[0,1,2], help='Verbose mode, 0 = silent, 1 = print log, 2 = print all')

# Common arguments
parser.add_argument('--base_model', type=str, default='', help='Base model name')
parser.add_argument('--output', type=str, default='results.csv', help='Output file for results')
parser.add_argument('--instructive_prompt', type=str, default=False, help='Adding instructive prompt when evaluating')
parser.add_argument('--measure_time', action=argparse.BooleanOptionalAction, help='Measure run time or not')
parser.add_argument('--output_console', action=argparse.BooleanOptionalAction, help='Print output to console or not')
# parser.add_argument('--run_dummy', type=bool, default=False, help='Run dummy mode (system testing) or not')
parser.add_argument('--bnb', type=str, default='none', choices=['none', '4bit', '8bit'], help='Load model with Bits and Bytes (4bit or 8bit) or not')

# For TRAINing mode
parser.add_argument('--model_path', type=str, default='', help='Path to model file')
parser.add_argument('--pruning', action=argparse.BooleanOptionalAction, help='Pruning model or not')
parser.add_argument('--pruning_layer_start', type=int, default=0, help='Pruning start layer')
parser.add_argument('--pruning_layer_end', type=int, default=0, help='Pruning end layer')
parser.add_argument('--dataset_path', type=str, default='', help='Path to dataset file')
parser.add_argument('--block_size', type=int, default=1024, help='Size of text chunk')
parser.add_argument('--precision', type=str, default='fp16', choices=['fp16','fp32'], help='Precision mode')
parser.add_argument('--eval_after_train', action=argparse.BooleanOptionalAction, help='Evaluate after training or not')
parser.add_argument('--save_path', type=str, default='./trained_model', help='Path to save model')

# For EVALuating mode
parser.add_argument('--benchmark', type=str, default='perplexity-vn', choices=['perplexity-vn','perplexity-en','villm-eval'], help='Benchmark to evaluate')
parser.add_argument('--repeat', type=int, default=1, help='Number of evaluation to repeat')
parser.add_argument('--modification', type=str, default='layer_reduction', choices=['layer_reduction','base'], help='Model modification method')
parser.add_argument('--eval_base', action=argparse.BooleanOptionalAction, help='Evaluate base model or not')
parser.add_argument('--layer_step', type=int, default=0, help='Step for layer modification')

# For INFERing mode
parser.add_argument('--prompt', type=str, default='', help='Input prompt for inference')
parser.add_argument('--file', type=str, default='', help='Input file for inference')
# Import config from config.json

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device using:', device)

if args.measure_time:
    start_time = time.time()

if args.run_mode == 'train':
    # print('Config path:', args.config)
    print('Loading as base model:', args.base_model)
    model = load_model(args.base_model, bnb=args.bnb)
    tokenizer = load_tokenizer(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    print('Model and Tokenizer loaded')
    if args.pruning:
        print('Pruning model')
        model = layer_removal(model, args.pruning_layer_start, args.pruning_layer_end)
        print('Model pruned')
    print('Training model')
    train_with_hf_dataset(model, tokenizer, args.dataset_path, max_seq_length=args.block_size, precision=args.precision, technique='lora', device=device)

    if args.eval_after_train:
        print('Evaluating model')
        eval_results = eval_perplexity(model, tokenizer, device, lang='vn', instructive=args.instructive_prompt,repeat=args.repeat, measure_time=args.measure_time)
        print('Perplexity:', eval_results['perplexity'])
        print('Time:', eval_results['time'])
    
    model.save_pretrained(args.save_path, from_pt=True)

if args.run_mode == 'eval':
    print('Evaluating with benchmark:', args.benchmark)
    
    print('Loading base model:', args.base_model)
    base_model = load_model(args.base_model, bnb=args.bnb)
    tokenizer = load_tokenizer(args.base_model)
    print('Model loaded')
    results = []
    if args.benchmark.startswith('perplexity'):
        lang = args.benchmark.split('-')[1] # 'vn' or 'en'

        if args.eval_base:
            print('Evaluating base model')
            eval_results = eval_perplexity(base_model, tokenizer, device, lang=lang, instructive=args.instructive_prompt,repeat=args.repeat, measure_time=args.measure_time)
            print('Perplexity:', eval_results['perplexity'])
            # _perplexity = f'{round(eval_results["perplexity"][0],3)} 0xC2 {round(round(eval_results["perplexity"][1],3))}'
            # _time = f'{round(eval_results["time"][0],3)} 0xC2 {round(round(eval_results["time"][1],3))}' if args.measure_time else None
            results.append({'Modification':'Base model', 
                            'Perplexity_mean':eval_results['perplexity'][0], 'Perplexity_stddev':eval_results['perplexity'][1], 
                            'Time_mean':eval_results['time'][0], 'Time_stddev':eval_results['time'][1]})
            
        if args.modification == 'layer_reduction':
            new_model_generator = layer_reduction_model_generator(base_model, num_layers = None, step=args.layer_step)
            while True:
                model, layer_start, layer_end = next(new_model_generator, (None, None, None))
                if model is None:
                    break
                print(f'Evaluating model with layers from {layer_start} to {layer_end} removed')
                eval_results = eval_perplexity(model, tokenizer, device, lang=lang, instructive=args.instructive_prompt,repeat=args.repeat, measure_time=args.measure_time)
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

    if args.output_console:
        print('Results:')
        with open(args.output,'r') as f:
            print(f.read())
        

if args.run_mode == 'infer':
    print('Infering')
    
    input = []
    if args.prompt:
        input.append(args.prompt)

    if args.file:
        with open(args.file,'r') as     f:
            input.extend(f.readlines())
    
    model = load_model(args.base_model, bnb=args.bnb)
    tokenizer = load_tokenizer(args.base_model)
    
if args.measure_time:
    print('Time elapsed:', time.time()-start_time)