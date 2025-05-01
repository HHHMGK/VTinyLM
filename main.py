import argparse
import os, gc, time
import torch
import json, csv
# from train import train_with_hf_dataset
from model import load_model, load_tokenizer
from eval import eval_essay_perplexity, eval
from prune.prune import prune_model_generator, estimate_importance, serial_pruning_model_generator
from prune.data4prune import get_examples


# Take arg from command line
parser = argparse.ArgumentParser(description='')
parser.add_argument('run_mode', type=str, default='train', choices=['train','eval','infer','prune'], help='Mode run mode')
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
parser.add_argument('--load_peft_path', type=str, default=None, help='Path to load Peft model')

# For TRAINing mode
parser.add_argument('--model_path', type=str, default='', help='Path to model file')
parser.add_argument('--dataset_path', type=str, default='', help='Path to dataset file')
parser.add_argument('--block_size', type=int, default=1024, help='Size of text chunk')
parser.add_argument('--precision', type=str, default='fp16', choices=['fp16','fp32'], help='Precision mode')
parser.add_argument('--eval_after_train', action=argparse.BooleanOptionalAction, help='Evaluate after training or not')
parser.add_argument('--save_full_model', action=argparse.BooleanOptionalAction, help='Save full model or just LoRA adapter')
parser.add_argument('--save_path', type=str, default='./trained_model', help='Path to save model')

# For EVALuating mode
parser.add_argument('--benchmark', type=str, default='perplexity-essay-vn', help='Benchmark to evaluate')
parser.add_argument('--eval_repeat', type=int, default=1, help='Number of evaluation to repeat')
parser.add_argument('--eval_base', action=argparse.BooleanOptionalAction, help='Evaluate base model or not')

# For Pruing mode
parser.add_argument('--pruning_method', type=str, default='magnitude', choices=['magnitude','gradient','activation','combine','xconsecutive'], help='Pruning method')
parser.add_argument('--pruning_rate', type=float, default=[], nargs='*', help='Pruning rate(s)')
parser.add_argument('--pruning_layer_num', type=int, default=[], nargs='*', help='Number(s) of layers to prune')
parser.add_argument('--pruning_target', type=str, default='', help='Pruning target')
parser.add_argument('--pruning_data', type=str, default='c4', choices=['c4','bookcorpus','oscarvi'], help='Data for estimating importance')
parser.add_argument('--pruning_n_samples', type=int, default=1000, help='Number of samples for estimating importance')
parser.add_argument('--pruning_rand_data', action=argparse.BooleanOptionalAction, help='Random data for estimating importance')
parser.add_argument('--pruning_batch_size', type=int, default=32, help='Batch size for pruning')
parser.add_argument('--pruning_avg', action=argparse.BooleanOptionalAction, help='Average pruning or not')
parser.add_argument('--pruning_mag_norm', type=str, default='l1', choices=['l1','l2'], help='Norm for pruning')
parser.add_argument('--pruning_grad_T_order', type=int, default=1, help='T order for pruning')


def write_result(results, output_file, benchmark_type='perplexity', output_console=False):
    """
    Write the results to a CSV file.
    """
    if benchmark_type == 'perplexity':
        # header = ['Modification', 'Perplexity_mean', 'Perplexity_stddev', 'Time_mean', 'Time_stddev']
        header = ['Modification', 'Perplexity_mean', 'Time_mean']

    with open(output_file, 'a', newline='') as f:
        csv_writer = csv.DictWriter(f, fieldnames=header)
        csv_writer.writeheader()
        csv_writer.writerows(results)
    
    if output_console:
        print('Results written to', output_file)
        print(header)
        print(*results,sep='\n')


args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device using:', device)

if args.measure_time:
    start_time = time.time()

if args.run_mode == 'train':
    if args.load_peft_path is not None:
        print('Loading model', args.base_model, 'with Peft adapter:', args.load_peft_path)
        model = load_model(args.base_model, bnb=args.bnb, peft_path=args.load_peft_path, device=device)
    else:
        print('Loading base model:', args.base_model)
        model = load_model(args.base_model, bnb=args.bnb, device=device)
    tokenizer = load_tokenizer(args.base_model)
    print('Model and Tokenizer loaded')
    
    print('ReTraining model')
    train_with_hf_dataset(model, tokenizer, args.dataset_path, max_seq_length=args.block_size, precision=args.precision, technique='lora', device=device)

    if args.eval_after_train:
        print('Evaluating model')
        eval_results = eval_essay_perplexity(model, tokenizer, device, lang='vn', instructive=args.instructive_prompt,repeat=args.eval_repeat, measure_time=args.measure_time)
        print('Evaluation results:', eval_results)
    
    if args.save_full_model:
        model.save_pretrained(args.save_path, from_pt=True)

if args.run_mode == 'eval':
    print(f'Loading {"model " + args.base_model + " with Peft adapter: " + args.load_peft_path if args.load_peft_path else "base model: " + args.base_model}')
    base_model = load_model(args.base_model, bnb=args.bnb, peft_path=args.load_peft_path, device=device)
    tokenizer = load_tokenizer(args.base_model)
    print('Model and Tokenizer loaded')
     
    results = []
    benchmark_type = args.benchmark.split('-')[0] # 'perplexity' or other benchmark types
    
    if args.eval_base:
        print('Evaluating base model')
        eval_results = eval(base_model, tokenizer, args.benchmark, device, 
                          repeat=args.eval_repeat, measure_time=args.measure_time, 
                          instructive=args.instructive_prompt)
        results.append({'Modification':'Base model', **eval_results})
        
    # if args.modification == 'layer_reduction':
    #     new_model_generator = serial_pruning_model_generator(base_model, num_layers=None, step=args.layer_step)
    #     while True:
    #         model, layer_start, layer_end = next(new_model_generator, (None, None, None))
    #         if model is None:
    #             break
    #         print(f'Evaluating model with layers from {layer_start} to {layer_end} removed')
    #         eval_results = eval(model, tokenizer, args.benchmark, device, 
    #                           repeat=args.eval_repeat, measure_time=args.measure_time, 
    #                           instructive=args.instructive_prompt)
    #         results.append({'Modification':f'Removed {layer_start} to {layer_end}', **eval_results})
            
    #         del model
    #         gc.collect()
    #         torch.cuda.empty_cache()

    write_result(results, args.output, benchmark_type=benchmark_type, output_console=args.output_console)
    
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

if args.run_mode == 'prune':
    print(f'Loading {"model " + args.base_model + " with Peft adapter: " + args.load_peft_path if args.load_peft_path else "base model: " + args.base_model}')
    base_model = load_model(args.base_model, bnb=args.bnb, peft_path=args.load_peft_path, device=device)
    tokenizer = load_tokenizer(args.base_model)
    print('Model and Tokenizer loaded')
     
    results = []
    results.append({'Modification':args.__dict__})

    if args.eval_base:
        print('Evaluating base model')

        eval_results = eval(base_model, tokenizer, args.benchmark, device, repeat=args.eval_repeat, measure_time=args.measure_time, instructive=args.instructive_prompt)

        results.append({'Modification':'Base model', **eval_results})

    
    print('Fetching examples for pruning')
    if args.pruning_method in ['gradient','activation','combine']:
        prune_data = get_examples(dataset=args.pruning_data, tokenizer=tokenizer, rand=args.pruning_rand_data, n_samples=args.pruning_n_samples).to(device)
    else:
        prune_data = None

    print('Pruning model with method:', args.pruning_method)
    if args.pruning_method == 'xconsecutive':
        new_model_generator = serial_pruning_model_generator(base_model, num_layers=args.pruning_layer_num)
        while True:
            model, layers_pruned = next(new_model_generator, (None, None))
            if model is None:
                break
            layer_start = layers_pruned[0]
            layer_end = layers_pruned[-1]
            print(f'Evaluating model with layers from {layer_start} to {layer_end} removed')
            eval_results = eval(model, tokenizer, args.benchmark, device, 
                              repeat=args.eval_repeat, measure_time=args.measure_time, 
                              instructive=args.instructive_prompt)
            results.append({'Modification':f'Removed {layer_start} to {layer_end}', **eval_results})
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
    else:
        ranking = estimate_importance(base_model, method=args.pruning_method, prune_data=prune_data, avg=args.pruning_avg,norm=args.pruning_mag_norm, target=args.pruning_target, T_order=args.pruning_grad_T_order, batch_size=args.pruning_batch_size)
        print('Importance estimated with layers rankings:', ranking) 
        new_model_generator = prune_model_generator(base_model, ranking, pruning_rate=args.pruning_rate, pruning_layer_num=args.pruning_layer_num)   
        while True:
            model, layers_pruned = next(new_model_generator, (None, None))
            if model is None:
                break
            print(f'Evaluating model with layers {layers_pruned} pruned by {args.pruning_method} method')
            eval_results = eval(base_model, tokenizer, args.benchmark, device, repeat=args.eval_repeat, measure_time=args.measure_time, instructive=args.instructive_prompt)
            results.append({'Modification':f'Layers {str(layers_pruned)} pruned by {args.pruning_method} method', **eval_results})

            del model
            gc.collect()
            torch.cuda.empty_cache()
    
    write_result(results, args.output, benchmark_type=args.benchmark.split('-')[0], output_console=args.output_console)
    
    del base_model
    gc.collect()
    torch.cuda.empty_cache()