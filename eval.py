from pathlib import Path
import copy, json
import torch
import numpy as np
from metrics import Perplexity
from prune.data4prune import get_examples

ESSAY_BENCHMARK_PATH = str(Path('./datasets/benchmarks/perplexity/essay.json').absolute())
NEWS_BENCHMARK_PATH = str(Path('./datasets/benchmarks/perplexity/news.json').absolute())

def eval_perplexity(model, tokenizer, prompts, device, repeat=1, measure_time=False):
    """
    Evaluate the perplexity of a model on a given benchmark.
    Args:

        model: The model to evaluate.
        tokenizer: The tokenizer to use.
        prompts: The prompts to evaluate on.
        device: The device to run the model on.
        repeat: Number of times to repeat the evaluation.
        measure_time: Whether to measure the time taken for each evaluation.
    """
    perplexity = Perplexity()
        
    # model = model.to(device)
    # model.eval()

        # --- Warm-up Run ---
    if repeat > 1 or measure_time: # Only warm-up if timing or multiple runs matter
        print("Performing GPU warm-up run...")
        with torch.no_grad():
            # Use a small subset or the first prompt for a quick warm-up
            _ = perplexity._compute(prompts, model, tokenizer, device,measure_time=measure_time)
        torch.cuda.synchronize() # Ensure warm-up is complete before proceeding
        print("Warm-up complete.")
    # --- End Warm-up ---

    with torch.no_grad():
        perplexity_score = []
        if measure_time:
            timing = []
            for i in range(repeat):
                ppl = perplexity._compute(prompts, model, tokenizer, device, measure_time=measure_time)
                perplexity_score.append(ppl['mean_perplexity'])
                timing.append(ppl['runtime'])
        else:
            for i in range(repeat):
                perplexity_score.append(perplexity._compute(prompts, model, tokenizer, device, measure_time=measure_time)['mean_perplexity'])

    perplexity_mean = perplexity_score[0]
    # perplexity_stddev = 0
    time_mean = 0
    # time_stddev = 0
    if measure_time:
        time_mean = timing[0]
        # time_stddev = 0
        
    if repeat > 1:
        perplexity_mean = np.mean(perplexity_score)
        # perplexity_stddev = np.std(perplexity_score)

        if measure_time:
            time_mean = np.mean(timing)
            # time_stddev = np.std(timing)
    # print(f'Execution time: {time_mean:.2f} seconds with stddev {time_stddev:.2f} seconds')
    # return {'perplexity': (perplexity_mean, perplexity_stddev), 'time': (time_mean, time_stddev) if measure_time else None}
    return {'Perplexity_mean':perplexity_mean, # 'Perplexity_stddev':perplexity_stddev, 
                            'Time_mean':time_mean} #'Time_stddev':time_stddev}

def eval_essay_perplexity(model, tokenizer, device, lang='vn', instructive=False ,repeat=1, measure_time=False):
    """
    Evaluate the perplexity of a model on a given language benchmark.
    Args:
        lang: The language benchmark to evaluate on, 'vn' or 'en'.
        instructive: Whether to add an instructive prompt before the benchmark.
    """
    perplexity = Perplexity()

    with open(ESSAY_BENCHMARK_PATH,'r',encoding='utf-8') as f:
        ESSAY_BENCHMARKS = json.load(f)
    prompts = ESSAY_BENCHMARKS["data"][lang]
    if instructive:
        if lang=='vn':
            additional_prompt = 'Viết một đoạn văn nghị luận về vấn đề sau: '
        elif lang=='en':
            additional_prompt = 'Write an argumentative essay about the following topic: '
        prompts = [additional_prompt + prompt for prompt in prompts]
    
    return eval_perplexity(model, tokenizer, prompts, device, repeat, measure_time)

def eval_news_perplexity(model, tokenizer, device, lang='vn', tag="Thị trường", instructive=False, repeat=1, measure_time=False):
    """
    Evaluate the perplexity of a model on a given language benchmark.
    Args:
        lang: The language benchmark to evaluate on, 'vn' or 'en'.
        instructive: Whether to add an instructive prompt before the benchmark.
    """
    perplexity = Perplexity()
    with open(NEWS_BENCHMARK_PATH,'r',encoding='utf-8') as f:
        NEWS_BENCHMARKS = json.load(f)
    prompts = NEWS_BENCHMARKS["data"][lang][tag]
    if instructive:
        if lang=='vn':
            prompt_format = 'Hoàn thiện bài báo về {title} thuộc thể loại {tag}'
        # elif lang=='en':
        #     additional_prompt = 'Write an argumentative essay about the following topic: '
        prompts = [prompt_format.format(title=prompt, tag=tag) for prompt in prompts]
    
    return eval_perplexity(model, tokenizer, prompts, device, repeat, measure_time)

def eval_dataset_perplexity(model, tokenizer, dataset, device, repeat=1, measure_time=False):
    """
    Evaluate the perplexity of a model on a given dataset.
    Args:
        dataset: The dataset to evaluate on.
    """
    perplexity = Perplexity()
    data = get_examples(dataset, tokenizer, n_samples=64, seq_len=64, rand=False, raw=True)
    
    return eval_perplexity(model, tokenizer, data, device, repeat, measure_time)

def eval(model, tokenizer, benchmark, device, repeat=1, measure_time=False, instructive=False):
    benchmark_type = benchmark.split('-')[0] # 'perplexity' or 'villm'
    if benchmark_type == 'perplexity':
        type = benchmark.split('-')[1] # 'essay' or 'news' or 'dataset'
        if type == 'essay':
            lang = benchmark.split('-')[2] # 'vn' or 'en'
            return eval_essay_perplexity(model, tokenizer, device, lang=lang, instructive=instructive, repeat=repeat, measure_time=measure_time)
        elif type == 'news':
            lang = benchmark.split('-')[2] # 'vn' or 'en'
            return eval_news_perplexity(model, tokenizer, device, lang=lang, instructive=instructive, repeat=repeat, measure_time=measure_time)
        elif type == 'dataset':
            dataset = benchmark.split('-')[2]
            return eval_dataset_perplexity(model, tokenizer, dataset, device, repeat, measure_time)
        else:
            raise ValueError(f"Unknown benchmark type, choose from 'essay', 'news', or 'dataset'")
