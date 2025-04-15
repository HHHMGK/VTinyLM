from pathlib import Path
import copy, json, time
import torch
import numpy as np
from metrics import Perplexity

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
    with torch.no_grad():
        perplexity_score = []
        if measure_time:
            timing = []
            for i in range(repeat):
                start = time.time()
                perplexity_score.append(perplexity._compute(prompts, model, tokenizer, device)['mean_perplexity'])
                timing.append(time.time()-start)
        else:
            for i in range(repeat):
                perplexity_score.append(perplexity._compute(prompts, model, tokenizer, device)['mean_perplexity'])

    perplexity_mean = perplexity_score[0]
    perplexity_stddev = 0
    time_mean, time_stddev = 0, 0
    if measure_time:
        time_mean = timing[0]
        time_stddev = 0
        
    if repeat > 1:
        perplexity_mean = np.mean(perplexity_score)
        perplexity_stddev = np.std(perplexity_score)

        if measure_time:
            time_mean = np.mean(timing)
            time_stddev = np.std(timing)

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

