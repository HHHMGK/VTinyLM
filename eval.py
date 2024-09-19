import copy, json, time
import torch
import numpy as np
from metrics import Perplexity

with open('benchmarks.json','r') as f:
    BENCHMARKS = json.load(f)


# def evaluate(model, tokenizer, benchmark, device):
#     model = model.to('cuda')
#     model.eval()



#     model.to('cpu')
#     torch.cuda.empty_cache()
    
def eval_perplexity(model, tokenizer, device, lang='vn', repeat=1, measure_time=False):
    perplexity = Perplexity()

    prompts = BENCHMARKS["perplexity"]["data"][lang]

    model = model.to('cuda')
    model.eval()
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
    model.to('cpu')
    torch.cuda.empty_cache()

    perplexity_mean = perplexity_score[0]
    perplexity_stddev = 0
    time_mean = timing[0]
    time_stddev = 0
    if repeat > 1:
        perplexity_mean = np.mean(perplexity_score)
        perplexity_stddev = np.std(perplexity_score)

        if measure_time:
            time_mean = np.mean(timing)
            time_stddev = np.std(timing)

    
    return {'perplexity': (perplexity_mean, perplexity_stddev), 'time': (time_mean, time_stddev) if measure_time else None}
