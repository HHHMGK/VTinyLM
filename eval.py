import copy, json
import torch
from metrics import Perplexity

with open('benchmarks.json','r') as f:
    BENCHMARKS = json.load(f)


# def evaluate(model, tokenizer, benchmark, device):
#     model = model.to('cuda')
#     model.eval()



#     model.to('cpu')
#     torch.cuda.empty_cache()
    
def eval_perplexity(model, tokenizer, device, lang='vn', repeat=1):
    perplexity = Perplexity()

    prompts = BENCHMARKS["perplexity"]["data"][lang]

    model = model.to('cuda')
    model.eval()
    perplexity_score = []
    for i in range(repeat):
        perplexity_score.append(perplexity._compute(prompts, model, tokenizer, device)['mean_perplexity'])

    model.to('cpu')
    torch.cuda.empty_cache()
    return perplexity_score
