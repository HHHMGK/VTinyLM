import copy, json
import torch
from metrics import Perplexity

with open('benchmarks.json','r') as f:
    BENCHMARKS = json.load(f)

def infer(model, tokenizer, instruction):
    PROMPT_TEMPLATE = f"### Câu hỏi: {instruction}\n### Trả lời:"  

    input_prompt = PROMPT_TEMPLATE.format_map({"instruction": instruction})  

    input_ids = tokenizer(input_prompt, return_tensors="pt")  

    outputs = model.generate(  
        inputs=input_ids["input_ids"].to("cuda"),  
        attention_mask=input_ids["attention_mask"].to("cuda"),  
        do_sample=True,  
        temperature=1.0,  
        top_k=50,  
        top_p=0.9,  
        max_new_tokens=1024,  
        eos_token_id=tokenizer.eos_token_id,  
        pad_token_id=tokenizer.pad_token_id  
    )  

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]  
    response = response.split("### Trả lời:")[1]
    return response

# def evaluate(model, tokenizer, benchmark, device):
#     model = model.to('cuda')
#     model.eval()



#     model.to('cpu')
#     torch.cuda.empty_cache()
    
def eval_perplexity(model, tokenizer, device, lang='vn'):
    perplexity = Perplexity()

    prompts = BENCHMARKS["perplexity"]["data"][lang]

    model = model.to('cuda')
    model.eval()

    perplexity_score = perplexity._compute(prompts, model, tokenizer, device)

    model.to('cpu')
    torch.cuda.empty_cache()
    return perplexity_score['mean_perplexity']
        
    