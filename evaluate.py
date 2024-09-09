# from model import load_model, load_tokenizer, clone_model
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import copy
import torch
import evaluate
import datasets

benchmarks = [
    "perplexity",
    "villm-eval"
]

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

def evaluate(model, tokenizer, benchmark):
    model = model.to('cuda')
    model.eval()

    model.to('cpu')
    torch.cuda.empty_cache()
    
def eval_perplexity(model, tokenizer):
    model = model.to('cuda')
    model.eval()
    
    perplexity = 0
    for i in range(10):
        with torch.no_grad():
            input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt").input_ids
            labels = input_ids.clone()
            labels[0, 1:] = -100

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            perplexity += torch.exp(loss)
    perplexity /= 10
    return perplexity

class Perplexity(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            module_type="metric",
            description="Perplexity - how well the model predicts the sample, measures a model's ability to predict the next word in a sequence",
            citation="",
            inputs_description="texts",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                }
            ),
            reference_urls=["https://huggingface.co/docs/transformers/perplexity"],
        )

    def _compute(self, 
                 predictions, model, tokenizer,
                 batch_size: int = 16, add_start_token: bool = True, device="cuda", max_length=1024):
        
        model = model.to(device)
        model.eval()

        perplexity = 0
