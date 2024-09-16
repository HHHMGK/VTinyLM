

def infer_original(model, tokenizer, instruction, device):
    PROMPT_TEMPLATE = f"### Câu hỏi: {instruction}\n### Trả lời:"  

    input_prompt = PROMPT_TEMPLATE.format_map({"instruction": instruction})  

    input_ids = tokenizer(input_prompt, return_tensors="pt")  

    outputs = model.generate(  
        inputs=input_ids["input_ids"].to(device),  
        attention_mask=input_ids["attention_mask"].to(device),  
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


def infer_x(model, tokenizer, input, device):
    inputs = tokenizer(input, return_tensors="pt")
    output = model.generate( 
        inputs=input["input_ids"].to(device),  
        attention_mask=input["attention_mask"].to(device),  
        do_sample=True,  
        temperature=1.0,  
        top_k=50,  
        top_p=0.9,  
        max_new_tokens=1024,  
        eos_token_id=tokenizer.eos_token_id,  
        pad_token_id=tokenizer.pad_token_id  
    )  

    response = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return response