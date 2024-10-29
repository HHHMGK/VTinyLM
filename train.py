import os
import json
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import pandas as pd

# DATASETS_INFO_PATH = Path(__file__).parent / 'datasets' / 'datasets_info.json'
DATASETS_INFO_PATH = str(Path('./datasets/datasets_info.json').absolute())
with open(DATASETS_INFO_PATH, 'r') as f:
    DATASETS_INFO = json.load(f)

def dataset_columns_mapping(dataset, dataset_name = 'vneconomy'):
    columns_mapping = DATASETS_INFO[dataset_name]
    for key in columns_mapping.keys():
        dataset = dataset.rename_column(key,columns_mapping[key])
    return dataset

def get_hf_dataset(file_path = None):
    if file_path is None or not os.path.exists(file_path):
        print('File',file_path,'not found. Loading dummy instead.')
        df = pd.DataFrame({'text': ['This is a dummy text']})
    else:
        print('Loading dataset from',file_path)
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path, encoding='utf-8',orient='records')
    
    return Dataset.from_pandas(df)
    
# def process_hf_dataset(dataset, tokenizer):
#     dataset = dataset_columns_mapping(dataset)
#     dataset = dataset.select_columns('text')
#     dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length'),
#                             batched=True)
#     # dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])    
#     return dataset

def format_instruction(sample):
    # return  f"""### Câu hỏi: 
    #             Hoàn thiện bài báo về {sample['title']} thuộc thể loại {sample['category']}\n
    #             ### Trả lời:
    #             {sample['content']}
    #             """
    formatted = []
    for i in range(len(sample['content'])):
        formatted.append(f"""### Câu hỏi: 
                            Hoàn thiện bài báo về {sample['title'][i]} thuộc thể loại {sample['category'][i]}
                            \n### Trả lời:
                            {sample['content'][i]}
                            """)
    return formatted    

RESPON_TEMPLATE = "\n### Trả lời:"

def train_with_hf_dataset(model, tokenizer, file_path, device, precision ='fp16', max_seq_length =2048, technique = 'lora'):
    if file_path is not None:
        file_path = str(Path(file_path).absolute())
    # dataset = process_hf_dataset(get_hf_dataset(file_path), tokenizer)
    dataset = get_hf_dataset(file_path)
    datacollator = DataCollatorForCompletionOnlyLM(
        response_template=RESPON_TEMPLATE, 
        tokenizer=tokenizer
    )  
    if technique == 'lora':
        # dataset = dataset.train_test_split(test_size=0.1)
        print(dataset)
        model.resize_token_embeddings(len(tokenizer))

        peft_cfg = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_cfg)
        
        training_args = TrainingArguments(
            output_dir='./train_results',
            save_strategy='epoch',
            num_train_epochs=2,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            # warmup_steps=500,
            # weight_decay=0.01,
            logging_dir='./train_logs',
            logging_steps=10,
            adam_beta1=0.9,
            adam_beta2=0.95,
            learning_rate=2e-4,
            lr_scheduler_type='cosine',
            report_to='none',
            fp16=(precision=='fp16'),
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            peft_config=peft_cfg,
            train_dataset=dataset,
            data_collator=datacollator,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            formatting_func=format_instruction,
        )
        trainer.train()
        trainer.save_model()

