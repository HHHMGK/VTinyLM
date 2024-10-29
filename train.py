import os
import json
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
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
    
def process_hf_dataset(dataset, tokenizer):
    dataset = dataset_columns_mapping(dataset)
    dataset = dataset.select_columns('text')
    dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length'),
                            batched=True)
    # dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])    
    return dataset

def train_with_hf_dataset(model, tokenizer, file_path, device, precision ='fp16', technique = 'full'):
    if file_path is not None:
        file_path = str(Path(file_path).absolute())
    dataset = process_hf_dataset(get_hf_dataset(file_path), tokenizer)
    datacollator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)  
    if technique == 'full':
        dataset = dataset.train_test_split(test_size=0.1)
        print(dataset)
        model.resize_token_embeddings(len(tokenizer))
        
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

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=datacollator
        )

        trainer.train()

