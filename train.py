import os
import json
from pathlib import Path
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# DATASETS_INFO_PATH = Path(__file__).parent / 'datasets' / 'datasets_info.json'
DATASETS_INFO_PATH = Path('./datasets/datasets_info.json').absolute()
with DATASETS_INFO_PATH.open('r') as f:
    DATASETS_INFO = json.load(f)

def dataset_columns_mapping(dataset, dataset_name = 'vneconomy'):
    columns_mapping = DATASETS_INFO[dataset_name]
    for key in columns_mapping.key():
        dataset = dataset.rename_column(key,columns_mapping[key])
    return dataset

def get_hf_dataset(file_path = None, file_type = 'json'):
    if file_path is None or not os.path.exists(file_path):
        print('File',file_path,'not found. Loading dummy instead.')
        return load_dataset('csv', data_files='dummy.csv')
    else:
        print('Loading dataset from',file_path)
        return load_dataset(file_type, data_files=file_path)
    
def process_hf_dataset(dataset, tokenizer):
    dataset = dataset_columns_mapping(dataset)
    dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length'),
                            batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])    
    return dataset

def train_with_hf_dataset(model, tokenizer, file_path, file_type, device, technique = 'full'):
    if file_path is not None:
        file_path = Path(file_path).absolute()
    dataset = process_hf_dataset(get_hf_dataset(file_path, file_type), tokenizer)
    datacollator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)  
    if technique == 'full':
        model.resize_token_embeddings(len(tokenizer))
        # dataset = dataset.train_test_split(test_size=0.2)
        
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
            lr_scheduler_type='cosine'
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=datacollator
        )

        trainer.train()

