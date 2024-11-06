import os
import json
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
# from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
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
    
# The following function is from HF's example
def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def process_hf_dataset(dataset, tokenizer, format_fn, max_seq_length):
    dataset = dataset.map(format_fn, batched=True)
    tokenized_dataset = dataset.map(lambda e: tokenizer([" ".join(x) for x in e['text']]), 
                                    batched=True, remove_columns=dataset.column_names)
    grouped_text_dataset = tokenized_dataset.map(lambda e: group_texts(e, max_seq_length), batched=True)
    return grouped_text_dataset

INSTRUCTION_TEMPLATE = "### Câu hỏi:"
RESPONSE_TEMPLATE = "\n### Trả lời:"

def format_fn(sample):
    return {
        'text': [
            f"""{INSTRUCTION_TEMPLATE}
                Hoàn thiện bài báo về {title} thuộc thể loại {category}
                {RESPONSE_TEMPLATE}
                {content}
            """ for title, category, content in zip(sample['title'], sample['category'], sample['content'])
        ]
    }

def format_instruction(sample):
    return  f"""{INSTRUCTION_TEMPLATE}
                Hoàn thiện bài báo về {sample['title']} thuộc thể loại {sample['category']}
                {RESPONSE_TEMPLATE}
                {sample['content']}
                """

def train_with_hf_dataset(model, tokenizer, file_path, precision, max_seq_length, technique = 'lora', device = 'cuda'):
    if file_path is not None:
        file_path = str(Path(file_path).absolute())
    dataset = process_hf_dataset(get_hf_dataset(file_path), tokenizer, format_fn, max_seq_length)
    dataset = dataset.train_test_split(test_size=0.1)
    # dataset = get_hf_dataset(file_path)
    # datacollator = DataCollatorForCompletionOnlyLM(
    #     instruction_template=INSTRUCTION_TEMPLATE,
    #     response_template=RESPONSE_TEMPLATE,
    #     tokenizer=tokenizer
    # )  
    datacollator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    if technique == 'lora':
        print(dataset)
        # model.resize_token_embeddings(len(tokenizer))

        peft_cfg = LoraConfig(
            # lora_alpha=16,
            # lora_dropout=0.1,
            # r=4,
            # # target_modules=["all_linear"]
            # bias="none",
            task_type="CAUSAL_LM",
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
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

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=datacollator,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            tokenizer=tokenizer,
        )

        trainer.train()
        trainer.save_model()

