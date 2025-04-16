from datasets import load_dataset, Dataset
import torch
import random

def get_examples(dataset, tokenizer, n_samples, seq_len = 64, rand=False, raw=False):
    if dataset == 'c4':
        traindata = load_dataset(
            'allenai/c4', data_files='en/c4-train.00000-of-01024.json.gz', split='train'
        )
    elif dataset == 'bookcorpus':
        traindata = load_dataset(
            'bookcorpus', split='train', streaming=True
        )
        data_list=[]
        for d in traindata:
            data_list.append(d)
            if len(data_list) == n_samples*5:
                break
        # data_list = list(traindata["train"].take(n_samples*10))
        traindata = Dataset.from_list(data_list)
    elif dataset == 'oscar-vi':
        # 'https://huggingface.co/datasets/oscar-corpus/OSCAR-2301/blob/main/vi_meta/vi_meta_part_1.jsonl.zst'
        traindata = load_dataset(
            'oscar-corpus/OSCAR-2301', language='vi', split='train', streaming=True, trust_remote_code=True
        )
        data_list=[]
        for d in traindata:
            data_list.append(d)
            if len(data_list) == n_samples*5:
                break
        # data_list = list(traindata["train"].take(n_samples*10))
        traindata = Dataset.from_list(data_list)

    samples, history = [], []
    if rand:
        for nn in range(n_samples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                if i in history:
                    continue
                if raw:
                    sample = traindata[i]['text']
                    if len(sample) >= seq_len:
                        history.append(i)
                        break
                else:
                    sample = tokenizer(traindata[i]['text'], return_tensors='pt', truncation=True)
                    if sample.input_ids.shape[1] >= seq_len:
                        history.append(i)
                        break
            if raw:
                i = random.randint(0, len(sample) - seq_len)
                samples.append(sample[i:i+seq_len])
            else:
                i = random.randint(0, sample.input_ids.shape[1] - seq_len)
                samples.append(sample.input_ids[:, i:i+seq_len])
    else:
        i = 0
        for _ in range(n_samples):
            while True:
                if raw:
                    sample = traindata[i]['text']
                    if len(sample) >= seq_len:
                        samples.append(sample[:seq_len])
                        break
                else:
                    sample = tokenizer(traindata[i]['text'], return_tensors='pt')
                    if sample.input_ids.shape[1] >= seq_len:
                        samples.append(sample.input_ids[:, :seq_len])
                        break
                i += 1
    if raw:
        return samples
    else: 
        return torch.cat(samples, dim=0 )