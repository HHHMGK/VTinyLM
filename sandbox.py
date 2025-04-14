import train, os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

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


INSTRUCTION_TEMPLATE = "### Câu hỏi:"
RESPONSE_TEMPLATE = "\n### Trả lời:"

def format_instruction(sample):
    # return  [f"""### Câu hỏi: 
    #             Hoàn thiện bài báo về {sample['title']} thuộc thể loại {sample['category']}\n
    #             ### Trả lời:
    #             {sample['content']}
    #             """]
    # print(sample)
    formatted = []
    for i in range(len(sample)):
        formatted.append(f"""### Câu hỏi: 
                            Hoàn thiện bài báo về {sample[i]['title']} thuộc thể loại {sample[i]['category']}
                            \n### Trả lời:
                            {sample[i]['content']}
                            """)
    return formatted    

tokenizer = AutoTokenizer.from_pretrained("C:\\Users\\HUY\\Downloads\\phoGPT-4B-Chat", use_remote_code=True) 

datacollator = DataCollatorForCompletionOnlyLM(
    instruction_template=INSTRUCTION_TEMPLATE,
    response_template=RESPONSE_TEMPLATE,
    tokenizer=tokenizer
)  

dataset = get_hf_dataset('datasets\\vneconomy\\vneconomy.json')


sample_batch = [dataset[i] for i in range(3)]
sample_batch = tokenizer(sample_batch)
print
print(sample_batch)