from transformers import AutoModelForCausalLM, AutoTokenizer

teacher_model_name = "vinai/PhoGPT-4B-Chat"
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, trust_remote_code=True)



