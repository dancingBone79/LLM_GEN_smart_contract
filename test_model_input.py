# test_model.py

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# 指定模型路径和预训练模型路径
# model_path = "/root/localspace/Code_here/my_projects/c_gen/experiments/2024-10-27_13-47-59/checkpoint-4500"
model_path = "/root/localspace/Code_here/my_projects/c_gen/experiments/2024-11-25_12-38-33/checkpoint-21051"
original_model_path = "Salesforce/codet5-base"  # 替换为预训练的原始模型路径


# 加载模型和tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to('cuda')  # 若有 GPU，使用 .to('cuda')
tokenizer = AutoTokenizer.from_pretrained(original_model_path)

# natural language description
input_text = """Please write a complete Solidity smart contract for a house rental. 
The contract should include the following features:
- Allow landlords to list properties with details (location, price, description).
- Tenants can rent properties and make payments.
- Both landlords and tenants can review the rental agreements.
- A security deposit is required and refunded after the rental period."""
inputs = tokenizer(input_text, return_tensors="pt").to('cuda')

# generate smart contract code
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=512)
    generated_code = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True)

print("Generated Smart Contract Code:\n", generated_code)


