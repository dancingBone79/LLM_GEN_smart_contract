# generate_contract.py


from transformers import AutoTokenizer
from model_management import load_model
from config import config

def generate_contract():
    # 加载微调后的模型
    model = load_model(config["finetuned_model_dir"])

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])

    # 输入提示
    input_text = """
    Generate a Solidity smart contract that represents a simple token with the following details:
    - Name: MyToken
    - Symbol: MTK
    - Initial supply: 1000000 tokens
    The contract should include functions for transferring tokens, checking balance, and owner controls.
    """

    # 编码输入
    inputs = tokenizer(input_text, return_tensors="pt").to('cuda')

    # 生成智能合约代码
    outputs = model.generate(
        **inputs,
        max_length=512,
        num_beams=5,
        early_stopping=True
    )

    # 解码生成的合约
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 保存合约
    with open(config["contract_output_file"], "w") as f:
        f.write(generated_text)

    print(f"Generated Smart Contract saved to {config['contract_output_file']}.")

if __name__ == "__main__":
    generate_contract()
