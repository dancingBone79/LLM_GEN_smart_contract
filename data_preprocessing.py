# data_preprocessing.py

from transformers import AutoTokenizer
from config import config
from data_loading import load_data, split_dataset

def preprocess_data(dataset, save_path):
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])

    def preprocess_function(examples):
        inputs = examples["contract_name"]  # 使用合约名称作为输入
        outputs = examples["source_code"]  # 使用合约的源代码作为输出
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(outputs, max_length=512, truncation=True, padding="max_length").input_ids
        model_inputs["labels"] = labels
        return model_inputs

    # 预处理并保存数据
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.save_to_disk(save_path)
    print(f"Data preprocessing complete and saved to {save_path}.")

if __name__ == "__main__":
    # 加载训练集
    train_dataset = load_data(split="train")

    # 手动拆分训练集、验证集和测试集
    train_dataset, validation_dataset, test_dataset = split_dataset(train_dataset)

    # 预处理训练集
    preprocess_data(train_dataset, config["data_path"])

    # 预处理验证集
    preprocess_data(validation_dataset, config["eval_data_path"])

    # 预处理测试集
    preprocess_data(test_dataset, config["test_data_path"])
