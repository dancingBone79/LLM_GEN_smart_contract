# config.py

config = {
    "batch_size": 32,
    "batch_size_train": 16,
    "batch_size_eval": 4,
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "model_path": "Salesforce/codet5-base",  # 使用的预训练模型路径
    "model_path_1": "Salesforce/codegen-6B-mono",  # 使用的预训练模型路径
    "model_path_2": "Salesforce/codet5-base",  # 使用的预训练模型路径
    "model_path_3": "Salesforce/codet5-base",  # 使用的预训练模型路径
    "data_path": "tokenized_data/train",  # 训练集预处理后的保存路径
    "eval_data_path": "tokenized_data/validation",  # 验证集预处理后的保存路径
    "test_data_path": "tokenized_data/test",  # 测试集预处理后的保存路径
    "output_dir": "results",                     # 训练输出路径
    "finetuned_model_dir": "finetuned_model",    # 微调后的模型保存路径
    "contract_output_file": "generated_contract.sol"  # 智能合约输出文件名
}
