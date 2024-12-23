# data_loading.py

from datasets import load_dataset

def load_data(dataset_name="andstor/smart_contracts", config_name="raw", split="train"):
    print("开始加载数据集...")
    dataset = load_dataset(dataset_name, config_name, split=split, trust_remote_code=True)
    print("数据集加载完成")
    return dataset

def split_dataset(dataset, test_size=0.1, validation_size=0.1):
    # 首先将训练集划分为训练集和测试集
    train_test_split = dataset.train_test_split(test_size=test_size)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    # 再将训练集的一部分划分为验证集
    train_validation_split = train_dataset.train_test_split(test_size=validation_size)
    train_dataset = train_validation_split['train']
    validation_dataset = train_validation_split['test']

    return train_dataset, validation_dataset, test_dataset
