# model_finetuning.py

from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSeq2SeqLM  # model_1
from transformers import AutoModelForCausalLM   # model_1
from datasets import load_from_disk
from config import config
from callback import PlottingCallback
from clear_cache_callback import ClearCacheCallback
from checkpoint_utils import get_last_checkpoint  # 引入检查点工具模块
from plotting import TrainingPlotter
import os
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def create_experiment_dir(base_dir="./experiments"):
    # 获取当前时间作为实验名称
    experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def load_model(model_path=None):
    if model_path is None:
        model_path = config["model_path"]
    return AutoModelForSeq2SeqLM.from_pretrained(model_path).to('cuda')   # model_0
    # return AutoModelForCausalLM.from_pretrained(model_path).to('cuda')   # model_1

def compute_metrics(eval_pred):
    try:
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        # 处理特殊的 label 值，例如 -100
        labels = labels.reshape(-1)
        predictions = predictions.reshape(-1)
        valid_indices = labels != -100
        labels = labels[valid_indices]
        predictions = predictions[valid_indices]
        print("开始计算准确率")
        accuracy = accuracy_score(labels, predictions)
        print("完成计算准确率")
        return {'eval_accuracy': accuracy}
    except Exception as e:
        print(f"计算指标时发生异常：{e}")
        return {}

def finetune_model(plotter):
    # 加载预处理好的训练集和验证集
    tokenized_train_dataset = load_from_disk(config["data_path"])
    tokenized_eval_dataset = load_from_disk(config["eval_data_path"])

    print(f"训练集大小: {len(tokenized_train_dataset)}")
    print(f"验证集大小: {len(tokenized_eval_dataset)}")

    model = load_model()

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        eval_steps=1000,      # 更频繁地评估
        logging_steps=500,    # 更频繁地记录日志
        learning_rate=1e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=16,
        num_train_epochs=config["num_epochs"],
        weight_decay=0.01,
        logging_dir=os.path.join(config["output_dir"], "logs"),
        fp16=True,
        gradient_accumulation_steps=4,  # 根据需要调整
        logging_first_step=True,
        report_to=["none"],  # 禁用默认的日志记录器，防止重复日志
        eval_accumulation_steps=20,  # 验证过程的评估步数，较小的值可能会频繁进行累积和计算，导致整体速度变
        save_total_limit=2,  # 保存模型的最大数量，防止占用过多存储
        dataloader_num_workers=8,  # 设置 num_workers
        dataloader_pin_memory=True,  # 设置 pin_memory
        max_grad_norm=1.0,   # 添加梯度剪裁，防止梯度爆炸
    )

    # 获取最近的检查点（如果存在）
    last_checkpoint = get_last_checkpoint(config["output_dir"])

    # 创建 Trainer 实例，添加自定义回调
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[PlottingCallback(plotter), ClearCacheCallback()]
    )

    # 开始微调，如果有保存的检查点则从检查点继续训练
    try:
        if last_checkpoint:
            print(f"从检查点继续训练：{last_checkpoint}")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            trainer.train()
    except Exception as e:
        print(f"训练过程中发生异常：{e}")
        raise
        
    # 训练完成后保存模型
    trainer.save_model(config["finetuned_model_dir"])

if __name__ == "__main__":
    # 创建实验目录
    experiment_dir = create_experiment_dir()
    config["output_dir"] = experiment_dir

    # 初始化绘图实例，传入实验目录
    plotter = TrainingPlotter(experiment_dir=experiment_dir)

    # 进行微调
    finetune_model(plotter)
