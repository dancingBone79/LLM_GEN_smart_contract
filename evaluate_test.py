from transformers import Trainer, TrainingArguments
from clear_cache_callback import ClearCacheCallback
from datasets import load_from_disk
from config import config
from transformers import AutoModelForSeq2SeqLM
from plotting import TrainingPlotter
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import threading
from nltk.translate.bleu_score import sentence_bleu
import coverage
import math
from checkpoint_utils import get_last_checkpoint
from model_selector import ModelSelector


def load_model(model_path=None):
    """
    加载预训练或微调后的模型。

    Args:
        model_path (str): 模型的路径。如果为 None,则使用默认配置中的模型路径。

    Returns:
        model: 加载的模型。
    """
    if model_path is None:
        model_path = config["model_path"]
    return AutoModelForSeq2SeqLM.from_pretrained(model_path).to('cuda')


def compute_metrics(eval_pred):
    """
    计算评估指标。

    Args:
        eval_pred: 包含模型预测和真实标签的元组。

    Returns:
        dict: 包含评估指标的字典。
    """
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    # 获取预测结果
    predictions = np.argmax(logits, axis=-1)
    # 展平数组
    labels = labels.reshape(-1)
    predictions = predictions.reshape(-1)
    # 过滤掉填充的部分
    valid_indices = labels != -100
    labels = labels[valid_indices]
    predictions = predictions[valid_indices]
    # 计算准确率
    accuracy = accuracy_score(labels, predictions)
    return {'eval_accuracy': accuracy}


def compute_perplexity(loss):
    """
    计算困惑度。

    Args:
        loss (float): 交叉熵损失值。

    Returns:
        float: 困惑度值。
    """
    perplexity = math.exp(loss)
    return perplexity


def compute_bleu_score(reference, candidate):
    """
    计算 BLEU 分数。

    Args:
        reference (str): 参考文本。
        candidate (str): 生成的文本。

    Returns:
        float: BLEU 分数。
    """
    return sentence_bleu([reference.split()], candidate.split())


def compute_bleu_score_async(reference, candidate, callback):
    """
    异步计算 BLEU 分数。

    Args:
        reference (str): 参考文本。
        candidate (str): 生成的文本。
        callback (function): 计算完成后的回调函数。
    """
    def compute():
        bleu_score = compute_bleu_score(reference, candidate)
        callback(bleu_score)

    # 创建并启动线程
    thread = threading.Thread(target=compute)
    thread.start()


def plot_bleu_score_async(plotter, bleu_score):
    """
    异步绘制 BLEU 分数曲线。

    Args:
        plotter (TrainingPlotter): 绘图工具实例。
        bleu_score (float): BLEU 分数。
    """
    def plot():
        # 更新步骤计数器
        plotter.step_counter += 1
        # 记录 BLEU 分数和步骤
        plotter.bleu_scores.append(bleu_score)
        plotter.bleu_steps.append(plotter.step_counter)
        # 绘制 BLEU 分数曲线
        plotter.plot_bleu(plotter.bleu_scores, plotter.bleu_steps)

    # 创建并启动线程来绘制 BLEU 图像
    thread = threading.Thread(target=plot)
    thread.start()


def measure_code_coverage(script_path):
    """
    测量代码覆盖率。

    Args:
        script_path (str): 需要测量的脚本路径。

    Returns:
        float: 代码覆盖率百分比。
    """
    # 创建 Coverage 实例
    cov = coverage.Coverage()
    cov.start()

    # 执行脚本
    exec(open(script_path).read())

    cov.stop()
    cov.save()

    # 获取覆盖率报告
    coverage_percentage = cov.report()
    return coverage_percentage


def measure_code_coverage_async(script_path, callback):
    """
    异步测量代码覆盖率。

    Args:
        script_path (str): 需要测量的脚本路径。
        callback (function): 测量完成后的回调函数。
    """
    def measure():
        coverage_score = measure_code_coverage(script_path)
        callback(coverage_score)

    # 创建并启动线程
    thread = threading.Thread(target=measure)
    thread.start()


def plot_coverage_async(plotter, coverage_score):
    """
    异步绘制代码覆盖率曲线。

    Args:
        plotter (TrainingPlotter): 绘图工具实例。
        coverage_score (float): 代码覆盖率分数。
    """
    def plot():
        # 更新步骤计数器
        plotter.step_counter += 1
        # 记录代码覆盖率和步骤
        plotter.coverage_scores.append(coverage_score)
        plotter.coverage_steps.append(plotter.step_counter)
        # 绘制代码覆盖率曲线
        plotter.plot_coverage(plotter.coverage_scores, plotter.coverage_steps)

    # 创建并启动线程来绘制代码覆盖率图像
    thread = threading.Thread(target=plot)
    thread.start()


def plot_confusion_matrix_async(plotter, labels, predictions):
    """
    异步绘制混淆矩阵。

    Args:
        plotter (TrainingPlotter): 绘图工具实例。
        labels (array): 真实标签。
        predictions (array): 模型预测。
    """
    thread = threading.Thread(target=plotter.plot_confusion_matrix, args=(labels, predictions))
    thread.start()


def plot_test_loss_curve_async(plotter, losses):
    """
    异步绘制测试集损失曲线。

    Args:
        plotter (TrainingPlotter): 绘图工具实例。
        losses (list): 损失值列表。
    """
    thread = threading.Thread(target=plotter.plot_test_loss_curve, args=(losses,))
    thread.start()


def evaluate_on_test_set():
    """
    在测试集上评估模型，计算各种指标并绘制相应的图像。
    """
    # 清理显存
    torch.cuda.empty_cache()

    # 加载预处理好的测试集，选择前1000条数据
    tokenized_test_dataset = load_from_disk(config["test_data_path"]).select(range(1000))

    # 获取最近的检查点（如果存在）
    last_checkpoint = get_last_checkpoint(config["output_dir"])

    # 加载微调后的模型
    model = load_model(model_path=last_checkpoint if last_checkpoint else config["finetuned_model_dir"])

    # 创建绘图实例
    experiment_dir = config["output_dir"]
    plotter = TrainingPlotter(experiment_dir=experiment_dir)

    # 定义评估参数
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_eval_batch_size=32,  # 根据显存大小调整
        eval_accumulation_steps=10,
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        fp16=True,
        save_steps=500,  # 保存检查点的频率
        save_total_limit=2  # 保持最多 2 个检查点，节省存储空间
    )

    # 创建 Trainer 实例
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[ClearCacheCallback()]
    )

    # 在测试集上评估
    if last_checkpoint:
        print(f"从检查点继续评估：{last_checkpoint}")
        trainer.evaluate(resume_from_checkpoint=last_checkpoint)
    else:
        test_results = trainer.evaluate()
        # print(f"Test set results: {test_results}")

    # 清理显存
    torch.cuda.empty_cache()

    # 获取预测值和真实标签
    predictions = trainer.predict(tokenized_test_dataset).predictions
    labels = np.array(tokenized_test_dataset["labels"])

    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=-1)

    labels = labels.reshape(-1)
    predictions = predictions.reshape(-1)
    valid_indices = labels != -100
    labels = labels[valid_indices]
    predictions = predictions[valid_indices]

    # 将 predictions 和 labels 转换为字符串格式
    generated_code = " ".join(map(str, predictions))
    reference_code = " ".join(map(str, labels))

    # 计算 BLEU 分数并异步绘制
    def print_and_plot_bleu(bleu_score):
        print(f"BLEU Score: {bleu_score}")
        plot_bleu_score_async(plotter, bleu_score)

    compute_bleu_score_async(reference_code, generated_code, print_and_plot_bleu)

    # 计算代码覆盖率并异步绘制
    def print_and_plot_coverage(coverage_score):
        print(f"Code Coverage: {coverage_score}%")
        plot_coverage_async(plotter, coverage_score)

    # 假设生成的代码保存为临时文件 'generated_code.py'
    with open('generated_code.py', 'w') as f:
        f.write(generated_code)

    # 异步测量代码覆盖率
    measure_code_coverage_async('generated_code.py', print_and_plot_coverage)

    # 异步绘制混淆矩阵和测试集损失曲线
    plot_confusion_matrix_async(plotter, labels, predictions)
    plot_test_loss_curve_async(plotter, [test_results['eval_loss']])


if __name__ == "__main__":
    evaluate_on_test_set()
