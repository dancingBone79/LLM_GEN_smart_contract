# plotting.py

import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

class TrainingPlotter:
    def __init__(self, experiment_dir):
        """
        初始化绘图工具。

        Args:
            experiment_dir (str): 实验结果保存的目录。
        """
        # 初始化步骤计数器
        self.step_counter = 0

        # 初始化指标列表
        self.train_steps = []
        self.eval_steps = []
        self.train_losses = []
        self.eval_losses = []
        self.train_accuracies = []
        self.eval_accuracies = []
        self.train_f1s = []
        self.eval_f1s = []
        self.learning_rates = []
        self.grad_norms = []

        # 初始化 BLEU 分数和代码覆盖率列表
        self.bleu_scores = []
        self.bleu_steps = []
        self.coverage_scores = []
        self.coverage_steps = []

        # 设置实验目录
        self.experiment_dir = experiment_dir
        os.makedirs(self.experiment_dir, exist_ok=True)

    def log_train_metrics(self, step, train_loss, learning_rate, grad_norm):
        """
        记录训练集的指标。

        Args:
            step (int): 当前步骤。
            train_loss (float): 训练损失。
            learning_rate (float): 学习率。
            grad_norm (float): 梯度范数。
        """
        self.train_steps.append(step)
        self.train_losses.append(train_loss)
        self.learning_rates.append(learning_rate)
        self.grad_norms.append(grad_norm)

    def log_eval_metrics(self, step, eval_loss, eval_accuracy, eval_f1):
        """
        记录验证集的指标。

        Args:
            step (int): 当前步骤。
            eval_loss (float): 验证损失。
            eval_accuracy (float): 验证准确率。
            eval_f1 (float): 验证 F1 分数。
        """
        self.eval_steps.append(step)
        self.eval_losses.append(eval_loss)
        self.eval_accuracies.append(eval_accuracy)
        self.eval_f1s.append(eval_f1)

    def plot_loss(self):
        """
        绘制训练和验证损失曲线。
        """
        plt.figure(figsize=(6, 4))
        if self.train_steps and self.train_losses:
            plt.plot(self.train_steps, self.train_losses, label="Train Loss", color="blue", marker='o')
        if self.eval_steps and self.eval_losses:
            plt.plot(self.eval_steps, self.eval_losses, label="Eval Loss", color="red", marker='x')
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Loss Over Steps")
        if plt.gca().has_data():
            plt.legend()
            plt.savefig(os.path.join(self.experiment_dir, "loss_curve_step.png"))
        plt.close()

    def plot_accuracy(self):
        """
        绘制训练和验证准确率曲线。
        """
        plt.figure(figsize=(6, 4))
        if self.train_steps and self.train_accuracies:
            plt.plot(self.train_steps, self.train_accuracies, label="Train Accuracy", color="blue", marker='o')
        if self.eval_steps and self.eval_accuracies:
            plt.plot(self.eval_steps, self.eval_accuracies, label="Eval Accuracy", color="green", marker='x')
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Over Steps")
        if plt.gca().has_data():
            plt.legend()
            plt.savefig(os.path.join(self.experiment_dir, "accuracy_curve_step.png"))
        plt.close()

    def plot_f1(self):
        """
        绘制训练和验证 F1 分数曲线。
        """
        plt.figure(figsize=(6, 4))
        if self.train_steps and self.train_f1s:
            plt.plot(self.train_steps, self.train_f1s, label="Train F1 Score", color="blue", marker='o')
        if self.eval_steps and self.eval_f1s:
            plt.plot(self.eval_steps, self.eval_f1s, label="Eval F1 Score", color="purple", marker='x')
        plt.xlabel("Step")
        plt.ylabel("F1 Score")
        plt.title("F1 Score Over Steps")
        if plt.gca().has_data():
            plt.legend()
            plt.savefig(os.path.join(self.experiment_dir, "f1_curve_step.png"))
        plt.close()

    def plot_learning_rate(self):
        """
        绘制学习率曲线。
        """
        plt.figure(figsize=(6, 4))
        if self.train_steps and self.learning_rates:
            plt.plot(self.train_steps, self.learning_rates, label="Learning Rate", color="orange", marker='o')
        plt.xlabel("Step")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Over Steps")
        if plt.gca().has_data():
            plt.legend()
            plt.savefig(os.path.join(self.experiment_dir, "learning_rate_curve_step.png"))
        plt.close()

    def plot_grad_norm(self):
        """
        绘制梯度范数曲线。
        """
        plt.figure(figsize=(6, 4))
        if self.train_steps and self.grad_norms:
            plt.plot(self.train_steps, self.grad_norms, label="Grad Norm", color="cyan", marker='o')
        plt.xlabel("Step")
        plt.ylabel("Grad Norm")
        plt.title("Gradient Norm Over Steps")
        if plt.gca().has_data():
            plt.legend()
            plt.savefig(os.path.join(self.experiment_dir, "grad_norm_curve_step.png"))
        plt.close()

    def plot_confusion_matrix(self, labels, predictions):
        """
        绘制混淆矩阵。

        Args:
            labels (array): 真实标签。
            predictions (array): 模型预测。
        """
        cm = confusion_matrix(labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.experiment_dir, "confusion_matrix.png"))
        plt.close()

    def plot_test_loss_curve(self, losses):
        """
        绘制测试集损失曲线。

        Args:
            losses (list): 损失值列表。
        """
        plt.figure(figsize=(6, 4))
        plt.plot(range(len(losses)), losses, label="Test Loss", color="magenta", marker='o')
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Test Loss Curve")
        if plt.gca().has_data():
            plt.legend()
            plt.savefig(os.path.join(self.experiment_dir, "test_loss_curve.png"))
        plt.close()

    def plot_perplexity(self, perplexities, steps, save_path="perplexity_curve.png"):
        """
        绘制困惑度曲线。

        Args:
            perplexities (list): 困惑度列表。
            steps (list): 步骤列表。
            save_path (str): 保存路径。
        """
        plt.figure(figsize=(6, 4))
        plt.plot(steps, perplexities, label="Perplexity", color="blue", marker='o')
        plt.xlabel("Steps")
        plt.ylabel("Perplexity")
        plt.title("Perplexity Over Steps")
        if plt.gca().has_data():
            plt.legend()
            plt.savefig(os.path.join(self.experiment_dir, save_path))
        plt.close()

    def plot_bleu(self, bleu_scores, steps, save_path="bleu_curve.png"):
        """
        绘制 BLEU 分数曲线。

        Args:
            bleu_scores (list): BLEU 分数列表。
            steps (list): 步骤列表。
            save_path (str): 保存路径。
        """
        plt.figure(figsize=(6, 4))
        if bleu_scores and steps:
            plt.plot(steps, bleu_scores, label="BLEU Score", color="green", marker='x')
            plt.xlabel("Steps")
            plt.ylabel("BLEU Score")
            plt.title("BLEU Score Over Steps")
            if plt.gca().has_data():
                plt.legend()
                plt.savefig(os.path.join(self.experiment_dir, save_path))
            plt.close()
        else:
            print("No data available for plotting BLEU score.")

    def plot_coverage(self, coverage_scores, steps, save_path="coverage_curve.png"):
        """
        绘制代码覆盖率曲线。

        Args:
            coverage_scores (list): 代码覆盖率列表。
            steps (list): 步骤列表。
            save_path (str): 保存路径。
        """
        plt.figure(figsize=(6, 4))
        plt.plot(steps, coverage_scores, label="Code Coverage", color="purple", marker='o')
        plt.xlabel("Steps")
        plt.ylabel("Coverage (%)")
        plt.title("Code Coverage Over Steps")
        if plt.gca().has_data():
            plt.legend()
            plt.savefig(os.path.join(self.experiment_dir, save_path))
        plt.close()
