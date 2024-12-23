# callback.py

from transformers import TrainerCallback

class PlottingCallback(TrainerCallback):
    def __init__(self, plotter):
        self.plotter = plotter  # 传入绘图实例

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # 获取当前的损失、学习率、梯度范数和步数
            loss = logs.get("loss")
            learning_rate = logs.get("learning_rate")
            grad_norm = logs.get("grad_norm")
            step = state.global_step  # 使用 global_step 获取当前步骤

            if loss is not None and learning_rate is not None and grad_norm is not None:
                # 更新训练集的绘图数据
                self.plotter.log_train_metrics(
                    step=step,
                    train_loss=loss,
                    learning_rate=learning_rate,
                    grad_norm=grad_norm
                )

                 # 每 1000 步绘制一次
                if step % 600 == 0:
                    try:
                        # 更新训练集的图像
                        self.plotter.plot_loss()
                        self.plotter.plot_learning_rate()
                        self.plotter.plot_grad_norm()
                    except Exception as e:
                        print(f"绘图时发生异常：{e}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            # 获取当前的步数
            step = state.global_step

            # 提取验证集评估指标
            eval_loss = metrics.get("eval_loss", None)
            eval_accuracy = metrics.get("eval_accuracy", None)
            eval_f1 = metrics.get("eval_f1", None)

            if eval_loss is not None and eval_accuracy is not None and eval_f1 is not None:
                # 记录验证集的评估数据
                self.plotter.log_eval_metrics(
                    step=step,
                    eval_loss=eval_loss,
                    eval_accuracy=eval_accuracy,
                    eval_f1=eval_f1
                )

                # 绘制验证集的图像
                self.plotter.plot_loss()
                self.plotter.plot_accuracy()
                self.plotter.plot_f1()
