# checkpoint_utils.py

import os
from transformers import Trainer, TrainingArguments

def get_last_checkpoint(output_dir):
    # 获取最近的检查点目录，如果存在则返回
    if os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
        checkpoints = [
            os.path.join(output_dir, d) for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("checkpoint-")
        ]
        if checkpoints:
            # 返回最新的检查点
            return max(checkpoints, key=os.path.getctime)
    return None
