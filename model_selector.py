# model_selector.py

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import AutoTokenizer

class ModelSelector:
    def __init__(self, model_paths):
        """
        初始化模型选择器。
        Args:
            model_paths (dict): 包含模型路径的字典。例如：
                {
                    "code-understanding": "path_to_codet5",
                    "code-generation": "path_to_codegen"
                }
        """
        self.model_paths = model_paths
        self.models = {}
        self.tokenizers = {}

    def load_model(self, task):
        """
        加载模型和对应的分词器。
        Args:
            task (str): 任务名称，比如 "code-understanding" 或 "code-generation"。
        """
        if task not in self.models:
            model_path = self.model_paths[task]
            if task == "code-understanding":
                self.models[task] = AutoModelForSeq2SeqLM.from_pretrained(model_path).to('cuda')
                self.tokenizers[task] = AutoTokenizer.from_pretrained(model_path)
            elif task == "code-generation":
                self.models[task] = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')
                self.tokenizers[task] = AutoTokenizer.from_pretrained(model_path)
        return self.models[task], self.tokenizers[task]

    def predict(self, task, input_text, max_length=512):
        """
        使用指定任务的模型进行推理。
        Args:
            task (str): 任务名称，比如 "code-understanding" 或 "code-generation"。
            input_text (str): 输入文本。
            max_length (int): 生成文本的最大长度。
        Returns:
            str: 生成的文本。
        """
        model, tokenizer = self.load_model(task)
        inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs, max_length=max_length)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
