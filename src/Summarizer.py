import openai
import os
import subprocess

class Summarizer():
    def __init__(self, model_name=None):
        self.engine = os.getenv('ENGINE')
        self.max_tokens = 80
        self.model_name = model_name
    
    def complete(self, prompt):
        params = { "prompt": prompt, "max_tokens": self.max_tokens, "engine": self.engine }
        if self.model_name:
            params["model"] = self.model_name
        completion = openai.Completion.create(**params)
        return completion.choices[0].text

    def fine_tune(self, file_path, model_name=None):
        assert file_path != None, "file_path for fine-tuning data missing"
        command = [
            'openai', 'api', 'fine_tunes.create', 
            '-t', file_path, 
            '-m', 'curie', 
            '--n_epochs', '1',
            '--prompt_loss_weight', '0',
            '--batch_size', '2',
            '--learning_rate_multiplier', '0.05']
        if model_name:
            command += ['--suffix', model_name]
        self.model_name = model_name
        subprocess.call(command)