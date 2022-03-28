import openai
import os

class Summarizer():
    def __init__(self, model_name=None, max_tokens=80):
        self.engine = os.getenv('ENGINE')
        self.model_name = model_name
        self.max_tokens = max_tokens
    
    def complete(self, prompt: str):
        params = { "prompt": prompt, "max_tokens": self.max_tokens, "engine": self.engine }
        if self.model_name:
            params["model"] = self.model_name
        completion = openai.Completion.create(**params)
        return completion.choices[0].text

    def fine_tune(self, params: dict, train_file_id: str, validation_file_id: str = None):
        """
        Creates a fine-tune job through OpenAI's API.
        
        Parameters
        ----------
            train_file_id (str): File id of training file already uploaded to OpenAI.

            params (dict): Model's fine-tune parameters. See options at https://beta.openai.com/docs/api-reference/fine-tunes/create

            validation_file_id (str): File id of validation file already uploaded to OpenAI.
        """
        assert train_file_id != None, "file_id for fine-tuning training data missing"
        params = {
            **params,
            "training_file": train_file_id,
            "suffix": self.model_name
            }
        if validation_file_id:
            params["validation_file"] = validation_file_id
        return openai.FineTune.create(**params)