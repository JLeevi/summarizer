from typing import Any
import openai
import os

from prompt import PromptStyle

class Summarizer():
    def __init__(
        self,
        prompt_style: PromptStyle,
        model_name: str = None,
        max_tokens: int = 80,
        request_params: dict[str, Any] = {}):
        self.prompt_style = prompt_style
        self.params = {
            "engine": os.getenv('ENGINE'),
            "max_tokens": max_tokens,
            **request_params
        }
        if model_name:
            self.params["model"] = model_name
    
    def summarize(self, prompt: str):
        """
        Calls the OpenAI API to summarize the given prompt.

        Parameters
        ----------
        prompt (str): The text you want to summarize.

        Returns
        ----------
        summarization (str): Summarization of the given text.
        """
        params = { **self.params, "prompt": prompt }
        summarization = openai.Completion.create(**params)
        summarization = summarization.choices[0].text
        return summarization

    def set_request_params(self, **params: dict[str, Any]):
        """
        Sets request params for this class instance
        to be used in each summarization call.

        Parameters
        ----------
        **params (dict): Dict with the following optional keys:
            
            max_tokens (int): max length of the output

            best_of (int): (Generates best_of completions server-side and returns the "best")

            top_p (float): (An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.)
            
            echo (bool): (Echo back the prompt in addition to the completion)

            stop: str (Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.)

            temperature (float): (What sampling temperature to use. Higher values means the model will take more risks)

            frequency_penalty (float): (Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.)
            
            presence_penalty (float): (Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.)
        
        Returns
        ----------
        None
        """
        self.params = {**self.params, **params}

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