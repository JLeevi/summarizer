import openai
import os

class Summarizer():
    def __init__(self):
        self.engine = os.getenv('ENGINE')
        self.max_tokens = 80
    
    def complete(self, prompt):
        completion = openai.Completion.create(
            engine=self.engine,
            max_tokens=self.max_tokens,
            prompt=prompt
        )
        return completion.choices[0].text

    def fine_tune(self):
        pass