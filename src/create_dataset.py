import json
import datetime
from typing import List
import os

from load_data import return_text_summaries
from prompt import PromptStyle
from utility import create_prompt, create_completion


# Create dataset based on the distribution provided
def create_dataset(
    training=True,
    prompt_style: PromptStyle = PromptStyle.BASIC,
    force_recreate=False) -> str:
    """Creates a training/validation dataset and writes in to a jsonl-file.
    
    Parameters
    ----------
    training (bool): Defines if the created file is for training or validation
    
    prompt_style (PromptStyle): Defines the prompt style that will be used to create the completions
    
    force_recreate (bool): If true, existing file will be overwritten
    
    Returns
    -------
    file_path (str): Path to the created dataset file"""
    
    root_path: str = __file__.split("/src/")[0]
    dir_path: str = f"training" if training else "validation"
    timestamp = datetime.datetime.now()
    timestamp = timestamp.strftime("%H:%M:%S")
    file_path: str = f"{root_path}/data/{dir_path}/{prompt_style.name}-{timestamp}.jsonl"

    if os.path.exists(file_path) and not force_recreate:
        return file_path
    
    # Download original datasets and their summaries to lists
    texts, summaries = return_text_summaries(training)
    
    # Create json-file with prompt and completion as keys
    with open(file_path, 'w') as f:
        for text, summary in zip(texts, summaries):
            line = { 
                "prompt": create_prompt(text, prompt_style),
                "completion": create_completion(summary)
            }
            f.write(json.dumps(line) + "\n")
    
    return file_path

create_dataset(training=False)