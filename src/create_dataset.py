import json
from typing import List
import os


from load_data import load_scitldr, load_wiki_lingua, load_xlsum
from prompt import PromptStyle
from settings import (
    training_amount,
    validation_amount,
)
from utility import create_prompt, create_completion

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Create dataset based on the distribution provided
def create_dataset(
    training=True,
    prompt_style: PromptStyle = PromptStyle.BASIC,
    force_recreate=False) -> str:
    
    distribution = training_amount if training else validation_amount
    file_path = f"training" if training else "test"
    file = f"../data/{file_path}_{prompt_style.name}.jsonl"

    if os.path.exists(file) and not force_recreate:
        return file
    
    # Download original datasets and their summaries
    sci_texts, sci_summaries = load_scitldr(distribution["scitldr"], training)
    wiki_texts, wiki_summaries = load_wiki_lingua(distribution["wiki_lingua"], training)
    xlsum_texts, xlsum_summaries = load_xlsum(distribution["xlsum"], training)
    
    # Concatenate the individual lists to single lists
    texts: List[str] = sci_texts + wiki_texts + xlsum_texts
    summaries: List[str] = sci_summaries + wiki_summaries + xlsum_summaries
    
    # Create json-file with prompt and completion as keys
    with open(file, 'w') as f:
        for text, summary in zip(texts, summaries):
            line = { 
                "prompt": create_prompt(text, prompt_style),
                "completion": create_completion(summary)
            }
            f.write(json.dumps(line) + "\n")
    
    return file