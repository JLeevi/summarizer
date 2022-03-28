import json
from typing import List
import os


from load_data.scitldr import load_scitldr
from load_data.wiki_lingua import load_wiki_lingua
from load_data.xlsum import load_xlsum
from prompt import PromptStyle
from settings import (
    training_amount,
    validation_amount,
    prompt_chat_bot,
    prompt_descriptive,
    prompt_basic,
    insert_summary,
    completion_start,
    completion_end)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def get_plain_prompt(prompt_style: PromptStyle):
    if prompt_style == PromptStyle.BASIC:
        return prompt_basic
    elif prompt_style == PromptStyle.DESCRIPTIVE:
        return prompt_descriptive
    else:
        return prompt_chat_bot

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
    
    # Create the dictionary with the prompt and the original text as key, and the summary as the value
    plain_prompt = get_plain_prompt(prompt_style)
    text_to_summary = { plain_prompt.replace(insert_summary, text):f"{completion_start}{summary}{completion_end}" for text, summary in zip(texts, summaries) }
    
    with open(file, 'w') as f:
        for prompt, summary in text_to_summary.items():
            line = { "prompt": prompt, "completion": summary }
            f.write(json.dumps(line) + "\n")
    
    return file