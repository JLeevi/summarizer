import json
from typing import List
from datetime import datetime, date

from load_data.scitldr import load_scitldr
from load_data.wiki_lingua import load_wiki_lingua
from load_data.xlsum import load_xlsum
from settings import training_amount, validation_amount, prompt_chat_bot
from utility import get_prompt


# Create dataset based on the distribution provided
def create_dataset(distribution: dict):
    # Download original datasets and their summaries
    sci_texts, sci_summaries = load_scitldr(distribution["scitldr"])
    wiki_texts, wiki_summaries = load_wiki_lingua(distribution["wiki_lingua"])
    xlsum_texts, xlsum_summaries = load_xlsum(distribution["xlsum"])
    
    # Concatenate the individual lists to single lists
    texts: List[str] = sci_texts + wiki_texts + xlsum_texts
    summaries: List[str] = sci_summaries + wiki_summaries + xlsum_summaries
    
    # Create the dictionary with the prompt and the original text as key, and the summary as the value
    plain_prompt = prompt_chat_bot
    text_to_summary = { plain_prompt.replace("[#####]", text):summary for text, summary in zip(texts, summaries) }
    
    # Get the current date and time
    today = date.today()
    current_date = today.strftime("%b-%d-%Y") # MM-DD-YY

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S") # HH:MM:SS
    
    # Write the data into a jsonl-file
    with open(f"../data/training/{current_date}_{current_time}.jsonl", 'w') as f:
        for prompt, summary in text_to_summary.items():
            line = { "prompt": prompt, "completion": summary}
            f.write(json.dumps(line) + "\n")
    
create_dataset(training_amount)