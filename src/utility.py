from prompt import PromptStyle
from settings import (
    prompt_basic,
    prompt_chat_bot,
    prompt_descriptive,
    token_padding,
    insert_summary,
    prompt_end,
    completion_end
)

def get_prompt(filename):
    with open(filename, 'r') as f:
        return f.read()
    
def create_prompt(original_text: str, prompt_style: PromptStyle):
    plain_prompt = ""
    if prompt_style == PromptStyle.BASIC:
        plain_prompt = prompt_basic
    elif prompt_style == PromptStyle.DESCRIPTIVE:
        plain_prompt = prompt_descriptive
    else:
        plain_prompt = prompt_chat_bot
    
    return f"{token_padding}{plain_prompt.replace(insert_summary, original_text)}{prompt_end}"

def create_completion(summary: str):
    return f"{token_padding}{summary}{completion_end}"