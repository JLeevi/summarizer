from prompt import PromptStyle
from settings import (
    prompt_basic,
    prompt_chat_bot,
    prompt_descriptive,
    prompt_empty,
    token_padding,
    insert_summary,
    prompt_end,
    completion_end
)
def create_prompt(original_text: str, prompt_style: PromptStyle):
    """Creates prompt from original text and prompt style for correct tokenization.
    
    Parameters
    ----------
    original_text (str): Original texts
    
    prompt_style (PromptStyle): Defines the context for prompt creation
    
    Returns
    -------
    prompt (str): original text inserted into the the prompt context"""
    
    plain_prompt = ""
    if prompt_style == PromptStyle.BASIC:
        plain_prompt = prompt_basic
    elif prompt_style == PromptStyle.DESCRIPTIVE:
        plain_prompt = prompt_descriptive
    elif prompt_style == PromptStyle.CHAT:
        plain_prompt = prompt_chat_bot
    elif prompt_style == PromptStyle.EMPTY:
        plain_prompt = prompt_empty
    else:
        raise Exception(f"Valid prompt_style missing, got {prompt_style}")
        
    prompt = f"{plain_prompt.replace(insert_summary, original_text)}{prompt_end}"
    return prompt

def create_completion(summary: str):
    """Modifies summaries to right format for correct tokenization.
    
    Parameters
    ----------
    summary (str): Original summary
    
    Returns
    -------
    modified_summary (str): modified summary with whitespace at the start and completion string in the end"""
    
    modified_summary = f"{token_padding}{summary}{completion_end}"
    return modified_summary

def get_base_model_name(params):
    lr = str(params["learning_rate_multiplier"]).replace('.','')
    prompt_style = params["prompt_style"].name
    model_name = f"model-{prompt_style}-{lr}"
    model_name = model_name.lower()
    return model_name

def get_final_model_name(fine_tune_params, req_params):
    lr = str(fine_tune_params["learning_rate_multiplier"]).replace('.','')
    prompt_style = str(fine_tune_params["prompt_style"].name).replace('.','')
    temp = str(req_params["temperature"]).replace('.','')
    freq_pen = str(req_params["frequency_penalty"]).replace('.','')
    pres_pen = str(req_params["presence_penalty"]).replace('.','')
    model_name = f"lr-{lr}_prompt-{prompt_style}_temp-{temp}_freq-{freq_pen}_pres-{pres_pen}"
    model_name = model_name.lower()
    return model_name
