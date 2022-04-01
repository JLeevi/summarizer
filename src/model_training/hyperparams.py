import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from typing import Literal
import numpy as np
from prompt import PromptStyle
from settings import completion_end

CURIE_MODEL = "curie"

fixed_fine_tune_params = {
    "model": CURIE_MODEL,
    "batch_size": 2,
    "prompt_loss_weight": 0,
    "n_epochs": 1
}

tunable_fine_tune_params = {
    "learning_rate_multiplier": [0.02, 0.03, 0.04],
    "prompt_style": [PromptStyle.BASIC, PromptStyle.DESCRIPTIVE, PromptStyle.CHAT]
}

fixed_request_params = {
    "max_tokens": 100,
    "best_of": 1,
    "top_p": 1,
    "echo": False,
    "stop": completion_end
}

tunable_request_params = {
    "temperature": [0.25, 0.5, 0.75],
    "frequency_penalty": [0.0, 0.1, 0.2],
    "presence_penalty": [0.0, 0.1, 0.2],
}

def get_max_combinations(params):
    """Returns maximum amount of combinations with given hyperparameters
    
    Parameters
    ----------
    params (dict): dictionary with fine-tuning/completion parameters as keys and responding parameter as values
    
    Returns
    -------
    max_combinations (int): maximum number of combinations"""
    max_combinations = 1
    for _, values in params.items():
        max_combinations *= len(values)
    return max_combinations

def contains_object(arr, b):
    for a in arr:
        is_same = True
        for key in a.keys():
            if a[key] != b[key]:
                is_same = False
                break
        if is_same:
            return True
    return False

def get_random_param_options(
    type: Literal["FINE_TUNE", "REQUEST"],
    n=5,
    get_all=False):
    tunable_params = tunable_fine_tune_params if (
        type == "FINE_TUNE") else tunable_request_params
    max_N = get_max_combinations(tunable_params)
    if get_all: n = max_N
    assert n <= max_N, f"n too large, max options is {max_N}"
    param_options = []
    i = 0
    while i < n:
        params = {}
        for key, values in tunable_params.items():
            params[key] = np.random.choice(values, 1)[0]
        if not contains_object(param_options, params):
            param_options.append(params)
            i += 1
    fixed_params = fixed_fine_tune_params if (
        type == "FINE_TUNE") else fixed_request_params
    param_options = [ {**p, **fixed_params} for p in param_options ]
        
    return param_options