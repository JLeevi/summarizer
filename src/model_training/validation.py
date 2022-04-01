import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import json
from typing import List

from load_data import return_text_summaries
from model_training.metrics import calculate_bertscore, calculate_bleu, calculate_meteor, calculate_rouge
from prompt import PromptStyle
from Summarizer import Summarizer
from utility import create_prompt


def formatted_print(score_name: str) -> None:
    """Prints scores in a nicely formatted way
    
    Parameters
    ----------
    score_name (str): Name of the score to be printed
    
    Returns
    -------
    None"""
    print(f"\n{20 * '-'}\n")
    print(score_name + ":\n")


def create_validation_file(summarizer: Summarizer) -> str:
    """Takes in a validation distribution and creates summaries according to the distribution.
    Saves summaries with model id / hyperparameter string
    
    Paramaters
    ----------
    model (Summarizer): Summarizer class
    
    distribution: (dict): Distribution with dataset names as keys and amount of texts to summarize as values
    
    Returns
    -------
    file_path (str): File path to the validation file
    """
    
    # Use training false to use validation distribution 
    texts, summaries = return_text_summaries(training=False)
    
    dir_path = __file__.split("/src/")[0]
    file_path = f"{dir_path}/data/own_metrics/{summarizer.params['model']}_3.json"
    with open(file_path, "w") as f:
        lines: List[str] = []
        
        if summarizer.params.get("model", "not found") != "not found":
            summarizer.params.pop("engine")
        
        for text, summary in zip(texts, summaries):
            prompt = create_prompt(text, summarizer.prompt_style)
            prediction = summarizer.summarize(prompt)
            
            line = { 
                "prediction": prediction,
                "actual": summary
            }
            lines.append(line)        
            
        f.write(json.dumps({ "validation": lines, **summarizer.params }))
        
    return file_path


def validate(summarizer: Summarizer) -> dict:
    """Creates a validation file and returns the validation scores for the file
    
    Parameters
    ----------
    model (Summarizer): Summarizer class
    
    Returns
    -------
    scores (dict): Dictionary with names of the scores as keys and their scores as values"""
        
    file_path = create_validation_file(summarizer)
    
    with open(file_path, "r") as json_file:
        sentence_pairs: List[dict] = json.load(json_file)["validation"]
    
    # Create prediction and references lists
    predictions = []
    references = []
    
    for pair in sentence_pairs: 
        predictions.append(pair["prediction"])
        references.append(pair["actual"])
    
    # Calculate metrics
    meteor = calculate_meteor(predictions, references)
    bertscore = calculate_bertscore(predictions, references)
    bleu = calculate_bleu(predictions, references)
    rouge = calculate_rouge(predictions, references)
    
    # Print the scores in a formatted way
    formatted_print("Bertscore")
    print(f"Precision: {bertscore['precision']}")
    print(f"Recall: {bertscore['recall']}")
    print(f"F1: {bertscore['f1']}")
    
    formatted_print("Bleu")
    for bleu_name, bleu_score in bleu.items():
        print(f"{bleu_name}: {bleu_score}")
        
    formatted_print("Meteor")
    print(f"Meteor: {meteor['meteor']}")
    
    formatted_print("Rouge")
    for rouge_name, rouge_score in rouge.items():
        print(f"{rouge_name}: {rouge_score}")
    
    # Return the scores Dictionary
    scores = { "bertscore": bertscore, "bleu": bleu, "meteor": meteor, "rouge": rouge }
    return scores