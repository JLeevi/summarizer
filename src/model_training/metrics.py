import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from datasets import load_metric
from typing import List
from transformers import GPT2Tokenizer
import nltk
import ssl

def calculate_bertscore(predictions: List[str], references: List[str]):
    """Calculates the precision, recall and f1 bertscores.
    
    Parameters
    ----------
    predictions (List[str]): String sequences predicted by the model
    
    references (List[str]): Target string sequences
    
    Returns
    -------
    score (dict): Dictionary with "precision", "recall" and "f1" as keys with floats between 0 and 1 as values."""

    bertscore = load_metric("bertscore")
    score = bertscore.compute(predictions=predictions, references=references, lang="en")
    score.pop("hashcode")
        
    return score


def calculate_bleu(predictions: List[str], references: List[str], confidence_level=1, n=4):
    """Calculates the BLEU-scores from 1 to n for the predictions.
    
    Parameters
    ----------
    predictions (List[str]): String sequences predicted by the model
    
    references (List[str]): Target string sequences
    
    confidence_level (int[0, 1, 2]): Defines the confidence level of the scoring.
    0 = low,
    1 = medium,
    2 = high,
    
    n (int): Degree of BLEU-score calculated, calculated BLEU score is BLEU-n
    
    Returns
    -------
    score (dict): Returns a dict with "BLEU-N" as keys and a float between 0 and 1 as the responding value"""
    
    assert confidence_level in (0, 1, 2), "Confidence level needs to be 0, 1 or 2!"
    assert n >= 1, "n has to be at least 1!"
    
    # Tokenize predictions and references
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")    
    preds = []
    refs = []
    
    for pred in predictions:
        if pred == "":
            preds.append([pred])
        else:
            preds.append(tokenizer.batch_decode(tokenizer(pred)["input_ids"]))

    for ref in references:
        if ref == "":
            refs.append([[ref]])
        else:
            refs.append([tokenizer.batch_decode(tokenizer(ref)["input_ids"])])
            
    print(preds)
    print(refs)
    
    # Calculate BLEU scores from 1 to n
    bleu = load_metric("bleu")
    all_scores = {}
    for deg in range(1, n + 1):
        score = bleu.compute(predictions=preds, references=refs, max_order=deg)["bleu"]
        all_scores[f"BLEU-{deg}"] = score
        
    return all_scores


def calculate_meteor(predictions: List[str], references: List[str]):
    """Calculates the METEOR-score for the predictions.
    
    Parameters
    ----------
    predictions (List[str]): String sequences predicted by the model
    
    references (List[str]): Target string sequences
    
    Returns
    -------
    score (dict): Returns a dict with "meteor" as a key and a float score between 0 and 1 as the responding value"""
    
    # Disable SSL checking: https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download("punkt")

    # Calculate meteor score
    meteor = load_metric("meteor")
    score = meteor.compute(predictions=predictions, references=references)
    return score


def calculate_rouge(predictions: List[str], references: List[str], confidence_level=1, n=4):
    """Calculates Rouge-N, Rouge-L and Rouge-Lsum scores for the predictions.
    
    NOTE! The documentation asks to give preds and refs as a string where tokens are seperated by whitespace.
    We just inserted the senteces as they were since GPT-3's BPE tokenization often has spaces prefixed to words.
    
    Parameters
    ----------
    predictions (List[str]): String sequences predicted by the model
    
    references (List[str]): Target string sequences
    
    confidence_level (int[0, 1, 2]): Defines the confidence level of the scoring.
    0 = low,
    1 = medium,
    2 = high,
    
    n (int): Number of ROUGE-scores calculated, scores are calculated from ROUGE-1 to ROUGE-N
    
    Returns
    -------
    score (dict): Returns a dict with different rouge-scores as keys and floats between 0 and 1 as the responding values"""
    
    assert confidence_level in (0, 1, 2), "Confidence level needs to be 0, 1 or 2!"
    assert n >= 1, "n has to be at least 1!"
    
    rouge = load_metric("rouge")
    score_types = [f"rouge{i}" for i in range(1, n + 1)] + ["rougeL", "rougeLsum"]
    all_scores = rouge.compute(predictions=predictions, references=references, rouge_types=score_types)
    
    final_scores = {}
    for score, aggregate_score in all_scores.items():
        final_scores[score] = aggregate_score[confidence_level]
        
    return final_scores

# predictions = [" We truly adored those books!", " I hated that so much"]
# references = [" I loved this book a lot!", " I hate you."]

# print(calculate_meteor(predictions=predictions, references=references))
# print(calculate_bertscore(predictions=predictions, references=references))
# print(calculate_bleu(predictions=predictions, references=references))
# print(calculate_rouge(predictions=predictions, references=references))

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# print(tokenizer.batch_decode(tokenizer("yees")["input_ids"]))
