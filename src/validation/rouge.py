from datasets import load_metric
from typing import List


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


predictions = [" I really loved reading the Hunger Games"]
references = [" I loved reading the Hunger Games"]
    
print(calculate_rouge(predictions, references))