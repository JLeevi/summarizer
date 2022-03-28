from datasets import load_metric
from typing import List

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

predictions = [" I really loved reading the Hunger Games"]
references = [" I loved reading the Hunger Games"]
    
print(calculate_bertscore(predictions, references))