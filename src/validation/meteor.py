import nltk
import ssl
from datasets import load_metric
from typing import List


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


predictions = [" I really loved reading the Hunger Games"]
references = [" I loved reading the Hunger Games"]
    
print(calculate_meteor(predictions, references))