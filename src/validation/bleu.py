from datasets import load_metric
from transformers import GPT2Tokenizer
from typing import List

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
    preds = list(map(lambda pred: tokenizer.batch_decode(tokenizer(pred)["input_ids"]), predictions))
    refs = [list(map(lambda ref: tokenizer.batch_decode(tokenizer(ref)["input_ids"]), references))]
    
    # Calculate BLEU scores from 1 to n
    bleu = load_metric("bleu")
    all_scores = {}
    for deg in range(1, n + 1):
        score = bleu.compute(predictions=preds, references=refs, max_order=deg)["bleu"]
        all_scores[f"BLEU-{deg}"] = score
        
    return all_scores

predictions = [" I really loved reading the Hunger Games"]
references = [" I loved reading the Hunger Games"]
    
print(calculate_bleu(predictions, references))