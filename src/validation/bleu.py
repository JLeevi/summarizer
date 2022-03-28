from datasets import load_metric
from transformers import GPT2Tokenizer
from typing import List
from datasets import MetricInfo

def calculate_bleu(predictions: List[str], references: List[str], confidence_level=1, n=4):
    # Confidence level
    # 0 = low
    # 1 = medium
    # 2 = high
    assert confidence_level in (0, 1, 2), "Confidence level needs to be 0, 1 or 2!"
    
    # Tokenize predictions and references
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    preds = list(map(lambda pred: tokenizer.batch_decode(tokenizer(pred)["input_ids"]), predictions))
    refs = [list(map(lambda ref: tokenizer.batch_decode(tokenizer(ref)["input_ids"]), references))]
    
    # Calculate bleu scores from 1 to n
    bleu = load_metric("bleu")
    all_scores = []
    for deg in range(1, n + 1):
        score = bleu.compute(predictions=preds, references=refs, max_order=deg)["bleu"]
        all_scores.append(score)
        
    # return final_scores
    return all_scores

predictions = [" I really loved reading the Hunger Games"]
references = [" I loved reading the Hunger Games"]
    
print(calculate_bleu(predictions, references))