from datasets import load_metric
from typing import List


def calculate_rouge(predictions: List[str], references: List[str], confidence_level=1):
    # Confidence level
    # 0 = low
    # 1 = medium
    # 2 = high
    assert confidence_level in (0, 1, 2), "Confidence level needs to be 0, 1 or 2!"
    
    rouge = load_metric("rouge")
    all_scores = rouge.compute(predictions=predictions, references=references)
    
    final_scores = {}
    for score, aggregate_score in all_scores.items():
        final_scores[score] = aggregate_score[confidence_level]
        
    return final_scores


predictions = ["I really loved reading the Hunger Games"]
references = ["I loved reading the Hunger Games"]
    
print(calculate_rouge(predictions, references))