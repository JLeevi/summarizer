from datasets import load_dataset

def load_xlsum(amount: int):
    dataset = load_dataset("csebuetnlp/xlsum", "english")["train"]
    orig_texts = []
    abstr_summaries = []
    i = 0
    idx = 0
    
    while i < amount:
        xlsum = dataset[idx]
        text = xlsum["text"]
        summary = xlsum["summary"]
        
        if len(text.split()) <= 1000:
            orig_texts.append(text)
            abstr_summaries.append(summary)
            i += 1
        
        idx += 1
    
    return orig_texts, abstr_summaries
    
print(load_xlsum(2))