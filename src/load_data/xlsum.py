from datasets import load_dataset

def load_xlsum(amount: int, training=True):
    dataset = load_dataset("csebuetnlp/xlsum", "english", split="train")
    orig_texts = []
    abstr_summaries = []
    i = 0
    idx = 1
    
    while i < amount:
        index = idx if training else -idx
        xlsum = dataset[index]
        text = xlsum["text"]
        summary = xlsum["summary"]
        
        if len(text.split()) <= 1000:
            orig_texts.append(text)
            abstr_summaries.append(summary)
            i += 1
        
        idx += 1
    
    return orig_texts, abstr_summaries
    
# print(load_xlsum(2))