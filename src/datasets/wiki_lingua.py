from datasets import load_dataset

def load_wiki_lingua(amount: int):
    dataset = load_dataset("wiki_lingua", "english")["train"]["article"]
    orig_texts = []
    abstr_summaries = []
    i = 0
    idx = 0
    
    while i < amount:
        wiki_lingua = dataset[idx]
        documents = wiki_lingua["document"]
        summaries = wiki_lingua["summary"]
        
        for document, summary in zip(documents, summaries):
            if len(document.split()) <= 1000:
                orig_texts.append(document)
                abstr_summaries.append(summary)
                i += 1
        
        idx += 1
    
    return orig_texts, abstr_summaries
    
print(load_wiki_lingua(1))