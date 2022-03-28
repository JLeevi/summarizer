from datasets import load_dataset

def load_wiki_lingua(amount: int, training=True):
    dataset = load_dataset("wiki_lingua", "english", split="train")["article"]
    orig_texts = []
    abstr_summaries = []
    i = 0
    idx = 1
    
    while i < amount:
        index = idx if training else -idx
        wiki_lingua = dataset[index]
        documents = wiki_lingua["document"]
        summaries = wiki_lingua["summary"]
        
        for document, summary in zip(documents, summaries):
            if len(document.split()) <= 1000 and i < amount:
                orig_texts.append(document)
                abstr_summaries.append(summary)
                i += 1
        
        idx += 1
    
    return orig_texts, abstr_summaries
    
# print(load_wiki_lingua(1))