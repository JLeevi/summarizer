from datasets import load_dataset

def load_scitldr(amount: int):
    dataset = load_dataset("scitldr", "AIC")["train"]# or "Abstract"
    orig_texts = []
    abstr_summaries = []
    i = 0
    idx = 0
    
    while i < amount:
        scitldr = dataset[idx]

        sci_src = scitldr["source"]
        sci_src = " ".join(sci_src)
        
        sci_target = scitldr["target"]
        sci_target = " ".join(sci_target)
        
        # Include texts that are less than 1000 words long
        if len(sci_src.split()) <= 1000:
            orig_texts.append(sci_src)
            abstr_summaries.append(sci_target)
            i += 1
    
        idx += 1
        
    return orig_texts, abstr_summaries
    
# print(load_scitldr(1))