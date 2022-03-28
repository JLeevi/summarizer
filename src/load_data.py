from datasets import load_dataset
from typing import List


def load_scitldr(amount: int, training=True):
    """Loads Science TLDRs with long texts and their summaries.
    
    Parameters
    ----------
    amount (int): Amount of articles and summaries returned in the lists
    
    training (bool): Starts the indexing from 1 during training, from -1 during validation
    
    Returns
    -------
    orig_texts: (list[str]): List of long, original texts
    abstr_summaries: (list[str]): List of shorter abstractive summarizations"""
    
    dataset = load_dataset("scitldr", "AIC", split="train")
    orig_texts: List[str]  = []
    abstr_summaries: List[str] = []
    i = 0
    idx = 1
    
    while i < amount:
        index = idx if training else -idx
        scitldr = dataset[index]

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


def load_wiki_lingua(amount: int, training=True):
    """Loads Wikihow articles with long texts and their summaries.
    
    Parameters
    ----------
    amount (int): Amount of articles and summaries returned in the lists
    
    training (bool): Starts the indexing from 1 during training, from -1 during validation
    
    Returns
    -------
    orig_texts: (list[str]): List of long, original texts
    abstr_summaries: (list[str]): List of shorter abstractive summarizations"""
    
    dataset = load_dataset("wiki_lingua", "english", split="train")["article"]
    orig_texts: List[str]  = []
    abstr_summaries: List[str] = []
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


def load_xlsum(amount: int, training=True):
    """Loads professionally annotated article-summary pairs from BBC.
    
    Parameters
    ----------
    amount (int): Amount of articles and summaries returned in the lists
    
    training (bool): Starts the indexing from 1 during training, from -1 during validation
    
    Returns
    -------
    orig_texts: (list[str]): List of long, original texts
    abstr_summaries: (list[str]): List of shorter abstractive summarizations"""
    
    dataset = load_dataset("csebuetnlp/xlsum", "english", split="train")
    orig_texts: List[str]  = []
    abstr_summaries: List[str] = []
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