from datasets import load_dataset
from typing import List
import numpy as np

def list_to_str(sentences: List[str]):
    string = ""
    for sentence in sentences:
        string += sentence + " "
    
    return [string.rstrip()]
  
def filter_1000_words(text: str):
    return len(text[0].split()) <= 1000

################################################################
# Wiki_lingua
################################################################

wiki_lingua = load_dataset("wiki_lingua", "english")
print(f"wiki_lingua: {wiki_lingua}")

################################################################
# Science tldr
################################################################

scitldr = load_dataset("scitldr", "AIC")["train"][:100] # or "Abstract"
sci_src = scitldr["source"]
sci_src = list(map(lambda x: list_to_str(x), sci_src))
sci_src = np.array(list(filter(lambda x: filter_1000_words(x), sci_src)))
sci_target = scitldr["target"]

print(len(sci_src))
print(sci_target)

################################################################
# Cnn_dailymail
################################################################

cnn_dailymail = load_dataset("cnn_dailymail", "3.0.0")["train"][:2]
cnn_src = cnn_dailymail["article"]
cnn_target = cnn_dailymail["highlights"] 

print(cnn_src)
print(cnn_target)

# ################################################################
# # XL_sum
# ################################################################

xl_sum = load_dataset("csebuetnlp/xlsum", "english")["train"]
print(xl_sum)