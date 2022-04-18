# Dataset distributions
training_amount = {
  "scitldr": 1000,
  "wiki_lingua": 800,
  "xlsum": 1200
}

validation_amount = {
  "scitldr": 5,
  "wiki_lingua": 2,
  "xlsum": 5
}

# Prompt settings
prompt_end = "\n\n###\n\n"
insert_summary = "[#####]"

# Completion settings
token_padding = " "
completion_end = " %%%%%"

# Different prompts
prompt_basic = f"""Summarize the following text abstractively:

The following text:
{insert_summary}

Summary:"""

prompt_descriptive = f"""Abstractive summarization is the task of generating a short and concise summary that captures the core ideas of the source text. The generated summaries potentially contain new phrases and sentences that may not appear in the source text. 

Create an abstractive summary of the following text. 

The following text: 
{insert_summary}

Summary:"""

prompt_chat_bot = f"""Prompt: The following is a conversation between a human who wants to get a summary of a text and an AI system capable of creating abstractive summaries of text.

Human: Hi, how are you doing?
AI: Hi, I’m doing great. What can I do to help you?
Human: Could you please summarise abstractively the following text for me?

The following text:
{insert_summary}

AI: Sure, here is the summary:"""

prompt_empty = f"""{insert_summary}"""
