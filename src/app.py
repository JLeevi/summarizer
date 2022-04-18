import streamlit as st

from Summarizer import Summarizer
from prompt import PromptStyle
from setup import setup

setup()

def create_summary():
    summarizer = Summarizer(PromptStyle.BASIC)
    text = st.session_state["text"]
    summary = summarizer.summarize(text)
    st.session_state.summary = summary

with st.sidebar:
    add_radio = st.radio(
        "Select summarization style",
        ("🤖 Deterministic", "🌶 Spicy")
    )

st.text_area("Text to summarize", key="text", height=400, placeholder="Write here...")
st.button("Create summary", on_click=create_summary)


if "summary" in st.session_state:
    st.write(st.session_state["summary"])