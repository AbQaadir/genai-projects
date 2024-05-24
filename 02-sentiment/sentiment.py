# Use a pipeline as a high-level helper
import torch
import transformers
import streamlit as st
from transformers import pipeline

model_path = "Model/models--distilbert--distilbert-base-uncased-finetuned-sst-2-english/snapshots/714eb0fa89d2f80546fda750413ed43d93601a13"

# pipe = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

analyser = pipeline("sentiment-analysis", model=model_path)

st.title("Sentiment Analysis")
text = st.text_area("Enter some text")

if st.button("Analyze"):
    result = analyser(text)
    st.write(result)
    st.write(result[0]['label'])
    st.write(result[0]['score'])
    if result[0]['label'] == 'POSITIVE':
        st.write("This text is positive")
    else:
        st.write("This text is negative")