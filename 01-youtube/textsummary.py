import torch
import gradio as gr

from transformers import pipeline

model_path = "Model/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff"

text_summary = pipeline("summarization", model=model_path)


def summarize_text(text):
    return text_summary(text)[0]["summary_text"]
