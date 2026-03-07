from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from transformers.utils import logging as hf_logging
import numpy as np
from scipy.special import softmax
from transformers import AutoModelForSeq2SeqLM
import sys

hf_logging.set_verbosity_error()

tokenizer_translate = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
model_translate = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")


def translate_to_english(text):
    encoded = tokenizer_translate(text, return_tensors="pt", truncation=True)
    generated = model_translate.generate(**encoded, max_new_tokens=256)
    return tokenizer_translate.decode(generated[0], skip_special_tokens=True)

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def translate_and_preprocess(text):
    translated_text = translate_to_english(text)
    processed_text = preprocess(translated_text)
    return processed_text