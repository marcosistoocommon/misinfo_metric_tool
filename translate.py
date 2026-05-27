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
    """Translate `text` from Spanish to English using Helsinki OPUS model.

    Args:
        text: Input text (assumed Spanish or similar) to translate.

    Returns:
        Translated English text as a string.
    """

    encoded = tokenizer_translate(text, return_tensors="pt", truncation=True)
    generated = model_translate.generate(**encoded, max_new_tokens=256)
    return tokenizer_translate.decode(generated[0], skip_special_tokens=True)

def preprocess(text):
    """Lightweight preprocessing used before pattern/tone models.

    - Strips leading `@` from mentions
    - Replaces http/https links with the token `link`

    Returns the cleaned text.
    """

    new_text = []
    for t in text.split(" "):
        if t.startswith('@'):
            t = t[1:]
            if not t:
                continue
        elif t.startswith('http'):
            t = 'link'
        new_text.append(t)
    return " ".join(new_text)

def translate_and_preprocess(text):
    """Translate the input to English and run `preprocess`.

    This helper is used by the pattern/tone pipeline to ensure models
    receive normalized English text.
    """

    translated_text = translate_to_english(text)
    processed_text = preprocess(translated_text)
    return processed_text