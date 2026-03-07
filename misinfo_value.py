import os

os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from huggingface_hub.utils import logging as hf_hub_logging
from transformers.utils import logging as transformers_logging

hf_hub_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

from Patterns.bias import bias_value
from Patterns.violence import violence_value
from translate import translate_and_preprocess
from Patterns.propaganda import propaganda_score
from Tone.tone import tone_value
from Patterns.emotion import emotion_score, hate_speech_value
from Patterns.fallacies import fallacy_score


input_text = input("Enter the text to analyze for misinformation: ")
context = input("Enter the value of the context: ")
veracity = input("Enter the value of the veracity: ")

text = translate_and_preprocess(input_text)
bias = bias_value(text)
violence = violence_value(text)
propaganda = propaganda_score(text)
tone = tone_value(text)
emotion = emotion_score(text)
hate_speech = hate_speech_value(text)
fallacy = fallacy_score(text)

misinfo_score = ((bias + violence + propaganda + emotion + hate_speech + fallacy) / 6)* 0.4 + tone * 0.2 + float(context) * 0.15 + float(veracity) * 0.25
print(f"Misinformation Score: {misinfo_score:.4f}")

"""premisa teórica"""
"""documento resumen"""