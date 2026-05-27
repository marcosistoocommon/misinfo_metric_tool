from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax


"""Tone/sentiment utilities.

Provides `tone_value` which returns a sentiment score for input text.
"""

model_sentiment = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer_sentiment = AutoTokenizer.from_pretrained(model_sentiment)
config_sentiment = AutoConfig.from_pretrained(model_sentiment)

model = AutoModelForSequenceClassification.from_pretrained(model_sentiment)


def tone_value(text) -> float:
    """Return a single sentiment score for `text`.

    The function tokenizes the input, runs the sentiment model and returns
    the probability for the primary positive sentiment class as a float.
    """

    encoded_input = tokenizer_sentiment(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return scores[0]