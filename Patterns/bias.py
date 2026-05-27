"""Bias detection wrapper.

Provides `bias_value` which returns a 0..1 score indicating the
perceived bias in `text` using a pretrained text-classification model.
"""

from transformers import pipeline


def bias_value(text) -> float:
    """Return a bias score for the given text.

    The function maps the model's label/scores into a single float where
    higher values indicate stronger bias.

    Args:
        text: Text to classify.

    Returns:
        A float between 0 and 1.
    """

    classifier = pipeline("text-classification", model="himel7/bias-detector", tokenizer="roberta-base")
    result = classifier(text)
    # if label 1, score = score, if label 0, score = 1 - score
    if result[0]['label'] == 'LABEL_1':
        score = result[0]['score']
        return score
    else:
        score = 1 - result[0]['score']
        return score
