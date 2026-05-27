"""Violence detection helper.

Expose `violence_value` which returns a probability-like score for
violent content using a pretrained StaticModelPipeline.
"""

from model2vec.inference import StaticModelPipeline


def violence_value(text) -> float:
    """Return a probability that `text` contains violent content.

    Args:
        text: The input text to evaluate.

    Returns:
        Float in 0..1 where higher values indicate stronger violence signals.
    """

    model = StaticModelPipeline.from_pretrained(
        "enguard/small-guard-32m-en-prompt-violence-binary-moderation"
    )
    prob = model.predict_proba([text])
    return prob[0][0]