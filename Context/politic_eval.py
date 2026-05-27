"""Politicalness classifier wrapper.

Provides a simple helper `handle_politicalness` that uses a zero-shot
classifier to return a 0..1 score indicating how political a piece of
text is (1.0 = strongly political).
"""

from transformers import pipeline

model_politicalness_pipe = pipeline(
    "zero-shot-classification", model="mlburnham/Political_DEBATE_large_v1.0"
)


def handle_politicalness(text) -> float:
    """Return a politicalness score for `text`.

    The function uses a zero-shot hypothesis template and maps the
    classifier's label scores into a single float where values closer to 1
    indicate stronger political content.

    Args:
        text: Input text to evaluate.

    Returns:
        Float between 0 and 1 representing politicalness.
    """

    prediction_politicalness_result = model_politicalness_pipe(
        text,
        ["is not", "is"],
        hypothesis_template="This text {} about politics.",
        multi_label=False,
    )
    predicted_class_politicalness = {"is not": "non-political", "is": "political"}[
        prediction_politicalness_result["labels"][0]
    ]
    if predicted_class_politicalness == "political":
        return prediction_politicalness_result["scores"][0]
    else:
        return 1 - prediction_politicalness_result["scores"][0]

if __name__ == "__main__":
    print(handle_politicalness("Penis"))