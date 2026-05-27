"""Emotion and hate-speech signal extraction utilities.

This module exposes `emotion_score` (anger/disgust/fear/sadness) and
`hate_speech_value` which returns a unified hate-speech probability.
"""

from pysentimiento import create_analyzer


analyzer = create_analyzer(task="sentiment", lang="es")


def emotion_score(text) -> float:
    """Estimate a composite negative-emotion score for `text`.

    The function averages the emotion probabilities for disgust, anger,
    sadness and fear to produce a single 0..1 score where higher values
    indicate stronger negative emotional content.
    """

    emotion_analyzer = create_analyzer(task="emotion", lang="en")
    res = emotion_analyzer.predict(text)
    disgust_points = res.probas["disgust"]
    anger_points = res.probas["anger"]
    sadness_points = res.probas["sadness"]
    fear_points = res.probas["fear"]
    total_points = (disgust_points + anger_points + sadness_points + fear_points) / 4
    if total_points < 0:
        total_points = 0
    return total_points


def hate_speech_value(text) -> float:
    """Return a hate-speech score for `text` (0..1).

    Uses the `hate_speech` analyzer to combine probabilities for hateful,
    targeted or aggressive language into a single score.
    """

    hate_speech_analyzer = create_analyzer(task="hate_speech", lang="en")
    res = hate_speech_analyzer.predict(text)
    hate_points = res.probas["hateful"]
    target_points = res.probas["targeted"]
    aggressive_points = res.probas["aggressive"]
    hate_speech_value = (hate_points + target_points + aggressive_points) / 3
    return hate_speech_value