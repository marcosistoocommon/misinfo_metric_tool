from pysentimiento import create_analyzer
analyzer = create_analyzer(task="sentiment", lang="es")

"""
Emotion Analysis in English
"""



def emotion_score(text):
    emotion_analyzer = create_analyzer(task="emotion", lang="en")
    res= emotion_analyzer.predict(text)
    disgust_points = res.probas["disgust"]
    anger_points = res.probas["anger"]
    joy_points = res.probas["joy"]
    sadness_points = res.probas["sadness"]
    surprise_points = res.probas["surprise"]
    fear_points = res.probas["fear"]
    other_points = res.probas["others"]
    total_points = (disgust_points + anger_points - joy_points + sadness_points - surprise_points + fear_points - other_points)/4
    if total_points < 0:
        total_points = 0
    return total_points


"""
Hate Speech (misogyny & racism)
"""
def hate_speech_value(text):
    hate_speech_analyzer = create_analyzer(task="hate_speech", lang="en")
    res = hate_speech_analyzer.predict(text)
    hate_points = res.probas["hateful"]
    target_points = res.probas["targeted"]
    aggressive_points = res.probas["aggressive"]
    hate_speech_value= (hate_points + target_points + aggressive_points) / 3
    return hate_speech_value