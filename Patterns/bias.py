from transformers import pipeline

def bias_value(text):
    classifier = pipeline("text-classification", model="himel7/bias-detector", tokenizer="roberta-base")
    result = classifier(text)
    """if label 1, score = score, if label 0, score = 1 - score"""
    if result[0]['label'] == 'LABEL_1':
        score = result[0]['score']
        return score
    else:
        score = 1 - result[0]['score']
        return score
