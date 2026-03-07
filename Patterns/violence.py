from model2vec.inference import StaticModelPipeline


def violence_value(text):
    model = StaticModelPipeline.from_pretrained(
    "enguard/small-guard-32m-en-prompt-violence-binary-moderation"
    )
    prob = model.predict_proba([text])
    return prob[0][0]