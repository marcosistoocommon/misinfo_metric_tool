from transformers import pipeline
model_politicalness_pipe = pipeline(
    "zero-shot-classification", model="mlburnham/Political_DEBATE_large_v1.0"
)
def handle_politicalness(text):


    prediction_politicalness_result = model_politicalness_pipe(
        text,
        ["is not", "is"],
        hypothesis_template="This text {} about politics.",
        multi_label=False,
    )
    predicted_class_politicalness = {"is not": "non-political", "is": "political"}[
        prediction_politicalness_result["labels"][0]
    ]
    return predicted_class_politicalness, prediction_politicalness_result["scores"][0]

print(handle_politicalness("The government has announced a new policy on climate change."))