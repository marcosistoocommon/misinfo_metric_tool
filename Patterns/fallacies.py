import os

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

def fallacy_score(text) -> float:
    """Estimate a fallacy score for `text` using a pretrained classifier.

    The function loads a DistilBERT fallacy classifier, computes softmax
    over labels and returns a single positive score representing the
    prominence of the top non-`miscellaneous` label.

    Args:
        text: Input text to evaluate.

    Returns:
        Float >= 0 indicating estimated fallaciousness (higher => more fallacious).
    """

    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"

    model = AutoModelForSequenceClassification.from_pretrained(
        "q3fer/distilbert-base-fallacy-classification",
        use_safetensors=False,
    )
    tokenizer = AutoTokenizer.from_pretrained("q3fer/distilbert-base-fallacy-classification")

    inputs = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        logits = model(**inputs)
    scores = logits[0][0]
    scores = torch.nn.Softmax(dim=0)(scores)

    _, ranking = torch.topk(scores, k=scores.shape[0])
    ranking = ranking.tolist()
    miscpos = model.config.label2id["miscellaneous"]
    score = scores[ranking[0]].item() - scores[miscpos].item()
    if score < 0:
        score = 0
    return score

def main():
    """Simple CLI entrypoint for interactive testing of `fallacy_score`.

    Prompts the user for text and prints the computed fallacy score.
    """

    input_text = input("Enter the text to analyze for fallacies: ")
    score = fallacy_score(input_text)
    print(f"Fallacy score: {score:.4f}")

if __name__ == "__main__":
    main()