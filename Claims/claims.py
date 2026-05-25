from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import os
import requests

# ============================================================================
# MODEL 1: CLAIM CLASSIFICATION (_clf)
# ============================================================================

LABELS_clf = ["NFS", "UFS", "CFS"]  # Non-Factual, Unimportant Factual, Check-worthy Factual

model_clf = AutoModelForSequenceClassification.from_pretrained("lucafrost/ClaimBuster-DeBERTaV2")
model_clf.eval()

tokenizer_clf = AutoTokenizer.from_pretrained("lucafrost/ClaimBuster-DeBERTaV2")


def process_output_clf(outputs_clf, text):
    """Process SequenceClassifierOutput for claim classification model."""
    logits_clf = outputs_clf.logits
    
    # Convert logits to probabilities using softmax
    probabilities_clf = F.softmax(logits_clf, dim=-1)
    
    # Get predicted class and confidence
    predicted_class_idx_clf = torch.argmax(probabilities_clf, dim=-1).item()
    confidence_clf = probabilities_clf[0, predicted_class_idx_clf].item()
    
    # Get all probabilities for reference
    prob_dict_clf = {LABELS_clf[i]: probabilities_clf[0, i].item() for i in range(len(LABELS_clf))}
    
    return {
        "text": text,
        "predicted_class_clf": LABELS_clf[predicted_class_idx_clf],
        "class_index_clf": predicted_class_idx_clf,
        "confidence_clf": confidence_clf,
        "probabilities_clf": prob_dict_clf,
        "logits_clf": logits_clf[0].detach().tolist()
    }


# ============================================================================
# MODEL 2: CLAIM VERIFICATION (_ver) - Google Fact Check API
# ============================================================================
