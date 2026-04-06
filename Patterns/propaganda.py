from typing import Any, Dict, Optional

import requests

BASE_URL = "https://apihub.tanbih.org"
PROPAGANDA_ENDPOINT = "/api/v1/propaganda-text-analysis/en"
TIMEOUT_SECONDS = 30
API_KEY = None  # Set your API key here if required


def post_json(endpoint: str, payload: Dict[str, str], token: Optional[str]) -> Dict[str, Any]:
    url = f"{BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        return data
    return {"result": data}


def extract_propaganda_result(response_data: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(response_data.get("result"), dict):
        return response_data["result"]
    return response_data


def normalize_confidence(confidence: Any) -> float:
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return 0.0

    if value > 1:
        value /= 100.0
    return max(0.0, min(1.0, value))


def propaganda_score(text):
    claim = text
    if not claim:
        raise SystemExit("Claim cannot be empty.")

    payload = {"data": claim}

    try:
        propaganda_response = post_json(PROPAGANDA_ENDPOINT, payload, API_KEY)
    except requests.HTTPError:
        return None

    result = extract_propaganda_result(propaganda_response)
    label = str(result.get("label", "")).strip().lower()
    confidence = normalize_confidence(result.get("confidence", 0.0))

    if label == "propaganda":
        return confidence
    if label == "not-propaganda":
        return 1.0 - confidence
    return None
if __name__ == "__main__":
    print(propaganda_score("The government is hiding the truth about the pandemic and is using propaganda to manipulate the public."))

