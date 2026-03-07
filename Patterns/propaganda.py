import json
from typing import Any, Dict, List, Optional, Tuple

import requests

BASE_URL = "https://apihub.tanbih.org"
PROPAGANDA_ENDPOINT = "/api/v1/propaganda-detection/en"
TECHNIQUE_ENDPOINT = "/api/v1/propaganda-technique-detection/en"
TIMEOUT_SECONDS = 30
API_KEY="YOUR_API_KEY"



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


def _walk_values(obj: Any) -> List[Any]:
    values: List[Any] = []
    if isinstance(obj, dict):
        for value in obj.values():
            values.extend(_walk_values(value))
    elif isinstance(obj, list):
        for value in obj:
            values.extend(_walk_values(value))
    else:
        values.append(obj)
    return values


def extract_propaganda_status(response_data: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    value = response_data["result"]
    return value


def extract_techniques(response_data: Dict[str, Any]) -> List[str]:
    string_values = [v for v in _walk_values(response_data) if isinstance(v, str)]
    cleaned = [s.strip() for s in string_values if s and len(s.strip()) > 2]
    return cleaned[:5]


def propaganda_score(text):
    claim = text
    if not claim:
        raise SystemExit("Claim cannot be empty.")

    payload = {"data": claim}

    try:
        propaganda_response = post_json(PROPAGANDA_ENDPOINT, payload, API_KEY)
        technique_response = post_json(TECHNIQUE_ENDPOINT, payload, API_KEY)
    except requests.HTTPError:
        return None

    status_text= extract_propaganda_status(propaganda_response)
    techniques = extract_techniques(technique_response)
    propaganda_score = 0
    technique_score = 0
    if status_text == "propagandistic":
        propaganda_score = 1

    if techniques:
        for technique in techniques:
            if technique.lower()!="other" and technique.lower()!= "no-technique":
                technique_score += 1
            elif technique.lower() == "other":
                technique_score += 0.5

    total_score = propaganda_score*0.5 + technique_score/techniques.__len__()*0.5 if techniques else 0
    return total_score

print(propaganda_score("The government is hiding the truth about the pandemic and is using propaganda to manipulate the public."))

