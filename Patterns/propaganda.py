import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import requests

BASE_URL = "https://apihub.tanbih.org"
PROPAGANDA_ENDPOINT = "/api/v1/propaganda-detection/en"
TIMEOUT_SECONDS = 30

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

API_KEY = os.environ["PROPAGANDA_API_KEY"]


def post_json(endpoint: str, payload: Dict[str, str], token: Optional[str]) -> Dict[str, Any]:
    """POST `payload` as JSON to the configured Tanbih propaganda API.

    Raises a RuntimeError when the upstream API returns a non-2xx response
    with any available diagnostic detail.
    """
    url = f"{BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
    if not response.ok:
        detail = response.text.strip()
        if detail:
            try:
                error_payload = response.json()
            except ValueError:
                error_payload = None
            if isinstance(error_payload, dict):
                detail = str(error_payload.get("detail") or error_payload.get("message") or detail)
        else:
            detail = "no response body"
        raise RuntimeError(f"Propaganda API rejected the request (HTTP {response.status_code}): {detail}")
    data = response.json()
    if isinstance(data, dict):
        return data
    return {"result": data}


def extract_propaganda_result(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the API response payload to a consistent result dict.

    Some API responses return a top-level `result` dict while others
    embed the data directly. This helper returns the inner `result` when
    present, otherwise returns the original payload.
    """

    if isinstance(response_data.get("result"), dict):
        return response_data["result"]
    return response_data


def normalize_confidence(confidence: Any) -> float:
    """Convert various numeric/confidence representations to 0..1 float.

    Accepts strings, integers (0..100), or floats and clamps the result
    into the inclusive 0..1 range.
    """

    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return 0.0

    if value > 1:
        value /= 100.0
    return max(0.0, min(1.0, value))


def _normalize_label(label: Any) -> str:
    """Normalize labels returned by the API for predictable matching.

    Returns a lowercased string with underscores replaced by dashes.
    """

    return str(label or "").strip().lower().replace("_", "-")


def propaganda_score(text) -> float:
    """Return a 0..1 score indicating how propagandistic `text` is.

    The function calls the external Tanbih propaganda-detection API and
    normalizes the returned label/probability into a single float where
    higher values indicate stronger propaganda signals.
    """

    claim = text
    if not claim:
        raise SystemExit("Claim cannot be empty.")

    payload = {"data": claim}

    propaganda_response = post_json(PROPAGANDA_ENDPOINT, payload, API_KEY)

    result = extract_propaganda_result(propaganda_response)
    label = _normalize_label(result.get("label"))
    probability = normalize_confidence(result.get("probability", result.get("confidence", 0.0)))

    if label in {"propagandistic"}:
        return probability
    if label in {"non-propagandistic"}:
        return 1.0 - probability

    raise ValueError(
        "Propaganda API returned an unrecognized label: "
        f"{result.get('label')!r}"
    )
if __name__ == "__main__":
    print(propaganda_score("The government is hiding the truth about the pandemic and is using propaganda to manipulate the public."))

