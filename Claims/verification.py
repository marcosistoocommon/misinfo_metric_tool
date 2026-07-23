"""Repo-local ClaimeAI adapter.

This module treats the cloned `ClaimeAI/` folder as an internal microservice
boundary without needing Docker or a separate process.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


CLAIMEAI_ROOT = Path(__file__).resolve().parents[1] / "ClaimeAI"
CLAIMEAI_AGENT_PATH = CLAIMEAI_ROOT / "apps" / "agent"
_AGENT_READY = False
_AGENT_ERROR = None



class ClaimeAIError(RuntimeError):
	"""Raised when the ClaimeAI backend cannot be initialized or executed."""


def _format_external_error(exc: Exception) -> str:
	"""Return a verbose, stable diagnostic string for upstream API failures."""
	parts: list[str] = [f"type={exc.__class__.__name__}"]

	status_code = getattr(exc, "status_code", None)
	if status_code is None:
		response = getattr(exc, "response", None)
		status_code = getattr(response, "status_code", None)
	if status_code is not None:
		parts.append(f"status_code={status_code}")

	code = getattr(exc, "code", None)
	if code:
		parts.append(f"code={code}")

	request_id = getattr(exc, "request_id", None)
	if request_id:
		parts.append(f"request_id={request_id}")

	message = str(exc).strip() or "<no message>"
	parts.append(f"message={message}")

	if int(status_code or 0) == 429:
		parts.append("classification=rate_limited")
	elif "APIConnectionError" in exc.__class__.__name__ or "Connection error" in message:
		parts.append("classification=connection_error")

	cause = exc.__cause__
	if cause is not None:
		parts.append(f"cause_type={cause.__class__.__name__}")
		cause_message = str(cause).strip()
		if cause_message:
			parts.append(f"cause_message={cause_message}")

	return ", ".join(parts)


def claimeai_available() -> bool:
	"""Return True when the repo-local ClaimeAI clone is present."""
	if CLAIMEAI_AGENT_PATH.is_dir() and str(CLAIMEAI_AGENT_PATH) not in sys.path:
		sys.path.insert(0, str(CLAIMEAI_AGENT_PATH))
	return CLAIMEAI_AGENT_PATH.is_dir()


def initialize_agent(force: bool = False) -> bool:
	"""Eagerly import and validate the ClaimeAI graphs."""
	global _AGENT_READY, _AGENT_ERROR

	if _AGENT_READY and not force:
		return True

	if not claimeai_available():
		_AGENT_READY = False
		_AGENT_ERROR = "ClaimeAI backend folder was not found."
		raise ClaimeAIError(_AGENT_ERROR)

	try:
		claim_extractor_module = importlib.import_module("claim_extractor")
		if getattr(claim_extractor_module, "graph", None) is None:
			raise ClaimeAIError("ClaimeAI claim extractor graph is not available.")

		claim_verifier_module = importlib.import_module("claim_verifier")
		if getattr(claim_verifier_module, "graph", None) is None:
			raise ClaimeAIError("ClaimeAI claim verifier graph is not available.")

		_AGENT_READY = True
		_AGENT_ERROR = None
		return True
	except ClaimeAIError:
		_AGENT_READY = False
		_AGENT_ERROR = str(sys.exc_info()[1])
		raise
	except Exception as exc:
		_AGENT_READY = False
		_AGENT_ERROR = str(exc)
		raise ClaimeAIError(f"ClaimeAI backend failed to initialize: {exc}") from exc


def agent_error() -> str | None:
	"""Return the last initialization/execution error, if any."""
	return _AGENT_ERROR


def extract_claims(text: str) -> list[str]:
	"""Extract claims from text using ClaimeAI claim extractor."""
	if not claimeai_available():
		raise ClaimeAIError("ClaimeAI backend folder was not found.")

	try:
		if not _AGENT_READY:
			initialize_agent()
		claim_extractor_module = importlib.import_module("claim_extractor")
		claim_extractor_graph = getattr(claim_extractor_module, "graph", None)
		if claim_extractor_graph is None:
			raise ClaimeAIError("ClaimeAI claim extractor graph is not available.")

		result = claim_extractor_graph.invoke({"answer_text": text})
		if not isinstance(result, dict):
			raise ClaimeAIError("ClaimeAI claim extractor returned an invalid payload.")

		validated_claims = result.get("validated_claims")
		if not validated_claims:
			# No validated claims were found; return empty list so caller can decide behavior
			return []

		claims: list[str] = []
		for claim in validated_claims:
			claim_text = getattr(claim, "claim_text", None) or getattr(claim, "text", None)
			if claim_text:
				claims.append(str(claim_text))

		if not claims:
			# No claim text present after validation
			return []

		return claims
	except Exception as exc:
		global _AGENT_ERROR
		detail = _format_external_error(exc)
		_AGENT_ERROR = detail
		if isinstance(exc, ClaimeAIError):
			raise
		raise ClaimeAIError(f"Error invoking ClaimeAI claim extractor: {detail}") from exc


def false_confidence(text: str, progress_callback=None) -> float:
	"""Return a 0..1 confidence that the text is false."""
	if progress_callback:
		progress_callback("extracting_claims")

	claims = extract_claims(text)
	# If extractor found no verifiable claims, score should be 0
	if not claims:
		if progress_callback:
			progress_callback("aggregating_verdict")
		return 0.0

	if progress_callback:
		progress_callback("verifying_claims")

	try:
		if not _AGENT_READY:
			initialize_agent()
		verdicts = _verify_claims_directly(claims)
		if not verdicts:
			raise ClaimeAIError("ClaimeAI claim verifier returned no verdicts.")

		if progress_callback:
			progress_callback("aggregating_verdict")
		return _score_from_verdicts(verdicts)
	except Exception as exc:
		global _AGENT_ERROR
		detail = _format_external_error(exc)
		_AGENT_ERROR = detail
		if isinstance(exc, ClaimeAIError):
			raise
		raise ClaimeAIError(f"Error invoking ClaimeAI fact checker: {detail}") from exc


def _verify_claims_directly(claims: list[str]) -> list[Any]:
	"""Verify extracted claims directly with the claim verifier graph."""
	claim_verifier_module = importlib.import_module("claim_verifier")
	claim_verifier_graph = getattr(claim_verifier_module, "graph", None)
	if claim_verifier_graph is None:
		raise ClaimeAIError("ClaimeAI claim verifier graph is not available.")

	verdicts: list[Any] = []
	for index, claim_text in enumerate(claims):
		claim_payload = {
			"claim_text": claim_text,
			"is_complete_declarative": True,
			"disambiguated_sentence": claim_text,
			"original_sentence": claim_text,
			"original_index": index,
		}
		result = claim_verifier_graph.invoke({"claim": claim_payload})
		if not isinstance(result, dict):
			raise ClaimeAIError("ClaimeAI claim verifier returned an invalid payload.")
		verdict = result.get("verdict")
		if verdict is None:
			raise ClaimeAIError(f"No verdict returned for claim: {claim_text}")
		verdicts.append(verdict)

	return verdicts


def _score_from_verdicts(verdicts: list[Any]) -> float:
	scores = [_map_verdict_to_false_confidence(verdict) for verdict in verdicts]
	if not scores:
		raise ClaimeAIError("ClaimeAI verifier did not return valid verdicts.")
	return max(0.0, min(1.0, sum(scores) / len(scores)))


def _map_verdict_to_false_confidence(verdict: Any) -> float:
	result = getattr(verdict, "result")
	result_value = getattr(result, "value", result)
	normalized = str(result_value).strip().lower()

	if normalized in {"refuted", "false", "contradicted"}:
		return 1.0
	if normalized in {"supported", "true"}:
		return 0.0
	return 0.4  # Uncertain or unknown verdicts get a moderate false confidence

if __name__ == "__main__":
	# Example usage
	sample_text = input("Enter text to analyze for false confidence: ")
	try:
		confidence = false_confidence(sample_text)
		print(f"False confidence for '{sample_text}': {confidence:.2f}")
	except ClaimeAIError as e:
		print(f"Error: {e}")