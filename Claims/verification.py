"""Repo-local ClaimeAI adapter.

This module treats the cloned `ClaimeAI/` folder as an internal microservice
boundary without needing Docker or a separate process. The app can import this
module directly, and if the ClaimeAI agent packages are available, it will use
them; otherwise it falls back to a neutral heuristic.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import re
import sys
from pathlib import Path
from typing import Any


CLAIMEAI_ROOT = Path(__file__).resolve().parents[1] / "ClaimeAI"
CLAIMEAI_AGENT_PATH = CLAIMEAI_ROOT / "apps" / "agent"
_AGENT_READY = False
_AGENT_ERROR = None


class ClaimeAIError(RuntimeError):
	"""Raised when the ClaimeAI backend cannot be initialized or executed."""


def claimeai_available() -> bool:
	"""Return True when the repo-local ClaimeAI clone is present."""
	if CLAIMEAI_AGENT_PATH.is_dir() and str(CLAIMEAI_AGENT_PATH) not in sys.path:
		sys.path.insert(0, str(CLAIMEAI_AGENT_PATH))
	return CLAIMEAI_AGENT_PATH.is_dir()


def initialize_agent(force: bool = False) -> bool:
	"""Eagerly import and validate the ClaimeAI graphs.

	This is safe to call from `app.py` at startup. It returns True when the
	agent can be used and raises `ClaimeAIError` with a human-readable message
	when initialization fails.
	"""
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

		fact_checker_module = importlib.import_module("fact_checker")
		if getattr(fact_checker_module, "graph", None) is None:
			raise ClaimeAIError("ClaimeAI fact checker graph is not available.")

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
	"""Extract claims from text using ClaimeAI when available."""
	if claimeai_available():
		try:
			if not _AGENT_READY:
				initialize_agent()
			_notify = None
			claim_extractor_module = importlib.import_module("claim_extractor")
			claim_extractor_graph = getattr(claim_extractor_module, "graph", None)
			if claim_extractor_graph is None:
				raise ClaimeAIError("ClaimeAI claim extractor graph is not available.")
			result = _run_async(claim_extractor_graph.ainvoke({"answer_text": text}))
			validated_claims = result.get("validated_claims") if isinstance(result, dict) else None
			if validated_claims:
				claims: list[str] = []
				for claim in validated_claims:
					claim_text = getattr(claim, "claim_text", None) or getattr(claim, "text", None)
					if claim_text:
						claims.append(str(claim_text))
				if claims:
					return claims
		except Exception:
			if isinstance(sys.exc_info()[1], ClaimeAIError):
				raise
			raise ClaimeAIError("Error invoking ClaimeAI claim extractor.") from sys.exc_info()[1]

	return _sentence_split(text)


def false_confidence(text: str, progress_callback=None) -> float:
	"""Return a 0..1 confidence that the text is false."""
	if progress_callback:
		progress_callback("extracting_claims")

	claims = extract_claims(text)

	if progress_callback:
		progress_callback("verifying_claims")

	if claimeai_available():
		try:
			if not _AGENT_READY:
				initialize_agent()
			fact_checker_module = importlib.import_module("fact_checker")
			fact_checker_graph = getattr(fact_checker_module, "graph", None)
			if fact_checker_graph is None:
				raise ClaimeAIError("ClaimeAI fact checker graph is not available.")
			result = _run_async(
				fact_checker_graph.ainvoke(
					{
						"question": "Verify the factual accuracy of the following text.",
						"answer": text,
					}
				)
			)
			final_report = None
			if isinstance(result, dict):
				final_report = result.get("final_report")
			else:
				final_report = getattr(result, "final_report", None)
			if final_report is not None:
				if progress_callback:
					progress_callback("aggregating_verdict")
				return _score_from_report(final_report)
		except Exception:
			if isinstance(sys.exc_info()[1], ClaimeAIError):
				raise
			raise ClaimeAIError("Error invoking ClaimeAI fact checker.") from sys.exc_info()[1]

	if progress_callback:
		progress_callback("aggregating_verdict")
	return _offline_false_confidence(text, claims)


def _offline_false_confidence(text: str, claims: list[str] | None = None) -> float:
	claims = claims if claims is not None else extract_claims(text)
	if not claims:
		return 0.5

	# Neutral offline fallback: more extracted claims means more surface area to
	# check, but we keep this conservative.
	return max(0.0, min(1.0, 0.35 + min(len(claims) * 0.1, 0.3)))


def _score_from_report(final_report: Any) -> float:
	verified_claims = getattr(final_report, "verified_claims", None) or []
	if not verified_claims:
		return 0.5

	scores = [_map_verdict_to_false_confidence(verdict) for verdict in verified_claims]
	if not scores:
		return 0.5
	return max(0.0, min(1.0, sum(scores) / len(scores)))


def _map_verdict_to_false_confidence(verdict: Any) -> float:
	result = getattr(verdict, "result", None)
	result_value = getattr(result, "value", result)
	normalized = str(result_value).strip().lower()

	if normalized in {"refuted", "false", "contradicted"}:
		return 1.0
	if normalized in {"supported", "true"}:
		return 0.0
	if normalized in {"conflicting", "conflicting_evidence", "mixed"}:
		return 0.65
	if normalized in {"insufficient_information", "unknown", "unverifiable"}:
		return 0.5
	return 0.5


def _sentence_split(text: str) -> list[str]:
	sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
	return sentences or ([text.strip()] if text.strip() else [])


def _run_async(coro: Any) -> Any:
	try:
		asyncio.get_running_loop()
	except RuntimeError:
		return asyncio.run(coro)

	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


__all__ = ["claimeai_available", "extract_claims", "false_confidence"]

