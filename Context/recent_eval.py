"""Evaluate whether an event is recent using Tavily search and OpenAI."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from typing import Any

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_RECENT_DAYS = 7
DEFAULT_MAX_RESULTS = 2

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _require_api_key(value: str | None, name: str) -> str:
	if not value:
		raise ValueError(f"Missing required API key: {name}")
	return value


def _build_search_query(event_description: str) -> str:
	cleaned = " ".join(event_description.split())
	return f'Is this event recent? {cleaned}'


def _extract_search_context(search_result: dict[str, Any]) -> str:
	lines: list[str] = []

	if isinstance(search_result.get("answer"), str) and search_result["answer"].strip():
		lines.append(f"Answer: {search_result['answer'].strip()}")

	results = search_result.get("results", [])
	if isinstance(results, list):
		for index, item in enumerate(results[:DEFAULT_MAX_RESULTS], start=1):
			if not isinstance(item, dict):
				continue

			title = str(item.get("title") or "").strip()
			url = str(item.get("url") or "").strip()
			published_date = str(item.get("published_date") or item.get("date") or "").strip()
			content = str(
				item.get("raw_content")
				or item.get("content")
				or item.get("snippet")
				or ""
			).strip()

			result_line = f"Result {index}: {title}"
			if published_date:
				result_line += f" | date: {published_date}"
			if url:
				result_line += f" | url: {url}"
			lines.append(result_line)

			if content:
				lines.append(f"Snippet: {content[:1200]}")

	return "\n".join(lines).strip()


def _parse_json_object(text: str) -> dict[str, Any]:
	try:
		return json.loads(text)
	except json.JSONDecodeError:
		match = re.search(r"\{.*\}", text, flags=re.DOTALL)
		if not match:
			raise ValueError("OpenAI response did not contain valid JSON") from None
		return json.loads(match.group(0))


def evaluate_event_recency(event_description: str,) -> dict[str, Any]:
	"""Evaluate whether an event is recent.

	Returns a dictionary with the boolean decision, confidence, reasoning, and the
	web evidence used to make the call.
	"""

	if not event_description or not event_description.strip():
		raise ValueError("event_description cannot be empty")

	openai_api_key = _require_api_key(os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY")
	tavily_api_key = _require_api_key(os.environ.get("TAVILY_API_KEY"), "TAVILY_API_KEY")

	client = OpenAI(api_key=openai_api_key)
	tavily_client = TavilyClient(api_key=tavily_api_key)

	search_query = _build_search_query(event_description)
	search_result = tavily_client.search(
		query=search_query,
		topic="news",
		search_depth="advanced",
		max_results=DEFAULT_MAX_RESULTS,
		include_answer=True,
		include_raw_content="markdown",
	)

	web_context = _extract_search_context(search_result)
	current_date = datetime.now(UTC).date().isoformat()

	prompt = (
		"You determine whether a described event is recent relative to today's date. "
		"Use the web evidence carefully and be conservative when evidence is weak. "
		"If the event happened within the last 30 days, or is clearly being reported as current, mark it recent. "
		"If the event is historical or older than 30 days, mark it not recent. "
		"If the evidence does not support a confident judgment, mark it not recent and lower the confidence. "
		"Return only JSON with these keys: is_recent (boolean), confidence (number 0 to 1), "
		"event_date (string or null), evidence_summary (string), reasoning (string), "
		"signals (array of strings)."
	)

	response = client.chat.completions.create(
		model=DEFAULT_MODEL,
		temperature=0,
		response_format={"type": "json_object"},
		messages=[
			{"role": "system", "content": prompt},
			{
				"role": "user",
				"content": (
					f"Current date: {current_date}\n"
					f"Recentness window: {DEFAULT_RECENT_DAYS} days\n\n"
					f"Event description:\n{event_description.strip()}\n\n"
					f"Web evidence:\n{web_context or 'No evidence returned by Tavily.'}"
				),
			},
		],
	)

	content = response.choices[0].message.content or ""
	payload = _parse_json_object(content)

	is_recent = bool(payload.get("is_recent"))

	if is_recent==True:
		return 1
	else: return 0


def main() -> None:
	event_description = input("Enter the event description: ")
	print(evaluate_event_recency(event_description))


if __name__ == "__main__":
	main()
