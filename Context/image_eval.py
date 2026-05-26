"""Evaluate whether an image contains political imagery using OpenAI."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_MODEL = "gpt-4o-mini"

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _require_api_key(value: str | None, name: str) -> str:
	if not value:
		raise ValueError(f"Missing required API key: {name}")
	return value


def _validate_image_url(image_url: str) -> str:
	cleaned_url = image_url.strip()
	if not cleaned_url:
		raise ValueError("image_url cannot be empty")

	parsed = urlparse(cleaned_url)
	if parsed.scheme not in {"http", "https"} or not parsed.netloc:
		raise ValueError("image_url must be a valid http or https URL")

	return cleaned_url


def _parse_json_object(text: str) -> dict[str, Any]:
	try:
		return json.loads(text)
	except json.JSONDecodeError:
		match = re.search(r"\{.*\}", text, flags=re.DOTALL)
		if not match:
			raise ValueError("OpenAI response did not contain valid JSON") from None
		return json.loads(match.group(0))


def evaluate_political_imagery(image_url: str) -> int:
	"""Return 1 if the image contains political imagery, otherwise 0.

	The function expects a publicly accessible image URL and uses OpenAI's vision
	capabilities to classify it conservatively.
	"""

	validated_image_url = _validate_image_url(image_url)
	openai_api_key = _require_api_key(os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY")
	client = OpenAI(api_key=openai_api_key)

	prompt = (
		"You inspect images and decide whether they contain political imagery. "
		"Political imagery includes politicians, political symbols, campaign signs, "
		"party logos, protest placards, flags used in a political context, election "
		"materials, government officials, political rallies, or other clearly political "
		"visuals. Return only JSON with one key: has_political_imagery (0 or 1). "
		"Use 1 only when the image clearly contains political imagery. If the evidence "
		"is weak or ambiguous, return 0."
	)

	response = client.chat.completions.create(
		model=DEFAULT_MODEL,
		temperature=0,
		response_format={"type": "json_object"},
		messages=[
			{"role": "system", "content": prompt},
			{
				"role": "user",
				"content": [
					{
						"type": "text",
						"text": "Classify this image as political imagery or not.",
					},
					{
						"type": "image_url",
						"image_url": {"url": validated_image_url},
					},
				],
			},
		],
	)

	content = response.choices[0].message.content or ""
	payload = _parse_json_object(content)
	decision = payload.get("has_political_imagery")

	if isinstance(decision, bool):
		return int(decision)
	if decision in {0, 1}:
		return int(decision)

	raise ValueError("OpenAI response did not contain a valid has_political_imagery value")


def main() -> None:
	image_url = input("Enter the image URL: ")
	print(evaluate_political_imagery(image_url))


if __name__ == "__main__":
	main()
