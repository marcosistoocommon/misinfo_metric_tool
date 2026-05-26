from __future__ import annotations

import json
import os
import re
from pathlib import Path

from typing import Any
from urllib.parse import urlparse


from Scweet import Scweet



STATUS_URL_PATTERN = re.compile(
    r"^https?://(?:www\.)?(?:x\.com|twitter\.com)/(?P<username>[A-Za-z0-9_]+)/status/(?P<tweet_id>\d+)(?:[/?#].*)?$",
    re.IGNORECASE,
)
DEFAULT_TIMELINE_LIMIT = 50
DEFAULT_SOCIAL_LIMIT = 5
DEFAULT_LAST_POSTS = 3




def _load_auth_candidates() -> tuple[list[dict[str, str | None]], Path]:
    cookies_path = Path(__file__).resolve().parent / "cookies.json"

    try:
        payload = json.loads(cookies_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in cookies file: {cookies_path}") from exc

    records: list[Any]
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("accounts"), list):
        records = payload["accounts"]
    elif isinstance(payload, dict):
        records = [payload]
    else:
        raise ValueError("cookies.json must be a JSON object or an array of account objects")

    candidates: list[dict[str, str | None]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue
        cookies = record.get("cookies")
        token = record.get("auth_token")
        if not token and isinstance(cookies, dict):
            token = cookies.get("auth_token")
        if not isinstance(token, str) or not token.strip():
            continue

        proxy = record.get("proxy")
        username = record.get("username")
        candidates.append(
            {
                "auth_token": token.strip(),
                "proxy": proxy.strip() if isinstance(proxy, str) and proxy.strip() else None,
                "username": username.strip() if isinstance(username, str) and username.strip() else f"account_{index}",
            }
        )

    if not candidates:
        raise ValueError(
            f"No usable accounts found in {cookies_path}. "
            "Each account needs auth_token at top-level or in cookies.auth_token."
        )

    return candidates, cookies_path


def _extract_with_client(
    client: Scweet,
    username: str,
    tweet_id: str,
    *,
    timeline_limit: int,
    social_limit: int,
    last_posts_count: int,
) -> dict[str, Any]:
    profile_info = _pick_first(client.get_user_info([username]))
    timeline = client.get_profile_tweets([username], limit=timeline_limit)

    matched_tweet = next(
        (
            item
            for item in timeline
            if isinstance(item, dict)
            and str(item.get("tweet_id") or "") == tweet_id
        ),
        None,
    )
    if matched_tweet is None:
        raise LookupError(
            f"Could not find tweet_id={tweet_id} in the collected timeline for @{username}"
        )

    followers = client.get_followers([username], limit=social_limit)
    following = client.get_following([username], limit=social_limit)
    following_lookup = {
        _user_key(item)
        for item in following
        if isinstance(item, dict) and _user_key(item)
    }
    mutual_followers = [
        _compact_user_item(item)
        for item in followers
        if isinstance(item, dict) and _user_key(item) in following_lookup
    ][:last_posts_count]

    return {
        "tweet": _compact_tweet_item(matched_tweet),
        "profile": {
            "username": profile_info.get("username") or username,
            "description": profile_info.get("description"),
            "profile_pic": profile_info.get("profile_image_url"),
            "name": profile_info.get("name"),
        },
        "mutual_followers": mutual_followers,
        "last_posts": [
            str(item.get("text") or "").strip()
            for item in timeline[:last_posts_count]
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ],
    }




def _normalize_status_url(status_url: str) -> str:
    cleaned_url = status_url.strip()
    if not cleaned_url:
        raise ValueError("status_url cannot be empty")

    parsed = urlparse(cleaned_url)
    if not parsed.scheme:
        cleaned_url = f"https://{cleaned_url}"

    match = STATUS_URL_PATTERN.match(cleaned_url)
    if not match:
        raise ValueError(
            "status_url must look like https://x.com/<username>/status/<tweet_id>"
        )

    return cleaned_url


def _extract_status_parts(status_url: str) -> tuple[str, str]:
    match = STATUS_URL_PATTERN.match(_normalize_status_url(status_url))
    if not match:
        raise ValueError(
            "status_url must look like https://x.com/<username>/status/<tweet_id>"
        )

    return match.group("username"), match.group("tweet_id")


def _pick_first(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise LookupError("Scweet returned no profile data for the requested account")
    first_item = items[0]
    if not isinstance(first_item, dict):
        raise TypeError("Scweet returned an unexpected profile payload")
    return first_item


def _compact_user_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "username": item.get("username") or item.get("screen_name") or item.get("handle"),
        "name": item.get("name"),
        "description": item.get("description"),
        "profile_image_url": item.get("profile_image_url"),
    }


def _user_key(item: dict[str, Any]) -> str:
    return str(item.get("username") or item.get("screen_name") or item.get("handle") or "").strip().lower()


def _compact_tweet_item(item: dict[str, Any]) -> dict[str, Any]:
    user = item.get("user") if isinstance(item.get("user"), dict) else {}
    media = item.get("media") if isinstance(item.get("media"), dict) else {}
    return {
        "text": item.get("text"),
        "image_links": list(media.get("image_links") or []),
    }


def extract_x_status_context(
    status_url: str,
    *,
    timeline_limit: int = DEFAULT_TIMELINE_LIMIT,
    social_limit: int = DEFAULT_SOCIAL_LIMIT,
    last_posts_count: int = DEFAULT_LAST_POSTS,
) -> dict[str, Any]:
    """Extract author and tweet context from an X status URL using Scweet."""

    username, tweet_id = _extract_status_parts(status_url)
    candidates, cookies_path = _load_auth_candidates()
    errors: list[str] = []

    for index, candidate in enumerate(candidates, start=1):
        client_kwargs: dict[str, Any] = {"auth_token": candidate["auth_token"]}
        if candidate["proxy"]:
            client_kwargs["proxy"] = candidate["proxy"]

        try:
            client = Scweet(**client_kwargs)
            return _extract_with_client(
                client,
                username,
                tweet_id,
                timeline_limit=timeline_limit,
                social_limit=social_limit,
                last_posts_count=last_posts_count,
            )
        except Exception as exc:
            account_label = candidate.get("username") or f"account_{index}"
            errors.append(f"[{index}/{len(candidates)}] {account_label}: {exc}")

    raise RuntimeError(
        f"All accounts in {cookies_path} failed. "
        + " | ".join(errors)
    )


def main() -> None:
    status_url = input("Enter the X status URL: ")
    result = extract_x_status_context(status_url)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

