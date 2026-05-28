from __future__ import annotations

import sys
from pathlib import Path
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from tweets import extract_x_status_context
from image_eval import evaluate_political_imagery
from politic_eval import handle_politicalness
from recent_eval import evaluate_event_recency
from Claims.verification import ClaimeAIError, false_confidence, initialize_agent
from misinfo_value import patterns_and_tone_score

weights = [0.05, 0.05, 0.2, 0.3, 0.3, 0.1]


def _score_context_data(context):
    """Compute an aggregated context score from extracted X status context.

    The function evaluates profile picture politicalness, username and
    description politicalness, mutual follower signals, recent posts, and
    recency signals and combines them using fixed `weights`.

    Args:
        context: The dictionary returned by `extract_x_status_context`.

    Returns:
        A float in the 0..1 range representing the aggregated context score.
    """

    pfp_score = 0.0
    name_score = 0.0
    description_score = 0.0
    mutual_followers_score = 0.0
    last_posts_score = 0.0
    recent_tweets_score = 0.0

    profile = context["profile"]

    profile_pic = profile.get("profile_pic")
    username = profile.get("username")
    description = profile.get("description")

    if profile_pic:
        pfp_score = evaluate_political_imagery(profile_pic)
    if username:
        name_score = handle_politicalness(username)
    if description:
        description_score = handle_politicalness(description)

    for follower in context["mutual_followers"]:
        follower_profile_pic = follower.get("profile_pic")
        follower_username = follower.get("username")
        follower_description = follower.get("description")

        if follower_profile_pic:
            mutual_followers_score += evaluate_political_imagery(follower_profile_pic) * 0.25
        if follower_username:
            mutual_followers_score += handle_politicalness(follower_username) * 0.25
        if follower_description:
            mutual_followers_score += patterns_and_tone_score(follower_description)[0] * 0.5

    if context["mutual_followers"]:
        mutual_followers_score /= (len(context["mutual_followers"]) * 3)

    for post in context["last_posts"]:

        pattern_score, tone_score = patterns_and_tone_score(post)
        fakeness_score = false_confidence(post)
        last_posts_score += (pattern_score * 0.5) + (tone_score * 0.15) + (fakeness_score * 0.35)
    if context["last_posts"]:
        last_posts_score /= len(context["last_posts"])

    tweet_text = context["tweet"].get("text") if isinstance(context.get("tweet"), dict) else None
    if tweet_text:
        recent_tweets_score = evaluate_event_recency(tweet_text)

    total_score = (
        pfp_score * weights[0]
        + name_score * weights[1]
        + description_score * weights[2]
        + mutual_followers_score * weights[3]
        + last_posts_score * weights[4]
        + recent_tweets_score * weights[5]
    ) / sum(weights)

    return {
        "score": total_score,
        "details": {
            "profile_pic": pfp_score,
            "name": name_score,
            "description": description_score,
            "friends": mutual_followers_score,
            "last_posts": last_posts_score,
            "recent": recent_tweets_score,
        },
    }


def analyze_x_url(url, progress_callback=None):
    """Extract and score an X status URL.

    Args:
        url: The X status URL to analyze.
        progress_callback: Optional callback receiving progress stage names.

    Returns:
        A dict with keys: `message`, `context`, `tweet`, `profile`, `raw_context`.
        Raises on extraction errors.
    """

    if "twitter.com" in url or "x.com" in url:
        try:
            if progress_callback:
                progress_callback("extracting_context")

            context = extract_x_status_context(url)
            context_result = _score_context_data(context)
            tweet_text = context["tweet"].get("text") if isinstance(context.get("tweet"), dict) else None
            if not tweet_text:
                raise ValueError("Could not extract tweet text from the provided X URL.")

            return {
                "message": tweet_text,
                "context": context_result["score"],
                "context_details": context_result["details"],
                "tweet": context["tweet"],
                "profile": context["profile"],
                "raw_context": context,
            }
        except Exception as e:
            print(f"Error extracting X status context: {e}")
            raise e
    else:
        print("Unsupported URL type for context evaluation.")
        return "ERROR_UNSUPPORTED_URL"


def calculate_context_value(url, progress_callback=None):
    """Helper that returns only the numeric context value for a URL.

    This wraps `analyze_x_url` and returns the `context` float when
    available, or the raw result (error token) otherwise.
    """

    result = analyze_x_url(url, progress_callback=progress_callback)
    if isinstance(result, dict):
        return result["context"]
    return result


calculate_context = calculate_context_value
    

def _bootstrap_standalone() -> None:
    try:
        initialize_agent()
    except ClaimeAIError as exc:
        print(f"Startup error: {exc}")
        raise SystemExit(1) from exc
if __name__ == "__main__":
    url = input("Enter the URL for context evaluation: ")
    _bootstrap_standalone()
    score = calculate_context_value(url)
    print(f"Context Score: {score}")