from __future__ import annotations

import json
from typing import Any, Callable
from urllib import parse

from config import AppConfig, WEB_SYSTEM_PROMPT
from ollama_client import OllamaClient
from utils import contains_trigger_phrase, fetch_json, fetch_json_post, normalize_transcript


def is_restaurant_request(user_text: str, keywords: tuple[str, ...]) -> bool:
    lowered = normalize_transcript(user_text)
    return any(contains_trigger_phrase(lowered, keyword) for keyword in keywords)


def fetch_web_context(config: AppConfig, query_text: str, max_notes: int = 5) -> str:
    params = parse.urlencode(
        {
            "q": query_text,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
    )

    try:
        data = fetch_json(f"{config.web_search_url}?{params}", timeout=15)
    except RuntimeError as exc:
        raise RuntimeError("Web search failed. Please check your internet connection and try again.") from exc

    notes: list[str] = []
    abstract = data.get("AbstractText", "").strip()
    abstract_url = data.get("AbstractURL", "").strip()
    if abstract:
        notes.append(f"Source: {abstract_url or 'DuckDuckGo'}\nNote: {abstract}")

    for result in data.get("Results", []):
        text = result.get("Text", "").strip()
        url = result.get("FirstURL", "").strip()
        if text and url:
            notes.append(f"Source: {url}\nNote: {text}")

    for topic in data.get("RelatedTopics", []):
        topic_list = topic.get("Topics", []) if isinstance(topic, dict) else []
        if topic_list:
            for nested_topic in topic_list:
                text = nested_topic.get("Text", "").strip()
                url = nested_topic.get("FirstURL", "").strip()
                if text and url:
                    notes.append(f"Source: {url}\nNote: {text}")
        elif isinstance(topic, dict):
            text = topic.get("Text", "").strip()
            url = topic.get("FirstURL", "").strip()
            if text and url:
                notes.append(f"Source: {url}\nNote: {text}")

    unique_notes: list[str] = []
    seen_notes: set[str] = set()
    for note in notes:
        if note in seen_notes:
            continue
        seen_notes.add(note)
        unique_notes.append(note)
        if len(unique_notes) == max_notes:
            break

    if not unique_notes:
        raise RuntimeError("Web search did not return useful results for that question. Try a more specific query.")
    return "\n\n".join(unique_notes)


def ask_with_web_notes(
    config: AppConfig,
    client: OllamaClient,
    query_text: str,
    history_messages: list[dict[str, str]] | None = None,
    on_chunk: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    web_context = fetch_web_context(config, query_text)
    answer = client.ask_with_messages(
        messages=[
            {"role": "system", "content": WEB_SYSTEM_PROMPT},
            *(history_messages or []),
            {
                "role": "user",
                "content": (
                    f"Question: {query_text}\n\n"
                    f"Web search notes:\n{web_context}\n\n"
                    "Answer the question using the notes above. Mention the sources briefly."
                ),
            },
        ],
        on_chunk=on_chunk,
    )
    return (answer, web_context)


def build_restaurant_query(user_text: str, location: dict[str, str] | None) -> str:
    location_label = str(location.get("area_label") or location.get("label")) if location else "my current area"
    return (
        f"best restaurants in {location_label}. "
        f"User request: {user_text}. "
        "Include the restaurant name, cuisine or signature dishes, and why it is recommended."
    )


def build_restaurant_prompt(user_text: str, web_context: str, location: dict[str, str] | None) -> str:
    location_label = str(location.get("area_label") or location.get("label")) if location else "the user's current area"
    coordinates_line = ""
    if location and location.get("latitude") is not None and location.get("longitude") is not None:
        coordinates_line = (
            f"GPS coordinates: {float(location['latitude']):.6f}, "
            f"{float(location['longitude']):.6f}\n\n"
        )
    return (
        f"Question: {user_text}\n\n"
        f"Detected location: {location_label}\n\n"
        f"{coordinates_line}"
        f"Web search notes:\n{web_context}\n\n"
        "Answer with exactly 5 restaurant suggestions near that location when the notes support it. "
        "For each restaurant, mention what they serve or their cuisine. "
        "If the notes are incomplete, say that briefly and give the best supported suggestions."
    )


def fetch_nearby_restaurants(config: AppConfig, location: dict[str, Any], limit: int = 5) -> list[dict[str, str]]:
    latitude = location.get("latitude")
    longitude = location.get("longitude")
    if latitude is None or longitude is None:
        raise RuntimeError("Internet location did not include usable coordinates for nearby restaurant search.")

    overpass_query = f"""
[out:json][timeout:20];
(
  node["amenity"="restaurant"](around:5000,{float(latitude):.6f},{float(longitude):.6f});
  way["amenity"="restaurant"](around:5000,{float(latitude):.6f},{float(longitude):.6f});
  relation["amenity"="restaurant"](around:5000,{float(latitude):.6f},{float(longitude):.6f});
);
out center tags;
"""
    endpoints = [config.overpass_url, config.overpass_fallback_url]
    elements: list[Any] = []
    last_error: RuntimeError | None = None
    form_payload = parse.urlencode({"data": overpass_query})

    for endpoint in endpoints:
        try:
            data = fetch_json_post(endpoint, payload=form_payload, timeout=25)
            raw_elements = data.get("elements", [])
            if isinstance(raw_elements, list) and raw_elements:
                elements = raw_elements
                break
        except RuntimeError as exc:
            last_error = exc

    if not elements:
        if last_error is not None:
            raise RuntimeError("Nearby restaurant services are temporarily unavailable.") from last_error
        raise RuntimeError("I found your area, but I could not find nearby restaurants online.")

    restaurants: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for element in elements:
        if not isinstance(element, dict):
            continue
        tags = element.get("tags", {})
        if not isinstance(tags, dict):
            continue
        name = str(tags.get("name", "")).strip()
        if not name:
            continue
        name_key = name.lower()
        if name_key in seen_names:
            continue
        seen_names.add(name_key)

        cuisine = str(tags.get("cuisine", "")).replace(";", ", ").strip()
        takeaway = str(tags.get("takeaway", "")).strip()
        description = str(tags.get("description", "")).strip()
        website = str(tags.get("website", "")).strip()
        opening_hours = str(tags.get("opening_hours", "")).strip()

        service_bits = [bit for bit in (cuisine, description) if bit]
        if takeaway == "yes":
            service_bits.append("takeaway available")
        serves = ", ".join(service_bits) if service_bits else "general restaurant meals"

        restaurants.append(
            {"name": name, "serves": serves, "website": website, "hours": opening_hours}
        )
        if len(restaurants) >= limit:
            break

    if not restaurants:
        raise RuntimeError("I found your area, but the nearby restaurant data was too limited to recommend places.")
    return restaurants


def build_restaurant_answer(restaurants: list[dict[str, str]], location: dict[str, Any]) -> str:
    area_label = str(location.get("area_label") or location.get("label") or "your area")
    lines = [f"Here are 5 restaurant suggestions near {area_label}:"]
    for index, restaurant in enumerate(restaurants[:5], start=1):
        detail = restaurant["serves"]
        extras: list[str] = []
        if restaurant.get("hours"):
            extras.append(f"hours: {restaurant['hours']}")
        if restaurant.get("website"):
            extras.append(f"site: {restaurant['website']}")
        extra_text = f" ({'; '.join(extras)})" if extras else ""
        lines.append(f"{index}. {restaurant['name']} - Serves {detail}{extra_text}.")
    lines.append("These suggestions are based on internet location and nearby online map data, so they are approximate.")
    return "\n".join(lines)


def fetch_restaurants_from_web_search(
    config: AppConfig,
    client: OllamaClient,
    user_text: str,
    location: dict[str, Any],
    history_messages: list[dict[str, str]] | None = None,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    search_query = build_restaurant_query(user_text, location)
    web_context = fetch_web_context(config, search_query, max_notes=8)
    return client.ask_with_messages(
        messages=[
            {"role": "system", "content": WEB_SYSTEM_PROMPT},
            *(history_messages or []),
            {"role": "user", "content": build_restaurant_prompt(user_text, web_context, location)},
        ],
        on_chunk=on_chunk,
    )


def serialize_restaurant_result(restaurants: list[dict[str, str]]) -> str:
    return json.dumps(restaurants)
