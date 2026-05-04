from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib import parse

from app.config import AppConfig
from app.utils import contains_trigger_phrase, fetch_json, normalize_transcript


class LocationService:
    def __init__(self, config: AppConfig):
        self.config = config

    def is_location_request(self, user_text: str, keywords: tuple[str, ...]) -> bool:
        lowered = normalize_transcript(user_text)
        return any(contains_trigger_phrase(lowered, keyword) for keyword in keywords)

    def load_cached_location(self) -> dict[str, Any] | None:
        if not self.config.location_cache_path.exists():
            return None
        try:
            cached = json.loads(self.config.location_cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return cached if isinstance(cached, dict) else None

    def save_cached_location(self, location: dict[str, Any]) -> None:
        self.config.location_cache_path.write_text(json.dumps(location, indent=2), encoding="utf-8")

    def is_cached_location_fresh(self, location: dict[str, Any]) -> bool:
        timestamp = float(location.get("timestamp", 0))
        return (time.time() - timestamp) <= self.config.location_cache_ttl_seconds

    def reverse_geocode_coordinates(self, latitude: float, longitude: float) -> dict[str, str]:
        params = parse.urlencode(
            {
                "format": "jsonv2",
                "lat": f"{latitude:.6f}",
                "lon": f"{longitude:.6f}",
                "zoom": "18",
                "addressdetails": "1",
            }
        )
        data = fetch_json(f"{self.config.gps_reverse_geocode_url}?{params}", timeout=15)
        address = data.get("address", {}) if isinstance(data.get("address"), dict) else {}

        road = str(address.get("road", "")).strip()
        suburb = str(address.get("suburb", "") or address.get("neighbourhood", "")).strip()
        city = str(
            address.get("city", "")
            or address.get("town", "")
            or address.get("municipality", "")
            or address.get("village", "")
        ).strip()
        region = str(address.get("state", "") or address.get("region", "")).strip()
        country = str(address.get("country", "")).strip()
        display_name = str(data.get("display_name", "")).strip()

        label_parts = [part for part in (road, suburb, city, region, country) if part]
        label = ", ".join(dict.fromkeys(label_parts)) if label_parts else display_name
        area_parts = [part for part in (city, region, country) if part]
        area_label = ", ".join(dict.fromkeys(area_parts)) if area_parts else label
        return {
            "label": label,
            "area_label": area_label,
            "city": city,
            "region": region,
            "country": country,
            "display_name": display_name or label,
        }

    def reverse_geocode_area(self, latitude: float, longitude: float) -> dict[str, str]:
        params = parse.urlencode(
            {
                "format": "jsonv2",
                "lat": f"{latitude:.6f}",
                "lon": f"{longitude:.6f}",
                "zoom": "10",
                "addressdetails": "1",
            }
        )
        data = fetch_json(f"{self.config.gps_reverse_geocode_url}?{params}", timeout=15)
        address = data.get("address", {}) if isinstance(data.get("address"), dict) else {}
        city = str(
            address.get("city", "")
            or address.get("town", "")
            or address.get("municipality", "")
            or address.get("county", "")
            or address.get("village", "")
        ).strip()
        region = str(address.get("state", "") or address.get("region", "")).strip()
        country = str(address.get("country", "")).strip()
        area_parts = [part for part in (city, region, country) if part]
        area_label = ", ".join(dict.fromkeys(area_parts))
        return {
            "label": area_label,
            "area_label": area_label,
            "city": city,
            "region": region,
            "country": country,
            "display_name": str(data.get("display_name", "")).strip() or area_label,
        }

    def save_client_location(
        self,
        latitude: float,
        longitude: float,
        accuracy_meters: float | None = None,
    ) -> dict[str, Any]:
        try:
            geocoded = self.reverse_geocode_coordinates(latitude, longitude)
        except RuntimeError:
            geocoded = {
                "label": f"{latitude:.6f}, {longitude:.6f}",
                "area_label": f"{latitude:.6f}, {longitude:.6f}",
                "city": "",
                "region": "",
                "country": "",
                "display_name": "",
            }

        location = {
            "latitude": latitude,
            "longitude": longitude,
            "accuracy_meters": accuracy_meters,
            "label": geocoded["label"],
            "area_label": geocoded["area_label"],
            "city": geocoded["city"],
            "region": geocoded["region"],
            "country": geocoded["country"],
            "display_name": geocoded["display_name"],
            "source": "client_gps",
            "timestamp": time.time(),
        }
        self.save_cached_location(location)
        return location

    def get_live_location(
        self,
        force_refresh: bool = False,
        require_gps: bool = False,
    ) -> dict[str, Any] | None:
        fixed_location_label = os.getenv("ROOMAI_FIXED_LOCATION", "").strip()
        manual_city = os.getenv("ROOMAI_CITY", "").strip()
        manual_region = os.getenv("ROOMAI_REGION", "").strip()
        manual_country = os.getenv("ROOMAI_COUNTRY", "").strip()

        cached_location = self.load_cached_location()
        if cached_location and not force_refresh and self.is_cached_location_fresh(cached_location):
            return cached_location

        if require_gps:
            raise RuntimeError("GPS coordinates are required. Submit them through the location API first.")

        if fixed_location_label:
            return {
                "city": manual_city,
                "region": manual_region,
                "country": manual_country,
                "label": fixed_location_label,
                "area_label": fixed_location_label,
                "source": "fixed",
            }

        if manual_city:
            label = ", ".join(part for part in (manual_city, manual_region, manual_country) if part)
            return {
                "city": manual_city,
                "region": manual_region,
                "country": manual_country,
                "label": label,
                "area_label": label,
                "source": "fixed",
            }

        try:
            data = fetch_json(self.config.ip_location_url, timeout=10)
        except RuntimeError:
            return None

        city = str(data.get("city", "")).strip()
        region = str(data.get("region", "") or data.get("region_code", "")).strip()
        country = str(data.get("country_name", "") or data.get("country", "")).strip()
        latitude = data.get("latitude")
        longitude = data.get("longitude")
        if latitude in {None, ""} or longitude in {None, ""}:
            latitude = data.get("lat")
            longitude = data.get("lon")

        try:
            latitude_value = float(latitude) if latitude not in {None, ""} else None
            longitude_value = float(longitude) if longitude not in {None, ""} else None
        except (TypeError, ValueError):
            latitude_value = None
            longitude_value = None

        label = ", ".join(part for part in (city, region, country) if part)
        area_label = label
        if latitude_value is not None and longitude_value is not None:
            try:
                geocoded = self.reverse_geocode_area(latitude_value, longitude_value)
                label = geocoded["label"] or label
                area_label = geocoded["area_label"] or area_label
                city = geocoded["city"] or city
                region = geocoded["region"] or region
                country = geocoded["country"] or country
            except RuntimeError:
                pass

        if not city and not label:
            return None

        location = {
            "city": city,
            "region": region,
            "country": country,
            "label": label,
            "area_label": area_label,
            "latitude": latitude_value,
            "longitude": longitude_value,
            "source": "internet",
            "timestamp": time.time(),
        }
        self.save_cached_location(location)
        return location

    def build_location_answer(self, location: dict[str, Any] | None) -> str:
        if not location:
            return (
                "I could not detect your internet-based location right now. "
                "Please check the internet connection and try again."
            )
        label = location["label"]
        if location.get("source") == "fixed":
            return f"My configured location is {label}."
        if location.get("latitude") is not None and location.get("longitude") is not None:
            latitude = float(location["latitude"])
            longitude = float(location["longitude"])
            return (
                f"My current detected area is {label}. "
                f"Coordinates: {latitude:.6f}, {longitude:.6f}. "
                "This is approximate unless client GPS was provided."
            )
        return f"My current detected area is {label}. This is approximate."
