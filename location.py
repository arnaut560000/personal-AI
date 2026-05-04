from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
from typing import Any
from urllib import parse
import webbrowser

from config import AppConfig
from utils import contains_trigger_phrase, fetch_json, normalize_transcript


GPS_CAPTURE_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RoomAI GPS Permission</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f4ec;
      --card: #fffdf8;
      --text: #1d1a16;
      --muted: #655b52;
      --accent: #0d6b4d;
      --border: #dccfbf;
    }
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: radial-gradient(circle at top, #fffdf6, var(--bg) 55%);
      color: var(--text);
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }
    .card {
      width: min(560px, 100%);
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 20px;
      box-shadow: 0 18px 50px rgba(64, 52, 35, 0.12);
      padding: 28px;
    }
    h1 {
      margin: 0 0 12px;
      font-size: 1.9rem;
    }
    p {
      line-height: 1.6;
      color: var(--muted);
    }
    .status {
      margin-top: 18px;
      padding: 14px 16px;
      border-radius: 14px;
      background: #f1ece1;
      color: var(--text);
      font-weight: 600;
    }
    button {
      margin-top: 18px;
      background: var(--accent);
      color: white;
      border: none;
      border-radius: 999px;
      padding: 12px 18px;
      font-size: 1rem;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <main class="card">
    <h1>Share your location with RoomAI</h1>
    <p>Allow browser GPS access so RoomAI can answer location questions and find nearby places more accurately.</p>
    <div class="status" id="status">Waiting to request your GPS location...</div>
    <button id="retry" type="button" hidden>Try again</button>
  </main>
  <script>
    const statusEl = document.getElementById("status");
    const retryButton = document.getElementById("retry");

    async function sendLocation(coords) {
      const response = await fetch("/api/location", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          latitude: coords.latitude,
          longitude: coords.longitude,
          accuracy: coords.accuracy
        })
      });
      if (!response.ok) {
        throw new Error("RoomAI could not save the GPS coordinates.");
      }
      statusEl.textContent = "Location shared. You can close this tab and return to RoomAI.";
      retryButton.hidden = true;
    }

    function requestLocation() {
      retryButton.hidden = true;
      statusEl.textContent = "Requesting GPS permission...";
      if (!navigator.geolocation) {
        statusEl.textContent = "This browser does not support geolocation.";
        retryButton.hidden = false;
        return;
      }
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          statusEl.textContent = "Sending your GPS coordinates to RoomAI...";
          try {
            await sendLocation(position.coords);
          } catch (error) {
            statusEl.textContent = error.message;
            retryButton.hidden = false;
          }
        },
        (error) => {
          statusEl.textContent = "Location access failed: " + error.message;
          retryButton.hidden = false;
        },
        {
          enableHighAccuracy: true,
          timeout: 20000,
          maximumAge: 0
        }
      );
    }
    retryButton.addEventListener("click", requestLocation);
    requestLocation();
  </script>
</body>
</html>
"""


def is_location_request(user_text: str, keywords: tuple[str, ...]) -> bool:
    lowered = normalize_transcript(user_text)
    return any(contains_trigger_phrase(lowered, keyword) for keyword in keywords)


def load_cached_location(config: AppConfig) -> dict[str, Any] | None:
    if not config.location_cache_path.exists():
        return None
    try:
        cached = json.loads(config.location_cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return cached if isinstance(cached, dict) else None


def save_cached_location(config: AppConfig, location: dict[str, Any]) -> None:
    config.location_cache_path.write_text(json.dumps(location, indent=2), encoding="utf-8")


def is_cached_location_fresh(config: AppConfig, location: dict[str, Any]) -> bool:
    timestamp = float(location.get("timestamp", 0))
    return (time.time() - timestamp) <= config.location_cache_ttl_seconds


def reverse_geocode_coordinates(config: AppConfig, latitude: float, longitude: float) -> dict[str, str]:
    params = parse.urlencode(
        {
            "format": "jsonv2",
            "lat": f"{latitude:.6f}",
            "lon": f"{longitude:.6f}",
            "zoom": "18",
            "addressdetails": "1",
        }
    )
    data = fetch_json(f"{config.gps_reverse_geocode_url}?{params}", timeout=15)
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


def reverse_geocode_area(config: AppConfig, latitude: float, longitude: float) -> dict[str, str]:
    params = parse.urlencode(
        {
            "format": "jsonv2",
            "lat": f"{latitude:.6f}",
            "lon": f"{longitude:.6f}",
            "zoom": "10",
            "addressdetails": "1",
        }
    )
    data = fetch_json(f"{config.gps_reverse_geocode_url}?{params}", timeout=15)
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


class GPSCaptureHandler(BaseHTTPRequestHandler):
    server: "GPSCaptureServer"

    def do_GET(self) -> None:
        if self.path not in {"/", "/index.html"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        body = GPS_CAPTURE_PAGE.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/api/location":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
            latitude = float(payload["latitude"])
            longitude = float(payload["longitude"])
            accuracy = float(payload.get("accuracy", 0))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid location payload")
            return
        self.server.set_location(latitude=latitude, longitude=longitude, accuracy=accuracy)
        response_body = json.dumps({"ok": True}).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class GPSCaptureServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int]) -> None:
        super().__init__(server_address, GPSCaptureHandler)
        self.location_event = threading.Event()
        self.received_location: dict[str, float] | None = None

    def set_location(self, latitude: float, longitude: float, accuracy: float) -> None:
        self.received_location = {"latitude": latitude, "longitude": longitude, "accuracy": accuracy}
        self.location_event.set()


def request_browser_gps_location(config: AppConfig) -> dict[str, Any]:
    server = GPSCaptureServer(("127.0.0.1", 0))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    capture_url = f"http://127.0.0.1:{server.server_port}/"

    print("Opening a browser tab to request GPS permission...")
    opened = webbrowser.open(capture_url)
    if not opened:
        print(f"Open this link in your browser to share GPS location: {capture_url}")

    if not server.location_event.wait(timeout=config.gps_request_timeout_seconds):
        server.shutdown()
        server.server_close()
        raise RuntimeError("GPS permission timed out. Please allow location access in the browser and try again.")

    server.shutdown()
    server.server_close()
    assert server.received_location is not None
    latitude = server.received_location["latitude"]
    longitude = server.received_location["longitude"]
    accuracy = server.received_location["accuracy"]

    try:
        geocoded = reverse_geocode_coordinates(config, latitude, longitude)
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
        "accuracy_meters": accuracy,
        "label": geocoded["label"],
        "area_label": geocoded["area_label"],
        "city": geocoded["city"],
        "region": geocoded["region"],
        "country": geocoded["country"],
        "display_name": geocoded["display_name"],
        "source": "gps",
        "timestamp": time.time(),
    }
    save_cached_location(config, location)
    return location


def get_live_location(
    config: AppConfig,
    force_refresh: bool = False,
    require_gps: bool = False,
) -> dict[str, Any] | None:
    fixed_location_label = os.getenv("ROOMAI_FIXED_LOCATION", "").strip()
    manual_city = os.getenv("ROOMAI_CITY", "").strip()
    manual_region = os.getenv("ROOMAI_REGION", "").strip()
    manual_country = os.getenv("ROOMAI_COUNTRY", "").strip()

    cached_location = load_cached_location(config)
    if cached_location and not force_refresh and is_cached_location_fresh(config, cached_location):
        return cached_location

    if require_gps:
        return request_browser_gps_location(config)

    if fixed_location_label:
        return {
            "city": manual_city,
            "region": manual_region,
            "country": manual_country,
            "label": fixed_location_label,
            "source": "fixed",
        }

    if manual_city:
        return {
            "city": manual_city,
            "region": manual_region,
            "country": manual_country,
            "label": ", ".join(part for part in (manual_city, manual_region, manual_country) if part),
            "source": "fixed",
        }

    try:
        data = fetch_json(config.ip_location_url, timeout=10)
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
            geocoded = reverse_geocode_area(config, latitude_value, longitude_value)
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
    save_cached_location(config, location)
    return location


def build_location_answer(location: dict[str, Any] | None) -> str:
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
            f"My current internet-detected area is {label}. "
            f"Coordinates: {latitude:.6f}, {longitude:.6f}. "
            "This is based on your laptop's internet location, so it can be off and is not an exact street address."
        )
    return (
        f"My current detected area is {label}. "
        "This is based on the laptop's internet location, so it is approximate."
    )
