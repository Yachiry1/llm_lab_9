"""Plain-Python implementations of the Developer 2 survival tools."""

import json
import math
from pathlib import Path


_ENVIRONMENTS = {"forest", "mountain", "desert", "coastal", "urban"}
_WEATHER = {
    "clear",
    "rain",
    "snow",
    "storm",
    "extreme_heat",
    "extreme_cold",
}
_DATA_FILE = Path(__file__).with_name("survival_data.json")


def _error(code: str, message: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": message}}


def _normalized_choice(value: str, allowed: set[str], name: str) -> tuple[str | None, dict | None]:
    if not isinstance(value, str) or not value.strip():
        return None, _error("invalid_arguments", f"{name} must be a nonempty string.")
    normalized = value.strip().lower().replace(" ", "_")
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        return None, _error("invalid_arguments", f"{name} must be one of: {choices}.")
    return normalized, None


def estimate_water_supply(
    water_liters: float, people: int, daily_liters_per_person: float
) -> dict:
    """Estimate how long drinking water lasts at a supplied consumption rate."""
    numeric_values = (water_liters, daily_liters_per_person)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in numeric_values):
        return _error("invalid_arguments", "Water amounts and consumption must be numbers.")
    if not all(math.isfinite(value) for value in numeric_values):
        return _error("invalid_arguments", "Water amounts and consumption must be finite.")
    if water_liters < 0:
        return _error("invalid_arguments", "water_liters must be nonnegative.")
    if daily_liters_per_person <= 0:
        return _error("invalid_arguments", "daily_liters_per_person must be positive.")
    if isinstance(people, bool) or not isinstance(people, int) or people <= 0:
        return _error("invalid_arguments", "people must be a positive integer.")

    total_daily_liters = people * daily_liters_per_person
    estimated_days = water_liters / total_daily_liters
    return {
        "ok": True,
        "data": {
            "total_daily_liters": total_daily_liters,
            "estimated_days": estimated_days,
            "estimated_hours": estimated_days * 24,
            "assumptions": [
                "Consumption stays constant at the supplied rate.",
                "Actual hydration needs vary significantly with activity, temperature, and health conditions; do not restrict water to an unsafe level.",
            ],
        },
    }


def create_survival_plan(
    environment: str,
    weather: str,
    equipment: list[str],
    people: int,
    duration_hours: float,
) -> dict:
    """Build a small, deterministic plan from the documented survival context."""
    environment, error = _normalized_choice(environment, _ENVIRONMENTS, "environment")
    if error:
        return error
    weather, error = _normalized_choice(weather, _WEATHER, "weather")
    if error:
        return error
    if not isinstance(equipment, list) or any(
        not isinstance(item, str) or not item.strip() for item in equipment
    ):
        return _error("invalid_arguments", "equipment must be a list of nonempty strings.")
    if isinstance(people, bool) or not isinstance(people, int) or people <= 0:
        return _error("invalid_arguments", "people must be a positive integer.")
    if (
        isinstance(duration_hours, bool)
        or not isinstance(duration_hours, (int, float))
        or not math.isfinite(duration_hours)
        or duration_hours <= 0
    ):
        return _error("invalid_arguments", "duration_hours must be a positive finite number.")

    resources = {item.strip().lower() for item in equipment}
    steps = [
        {
            "action": "Check everyone for immediate danger or injury and contact emergency services when possible.",
            "reason": "Life-threatening hazards and medical emergencies take priority.",
        },
        {
            "action": "Move to the safest nearby location and protect the group from the weather.",
            "reason": f"Exposure in {weather.replace('_', ' ')} conditions can worsen quickly.",
        },
        {
            "action": "Inventory and ration resources without restricting essential drinking water.",
            "reason": f"The plan must support {people} people for about {duration_hours} hours.",
        },
        {
            "action": "Establish a reliable way to signal rescuers and share the route or location.",
            "reason": "Being locatable generally reduces time exposed to hazards.",
        },
    ]
    warnings = ["Do not take unnecessary travel risks; stay together and reassess conditions."]

    if weather in {"snow", "extreme_cold"}:
        warnings.append("Prevent heat loss: stay dry, insulate from the ground, and warm people gradually.")
    elif weather == "extreme_heat" or environment == "desert":
        warnings.append("Limit exertion during peak heat and monitor everyone for heat illness or dehydration.")
    elif weather in {"rain", "storm"}:
        warnings.append("Avoid floodwater, exposed high ground, isolated trees, and unstable shelter sites.")
    if not resources.intersection({"phone", "radio", "satellite phone", "personal locator beacon", "plb"}):
        warnings.append("No communication device was identified; use visible or audible signals without creating new hazards.")

    numbered_steps = [
        {"priority": priority, **step} for priority, step in enumerate(steps, start=1)
    ]
    return {
        "ok": True,
        "data": {
            "summary": f"Prioritized {environment} survival plan for {people} people in {weather.replace('_', ' ')} weather.",
            "steps": numbered_steps,
            "warnings": warnings,
        },
    }


def get_first_aid_guide(topic: str) -> dict:
    """Return a conservative first-aid guide from the static knowledge file."""
    if not isinstance(topic, str) or not topic.strip():
        return _error("invalid_arguments", "topic must be a nonempty string.")

    normalized = topic.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "minor_cut": "bleeding",
        "minor_cuts": "bleeding",
        "minor_burn": "burns",
        "minor_burns": "burns",
        "suspected_hypothermia": "hypothermia",
        "suspected_dehydration": "dehydration",
    }
    normalized = aliases.get(normalized, normalized)

    try:
        with _DATA_FILE.open(encoding="utf-8") as data_file:
            guides = json.load(data_file).get("first_aid_guides", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return _error("data_unavailable", "First-aid guide data is unavailable.")

    guide = guides.get(normalized)
    if guide is None:
        return _error("unsupported_topic", f"No first-aid guide is available for '{topic.strip()}'.")
    return {"ok": True, "data": {"topic": normalized, **guide}}
