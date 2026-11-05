"""Six deterministic tools; see README.md for their shared contracts."""

import json
import math
from pathlib import Path

ENVIRONMENTS = {"forest", "mountain", "desert", "coastal", "urban"}
WEATHER = {"clear", "rain", "snow", "storm", "extreme_heat", "extreme_cold"}
DATA_PATH = Path(__file__).resolve().with_name("survival_data.json")

# Category: (accepted item names, possible use).
EQUIPMENT = {
    "water": (["water", "drinking water"], "Drink available safe water."),
    "water treatment": (["water filter", "purification tablets"], "Treat collected water according to product instructions."),
    "shelter": (["tent", "tarp", "bivy", "emergency blanket"], "Provide temporary protection from exposure."),
    "navigation": (["map", "compass", "gps"], "Help establish location and direction."),
    "light": (["flashlight", "headlamp"], "Illuminate surroundings or signal for help."),
    "first aid": (["first aid kit", "first-aid kit"], "Provide basic first-aid supplies."),
    "signaling": (["whistle", "signal mirror"], "Help rescuers locate you."),
    "warm clothing": (["warm jacket", "warm clothing"], "Add insulation in cold conditions."),
    "rain protection": (["raincoat", "rain jacket", "poncho"], "Help keep clothing dry."),
    "sun protection": (["sun hat", "sunscreen"], "Reduce sun exposure."),
    "food": (["food", "snacks"], "Supply energy while waiting for help."),
    "repair tools": (["knife", "multi-tool", "duct tape"], "Help repair equipment."),
}
BASE_ESSENTIALS = ["water", "shelter", "navigation", "light", "first aid", "signaling"]
EXTRA_ESSENTIALS = {
    "forest": ["rain protection", "water treatment"],
    "mountain": ["warm clothing", "rain protection"],
    "desert": ["sun protection"],
    "coastal": ["rain protection", "sun protection"],
    "urban": [],
}
TOPIC_ALIASES = {
    "lost in the forest": "lost_in_forest", "lost in forest": "lost_in_forest",
    "cold weather": "cold_weather", "water shortage": "water",
    "water_shortage": "water", "emergency shelter": "shelter",
    "emergency_shelter": "shelter",
}
GUIDE_TOPICS = ("lost_in_forest", "cold_weather", "water", "shelter", "fire", "navigation", "signaling")


def _normalize(value):
    return value.strip().lower() if isinstance(value, str) else ""


def _string_list(value):
    return isinstance(value, list) and all(
        isinstance(item, str) and item.strip() for item in value
    )


def _error(code, message):
    return {"ok": False, "error": {"code": code, "message": message}}


def assess_risk(environment=None, weather=None, time_of_day=None, hazards=None,
                temperature_c=None, has_shelter=None) -> dict:
    """Return an educational heuristic, not a professional safety assessment."""
    environment, weather = _normalize(environment), _normalize(weather)
    time_of_day = _normalize(time_of_day)
    if environment not in ENVIRONMENTS or weather not in WEATHER:
        return _error("invalid_arguments", "Provide a supported environment and weather.")
    if time_of_day not in {"day", "night"} or not _string_list(hazards):
        return _error("invalid_arguments", "Provide day/night and a list of hazard strings.")
    if temperature_c is not None:
        if isinstance(temperature_c, bool) or not isinstance(temperature_c, (int, float)):
            return _error("invalid_arguments", "temperature_c must be a finite number or null.")
        try:
            finite = math.isfinite(temperature_c)
        except OverflowError:
            finite = False
        if not finite:
            return _error("invalid_arguments", "temperature_c must be a finite number or null.")
    if has_shelter is not None and not isinstance(has_shelter, bool):
        return _error("invalid_arguments", "has_shelter must be true, false, or null.")

    factors, recommendations = [], []
    severity = 0

    def add(level, factor, recommendation):
        nonlocal severity
        severity = max(severity, level)
        factors.append(factor)
        recommendations.append(recommendation)

    if weather in {"storm", "extreme_heat", "extreme_cold"}:
        add(2, "Severe weather reported.", "Seek suitable protection from the weather.")
    elif weather in {"rain", "snow"}:
        add(1, "Wet or snowy conditions.", "Keep clothing and equipment dry.")
    if temperature_c is not None:
        if temperature_c <= 0:
            add(2, "Freezing temperature reported.", "Prioritize warmth and shelter.")
        elif temperature_c < 10:
            add(1, "Cool temperature reported.", "Use dry insulating layers.")
        elif temperature_c >= 35:
            add(2, "Very hot temperature reported.", "Reduce exertion and seek shade.")
        elif temperature_c >= 30:
            add(1, "Hot temperature reported.", "Avoid unnecessary exertion.")
    if has_shelter is False:
        exposed = weather != "clear" or (
            temperature_c is not None and (temperature_c < 10 or temperature_c >= 30)
        )
        add(2 if exposed else 1, "No shelter available.", "Prioritize suitable shelter.")
    if time_of_day == "night":
        add(1, "Reduced visibility at night.", "Avoid unnecessary travel in darkness.")
    if environment in {"mountain", "desert", "coastal"}:
        add(1, f"Potential exposure in a {environment} environment.", "Check local terrain before moving.")
    for hazard in dict.fromkeys(_normalize(item) for item in hazards):
        add(2, f"Reported hazard: {hazard}.", "Avoid the reported hazard and seek help if needed.")
    level = ["low", "medium", "high"][severity]
    explanation = (
        f"The highest matching rule gives a {level} risk estimate. "
        + (" ".join(factors) if factors else "No elevated-risk rule matched the supplied context.")
        + " This is an educational heuristic, not a professional safety assessment; "
        "a low result does not establish safety."
    )
    if temperature_c is None or has_shelter is None:
        explanation += " Temperature or shelter information is incomplete."
    return {"ok": True, "data": {
        "risk_level": level, "risk_factors": factors,
        "recommendations": list(dict.fromkeys(recommendations)), "explanation": explanation,
    }}


def check_equipment(environment=None, equipment=None) -> dict:
    """Compare equipment against a small environment-specific checklist."""
    environment = _normalize(environment)
    if environment not in ENVIRONMENTS or not _string_list(equipment):
        return _error("invalid_arguments", "Provide a supported environment and a list of equipment strings.")
    available = list(dict.fromkeys(_normalize(item) for item in equipment))
    useful, present = [], set()
    for item in available:
        for category, (names, use) in EQUIPMENT.items():
            if item in names:
                present.add(category)
                useful.append({"item": item, "possible_uses": [use]})
                break
    missing = [name for name in BASE_ESSENTIALS + EXTRA_ESSENTIALS[environment] if name not in present]
    return {"ok": True, "data": {
        "available": available, "useful_available": useful, "missing_essential": missing,
        "recommendations": [f"Add equipment for: {name}." for name in missing],
    }}


def get_survival_guide(topic=None) -> dict:
    """Load a known topic from local JSON or return a clear fallback error."""
    topic = _normalize(topic)
    if not topic:
        return _error("invalid_arguments", "Provide a nonempty survival topic string.")
    topic = TOPIC_ALIASES.get(topic, topic)
    if topic not in GUIDE_TOPICS:
        return _error("unsupported_topic", "No guide found. Supported topics: " + ", ".join(GUIDE_TOPICS) + ".")
    try:
        knowledge = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return _error("data_unavailable", "The survival knowledge file could not be read.")
    if not isinstance(knowledge, dict):
        return _error("data_unavailable", "The survival knowledge file must contain an object.")
    guides = knowledge.get("survival_guides")
    guide = guides.get(topic) if isinstance(guides, dict) else None
    if (not isinstance(guide, dict) or guide.get("topic") != topic
            or not isinstance(guide.get("title"), str) or not guide["title"].strip()
            or not _string_list(guide.get("steps")) or not guide["steps"]
            or not _string_list(guide.get("warnings"))):
        return _error("data_unavailable", f"The guide for {topic} is missing or invalid.")
    return {"ok": True, "data": {
        key: guide[key] for key in ("topic", "title", "steps", "warnings")
    }}


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
    environment, error = _normalized_choice(environment, ENVIRONMENTS, "environment")
    if error:
        return error
    weather, error = _normalized_choice(weather, WEATHER, "weather")
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
        with DATA_PATH.open(encoding="utf-8") as data_file:
            guides = json.load(data_file).get("first_aid_guides", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return _error("data_unavailable", "First-aid guide data is unavailable.")

    guide = guides.get(normalized)
    if guide is None:
        return _error("unsupported_topic", f"No first-aid guide is available for '{topic.strip()}'.")
    return {"ok": True, "data": {"topic": normalized, **guide}}
