"""Shared parsing and database lookup rules for the two evaluators."""
import re

from utils.func import extract_before_parenthesis, get_valid_name_city


def normalize_city(value):
    return extract_before_parenthesis(value).strip() if isinstance(value, str) else ""


def parse_attractions(value):
    """Accept lists with or without a trailing semicolon."""
    return [item.strip() for item in (value or "").split(";") if item.strip() not in ("", "-")]


def extract_from_to(text):
    if not isinstance(text, str):
        return None, None
    match = re.search(r"\bfrom\s+([^,]+?)\s+to\s+([^,]+)", text, re.IGNORECASE)
    if not match:
        return None, None
    origin, destination = (normalize_city(city) for city in match.groups())
    return (origin, destination) if origin and destination else (None, None)


def city_route(value):
    """Return the day's start/end cities, rejecting incomplete route labels."""
    if not isinstance(value, str) or value.strip() in ("", "-"):
        return None, None
    if re.search(r"\b(from|to)\b", value, re.IGNORECASE):
        if not re.fullmatch(r"\s*from\s+[^,]+?\s+to\s+[^,]+\s*", value, re.IGNORECASE):
            return None, None
        return extract_from_to(value)
    city = normalize_city(value)
    return city, city


def matching_records(data, value, name_column, city_column):
    """Prefer exact names; preserve unique partial-name lookup compatibility.

    Callers must reject unresolved/ambiguous results instead of picking a row
    based on database order. Identical duplicate rows are harmless.
    """
    name, city = get_valid_name_city(value)
    if name in ("", "-") or city in ("", "-"):
        return data.iloc[:0]
    candidates = data[data[city_column] == city]
    exact = candidates[candidates[name_column].astype(str) == name]
    if len(exact):
        return exact.drop_duplicates()
    return candidates[candidates[name_column].astype(str).str.contains(name, regex=False)].drop_duplicates()


def flight_records(data, value, origin, destination, date=None):
    match = re.search(r"\bflight\s+number\s*:\s*([^,\s]+)", value, re.IGNORECASE)
    if not match:
        return data.iloc[:0]
    result = data[(data['Flight Number'] == match.group(1)) &
                  (data['OriginCityName'] == origin) &
                  (data['DestCityName'] == destination)]
    if date is not None:
        result = result[result['FlightDate'] == date]
    return result.drop_duplicates()


def travel_date(question, index):
    dates = question.get('date')
    return dates[index] if isinstance(dates, (list, tuple)) and index < len(dates) else None
