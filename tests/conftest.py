"""Small in-memory databases keep evaluator tests offline and deterministic."""
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def evaluators():
    # utils.func imports the UI package, but these tests never call its UI helpers.
    with patch.dict(sys.modules, {"gradio": ModuleType("gradio")}):
        classes = [
            ("tools.flights.apis", "Flights"),
            ("tools.accommodations.apis", "Accommodations"),
            ("tools.restaurants.apis", "Restaurants"),
            ("tools.googleDistanceMatrix.apis", "GoogleDistanceMatrix"),
            ("tools.attractions.apis", "Attractions"),
        ]
        from contextlib import ExitStack
        cwd = os.getcwd()
        try:
            with ExitStack() as stack:
                for module, name in classes:
                    cls = getattr(importlib.import_module(module), name)
                    stack.enter_context(patch.object(cls, "__init__", return_value=None))
                stack.enter_context(patch("builtins.open", mock_open(read_data="Home\tOrigin\nAway\tState\nOther\tState\nElsewhere\tOtherState")))
                common = importlib.import_module("evaluation.commonsense_constraint")
                hard = importlib.import_module("evaluation.hard_constraint")
        finally:
            os.chdir(cwd)
    return common, hard


@pytest.fixture
def db(evaluators, monkeypatch):
    common, hard = evaluators
    hotels = pd.DataFrame([
        {"NAME": "Grand Hotel Annex", "city": "Away", "minimum nights": 1, "price": 10, "maximum occupancy": 10},
        {"NAME": "Grand Hotel", "city": "Away", "minimum nights": 2, "price": 100, "maximum occupancy": 2},
        {"NAME": "Other Hotel", "city": "Away", "minimum nights": 1, "price": 60, "maximum occupancy": 3},
    ])
    restaurants = pd.DataFrame([
        {"Name": "Cafe Annex", "City": "Away", "Average Cost": 1},
        {"Name": "Cafe", "City": "Away", "Average Cost": 10},
    ])
    flights = pd.DataFrame([
        {"Flight Number": "F1", "OriginCityName": "Other", "DestCityName": "Away", "FlightDate": "2022-03-01", "Price": 1},
        {"Flight Number": "F1", "OriginCityName": "Home", "DestCityName": "Away", "FlightDate": "2022-03-02", "Price": 20},
        {"Flight Number": "F1", "OriginCityName": "Home", "DestCityName": "Away", "FlightDate": "2022-03-01", "Price": 50},
    ])
    attractions = pd.DataFrame([{"Name": "Museum", "City": "Away"}, {"Name": "Park", "City": "Away"}])
    for module in (common, hard):
        for name, data in [("accommodation", hotels), ("restaurants", restaurants), ("flight", flights), ("attractions", attractions)]:
            monkeypatch.setattr(getattr(module, name), "data", data.copy(), raising=False)
        monkeypatch.setattr(module.googleDistanceMatrix, "run_for_evaluation", lambda *args, **kwargs: {"cost": 25})
    return common, hard


@pytest.fixture
def question():
    return {"days": 3, "org": "Home", "dest": "Away", "visiting_city_number": 1,
            "people_number": 3, "budget": 640, "date": ["2022-03-01", "2022-03-02", "2022-03-03"],
            "local_constraint": {"house rule": None, "cuisine": None, "room type": None, "transportation": None}}


def day(**kwargs):
    result = dict(current_city="Away", transportation="-", breakfast="-", lunch="-", dinner="-", attraction="-", accommodation="-")
    result.update(kwargs)
    return result
