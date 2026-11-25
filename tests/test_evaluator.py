import math

import pandas as pd
import pytest

from conftest import day


@pytest.mark.parametrize("suffix", ["", ";", "; "])
def test_attraction_last_item_is_checked(db, question, suffix):
    common, _ = db
    plan = [day(attraction="Museum, Away;Park, Away" + suffix), day(attraction=" Park, Away ")]
    assert common.is_valid_attractions(question, plan)[0] is False


@pytest.mark.parametrize("value", ["Museum, Away;Missing, Away", "Missing, Away"])
def test_attraction_sandbox_checks_without_trailing_separator(db, question, value):
    assert db[0].is_valid_information_in_sandbox(question, [day(attraction=value)])[0] is False


def test_attraction_city_checks_last_item(db, question):
    assert db[0].is_valid_information_in_current_city(question, [day(attraction="Museum, Away;Park, Other")])[0] is False


def test_attraction_empty_segments_are_ignored(db, question):
    plan = [day(attraction=" ; Museum, Away;; Park, Away; ; -;")]
    for check in (db[0].is_valid_attractions, db[0].is_valid_information_in_sandbox, db[0].is_valid_information_in_current_city):
        assert check(question, plan) == (True, None)


def test_minimum_nights_uses_exact_hotel(db, question):
    assert db[0].is_valid_accommodaton(question, [day(accommodation="Grand Hotel, Away"), day()])[0] is False


def test_minimum_nights_accepts_normalized_consecutive_stay(db, question):
    plan = [day(accommodation="Grand Hotel, Away"), day(accommodation=" Grand Hotel, Away (State) "), day()]
    assert db[0].is_valid_accommodaton(question, plan) == (True, None)


@pytest.mark.parametrize("gap", ["-", "", None, "Other Hotel, Away"])
def test_minimum_nights_does_not_combine_separate_stays(db, question, gap):
    plan = [day(accommodation="Grand Hotel, Away"), day(accommodation=gap), day(accommodation="Grand Hotel, Away")]
    assert db[0].is_valid_accommodaton(question, plan)[0] is False


@pytest.mark.parametrize("name", ["Missing, Away", "Hotel, Away"])
def test_minimum_nights_rejects_unresolved_accommodation(db, question, name):
    assert db[0].is_valid_accommodaton(question, [day(accommodation=name)])[0] is False


def test_duplicate_identical_rows_do_not_skip_minimum_nights(db, question):
    common, _ = db
    common.accommodation.data = pd.concat([common.accommodation.data] * 2, ignore_index=True)
    assert common.is_valid_accommodaton(question, [day(accommodation="Grand Hotel, Away")])[0] is False


def test_budget_counts_people_rooms_meals_and_each_night(db, question):
    plan = [day(current_city="from Home to Away", transportation="Flight Number: F1, from Home to Away", breakfast="Cafe, Away", lunch="Cafe, Away", dinner="Cafe, Away", accommodation="Grand Hotel, Away"),
            day(accommodation="Grand Hotel, Away"), day()]
    # 3 tickets * 50 + 3 meals * 10 * 3 people + 2 nights * 100 * 2 rooms.
    assert db[1].get_total_cost(question, plan) == 640
    assert db[1].evaluation(question, plan)["valid_cost"][0] is True
    assert db[1].evaluation(dict(question, budget=639), plan)["valid_cost"][0] is False


@pytest.mark.parametrize("mode,people,expected", [("self-driving", 5, 25), ("self-driving", 6, 50), ("taxi", 4, 25), ("taxi", 5, 50)])
def test_budget_vehicle_capacity(db, question, mode, people, expected):
    assert db[1].get_total_cost(dict(question, people_number=people), [day(transportation=f"{mode}, from Home to Away")]) == expected


@pytest.mark.parametrize("field,value", [("transportation", "Flight Number: missing, from Home to Away"), ("transportation", "Flight Number: F1, from Away to Home"), ("transportation", "taxi"), ("breakfast", "Missing, Away"), ("accommodation", "Missing, Away"), ("accommodation", "Hotel, Away")])
def test_unpriced_items_cannot_pass_budget(db, question, field, value):
    plan = [day(**{field: value})]
    assert math.isinf(db[1].get_total_cost(question, plan))
    assert db[1].evaluation(question, plan)["valid_cost"][0] is False


@pytest.mark.parametrize("cost", [None, float("nan"), -1, float("inf")])
def test_invalid_ground_cost_fails_budget_without_crashing(db, question, monkeypatch, cost):
    monkeypatch.setattr(db[1].googleDistanceMatrix, "run_for_evaluation", lambda *args, **kwargs: {"cost": cost})
    assert db[1].evaluation(question, [day(transportation="taxi, from Home to Away")])["valid_cost"][0] is False


@pytest.mark.parametrize("occupancy", [0, -1, float("nan")])
def test_invalid_occupancy_fails_budget(db, question, occupancy):
    db[1].accommodation.data.loc[1, "maximum occupancy"] = occupancy
    assert db[1].evaluation(question, [day(accommodation="Grand Hotel, Away")])["valid_cost"][0] is False


@pytest.mark.parametrize("label", ["Flight Number: F1", "flight number: F1", "FLIGHT NUMBER: F1"])
def test_flight_parsing_is_case_insensitive_and_route_specific(db, question, label):
    plan = [day(current_city="from Home to Away", transportation=label + ", FROM Home TO Away")]
    assert db[0].is_valid_information_in_sandbox(question, plan) == (True, None)
    assert db[1].get_total_cost(question, plan) == 150


@pytest.mark.parametrize("cities", [
    ["from Home to Away", "Away", "from Away to Home"],
    ["FROM Home TO Away", "Away (State)", "from Away to Home"],
])
def test_valid_closed_route(db, question, cities):
    assert db[0].is_reasonable_visiting_city(question, [day(current_city=c) for c in cities]) == (True, None)


@pytest.mark.parametrize("cities", [
    [], ["from Home to Away"],
    ["Other", "from Other to Away", "from Away to Other"],
    ["from Home to Other", "Other", "from Other to Home"],
    ["from Home to Away", "Away", "Away"],
    ["from Home", "Away", "from Away to Home"],
    [None, "Away", "from Away to Home"],
    ["from Home to Away", "from Away to Other", "from Away to Home"],
])
def test_invalid_routes_return_false(db, question, cities):
    assert db[0].is_reasonable_visiting_city(question, [day(current_city=c) for c in cities])[0] is False


def test_disconnected_route_with_repeated_cities_is_rejected(db, question):
    cities = ["from Home to Away", "Away", "Other", "Other", "from Other to Home"]
    assert db[0].is_reasonable_visiting_city(dict(question, days=5, dest="State"), [day(current_city=c) for c in cities])[0] is False


def test_valid_multi_city_route(db, question):
    cities = ["from Home to Away", "Away", "from Away to Other", "Other", "from Other to Home"]
    assert db[0].is_reasonable_visiting_city(dict(question, days=5, dest="State"), [day(current_city=c) for c in cities]) == (True, None)


@pytest.mark.parametrize("transport", ["taxi, from Other to Away", "taxi, from Away to Home", "taxi, from Home to Awayville"])
def test_transport_route_must_match_current_city_direction(db, question, transport):
    assert db[0].is_valid_information_in_current_city(question, [day(current_city="from Home to Away", transportation=transport)])[0] is False


def test_attraction_city_is_not_substring_match(db, question):
    assert db[0].is_valid_information_in_current_city(question, [day(attraction="Park, Awayville")])[0] is False


@pytest.mark.parametrize("cities", [[], ["Home", "Away", "Other"], ["Home", "Away", "Away", "Other"], ["Home", "Away", "Away", "Other", "Other", "Away", "Home"]])
def test_city_sequence_requires_closed_non_revisiting_route(db, cities):
    assert db[0].is_valid_city_sequence(cities) is False


def test_evaluation_empty_and_malformed_routes_do_not_crash(db, question):
    for plan in ([], [day(current_city="from Home"), day(), day(current_city="Home")]):
        result = db[0].evaluation(question, plan)
        assert result["is_reasonable_visiting_city"][0] is False


@pytest.mark.parametrize("minimum", [0, -1, float("nan"), None, "unknown"])
def test_invalid_minimum_nights_returns_false(db, question, minimum):
    db[0].accommodation.data['minimum nights'] = db[0].accommodation.data['minimum nights'].astype(object)
    db[0].accommodation.data.loc[1, 'minimum nights'] = minimum
    assert db[0].is_valid_accommodaton(question, [day(accommodation="Grand Hotel, Away")])[0] is False


def test_unique_partial_names_remain_supported(db, question):
    plan = [day(accommodation="Grand Hotel Annex, Away", breakfast="Cafe Annex, Away")]
    # No exact match, but the literal partial name has exactly one candidate.
    plan[0]['accommodation'] = 'Hotel Annex, Away'
    plan[0]['breakfast'] = 'afe Annex, Away'
    assert db[0].is_valid_accommodaton(question, plan) == (True, None)
    assert db[1].get_total_cost(question, plan) == 13


def test_budget_ignores_padding_beyond_requested_days(db, question):
    assert db[1].get_total_cost(question, [day(), day(), day(), day(accommodation="Grand Hotel, Away")]) == 0


def test_minimum_nights_does_not_count_padding(db, question):
    plan = [day(), day(), day(accommodation="Grand Hotel, Away"), day(accommodation="Grand Hotel, Away")]
    assert db[0].is_valid_accommodaton(question, plan)[0] is False


def test_flight_on_wrong_date_is_rejected(db, question):
    plan = [day(transportation="Flight Number: F1, from Home to Away")]
    query = dict(question, date=['2022-03-04'])
    assert db[0].is_valid_information_in_sandbox(query, plan)[0] is False
    assert db[1].evaluation(query, plan)['valid_cost'][0] is False


def test_city_sequence_rejects_destination_revisit_at_end(db):
    assert db[0].is_valid_city_sequence(['Home', 'Away', 'Away', 'Other', 'Other', 'Away']) is False


@pytest.mark.parametrize("uppercase", [False, True])
def test_valid_plan_passes_both_public_entry_points(db, question, uppercase):
    common, hard = db
    for module in db:
        module.flight.data = pd.concat([module.flight.data, pd.DataFrame([{
            'Flight Number': 'F2', 'OriginCityName': 'Away', 'DestCityName': 'Home',
            'FlightDate': '2022-03-03', 'Price': 30,
        }])], ignore_index=True)
        module.restaurants.data = pd.DataFrame([
            {'Name': name, 'City': 'Away', 'Average Cost': 10}
            for name in ('Breakfast Cafe', 'Lunch Cafe', 'Dinner Cafe')
        ])
    plan = [
        day(current_city='from Home to Away', transportation='Flight Number: F1, from Home to Away',
            attraction='Museum, Away', accommodation='Grand Hotel, Away'),
        day(breakfast='Breakfast Cafe, Away', lunch='Lunch Cafe, Away', dinner='Dinner Cafe, Away',
            attraction='Park, Away', accommodation='Grand Hotel, Away'),
        day(current_city='from Away to Home', transportation='Flight Number: F2, from Away to Home'),
    ]
    if uppercase:
        for unit in plan:
            for field in ('current_city', 'transportation'):
                unit[field] = unit[field].replace('from ', 'FROM ').replace(' to ', ' TO ')
    query = dict(question, budget=730)
    assert all(result[0] is True for result in common.evaluation(query, plan).values())
    assert common.boolean_evaluation(query, plan) is True
    assert hard.get_total_cost(query, plan) == 730
    assert hard.boolean_evaluation(query, plan) is True
    assert hard.boolean_evaluation(dict(query, budget=729), plan) is False


@pytest.mark.parametrize('cities', [
    ['from Home to Away', 'Away', 'from Away to Home', 'from Home to Other', 'from Other to Home'],
    ['from Home to Away', 'from Away to Elsewhere', 'Elsewhere', 'Elsewhere', 'from Elsewhere to Home'],
    ['from Home to Unknown', 'Unknown', 'Unknown', 'Unknown', 'from Unknown to Home'],
])
def test_multi_city_route_rejects_early_return_wrong_state_and_unknown_city(db, question, cities):
    assert db[0].is_reasonable_visiting_city(dict(question, days=5, dest='State'), [day(current_city=c) for c in cities])[0] is False
