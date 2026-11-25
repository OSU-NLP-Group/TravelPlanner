from utils.func import get_valid_name_city,extract_before_parenthesis,extract_numbers_from_filenames
from tools.flights.apis import Flights
from tools.accommodations.apis import Accommodations
from tools.restaurants.apis import Restaurants
from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
from tools.attractions.apis import Attractions
from evaluation.parsing import extract_from_to, flight_records, matching_records, travel_date
import math
import json
import re
import numpy as np
import os
import sys
from tqdm import tqdm
import argparse

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))


flight = Flights()
accommodation = Accommodations()
restaurants = Restaurants()
googleDistanceMatrix = GoogleDistanceMatrix()
attractions = Attractions()


def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            data.append(unit)
    return data


def convert_bool_values(item):
    if isinstance(item, dict):
        # If the item is a dictionary, recurse on each value
        return {key: convert_bool_values(value) for key, value in item.items()}
    elif isinstance(item, list):
        # If the item is a list, recurse on each item in the list
        return [convert_bool_values(value) for value in item]
    elif isinstance(item, tuple):
        # If the item is a tuple, recurse on each item in the tuple and repackage as a tuple
        return tuple(convert_bool_values(value) for value in item)
    elif isinstance(item, np.bool_):  # Here we check for numpy's bool_ type
        # If the item is a numpy bool_, convert it to a standard Python bool
        return bool(item)
    else:
        # If the item is any other type, return it unchanged
        return item




def _valid_amount(value, positive=False):
    try:
        value = float(value)
        return math.isfinite(value) and (value > 0 if positive else value >= 0)
    except (TypeError, ValueError):
        return False


def get_total_cost(question, tested_data):
    """Price all selected services; unresolvable prices return infinity.

    This keeps missing/ambiguous database records from silently becoming free
    services, and lets both public evaluation entry points fail the budget check.
    """
    total_cost = 0
    people = question['people_number']
    if not _valid_amount(people, positive=True):
        return math.inf
    people = float(people)
    for i, unit in enumerate(tested_data[:question['days']]):
        value = unit.get('transportation')
        if value and value != '-':
            origin, destination = extract_from_to(value)
            if origin is None or destination is None:
                origin, destination = extract_from_to(unit.get('current_city'))
            if origin is None or destination is None:
                return math.inf
            if 'flight number' in value.lower():
                records = flight_records(flight.data, value, origin, destination, travel_date(question, i))
                if len(records) != 1 or not _valid_amount(records.iloc[0]['Price']):
                    return math.inf
                total_cost += float(records.iloc[0]['Price']) * people
            elif 'self-driving' in value.lower() or 'taxi' in value.lower():
                mode = 'self-driving' if 'self-driving' in value.lower() else 'taxi'
                cost = googleDistanceMatrix.run_for_evaluation(origin, destination, mode)['cost']
                if not _valid_amount(cost):
                    return math.inf
                capacity = 5 if mode == 'self-driving' else 4
                total_cost += float(cost) * math.ceil(people / capacity)
            else:
                return math.inf

        for meal in ('breakfast', 'lunch', 'dinner'):
            value = unit.get(meal)
            if value and value != '-':
                records = matching_records(restaurants.data, value, 'Name', 'City')
                if len(records) != 1 or not _valid_amount(records.iloc[0]['Average Cost']):
                    return math.inf
                total_cost += float(records.iloc[0]['Average Cost']) * people

        value = unit.get('accommodation')
        if value and value != '-':
            records = matching_records(accommodation.data, value, 'NAME', 'city')
            if len(records) != 1:
                return math.inf
            record = records.iloc[0]
            if not _valid_amount(record['price']) or not _valid_amount(record['maximum occupancy'], positive=True):
                return math.inf
            total_cost += float(record['price']) * math.ceil(people / float(record['maximum occupancy']))
    return total_cost


def is_valid_room_rule(question, tested_data):

    if question['local_constraint']['house rule'] is None:
        return None,None
    
    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]
        if unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            res = matching_records(accommodation.data, unit['accommodation'], 'NAME', 'city')
            if len(res) > 0:
                if question['local_constraint']['house rule'] == 'smoking' and 'No smoking' in str(res['house_rules'].values[0]):
                    return False, f"The house rule should be {question['local_constraint']['house rule']}."
                if question['local_constraint']['house rule'] == 'parties' and 'No parties' in str(res['house_rules'].values[0]):
                    return False, f"The house rule should be {question['local_constraint']['house rule']}."
                if question['local_constraint']['house rule'] == 'children under 10' and 'No children under 10' in str(res['house_rules'].values[0]):
                    return False, f"The house rule should be {question['local_constraint']['house rule']}."
                if question['local_constraint']['house rule'] == 'visitors' and 'No visitors' in str(res['house_rules'].values[0]):
                    return False, f"The house rule should be {question['local_constraint']['house rule']}."
                if question['local_constraint']['house rule'] == 'pets' and 'No pets' in str(res['house_rules'].values[0]):
                    return False, f"The house rule should be {question['local_constraint']['house rule']}."
                
            
    return True, None



def is_valid_cuisine(question, tested_data):
    cuisine_set = set()
    if question['local_constraint']['cuisine']:
        for i in range(min(question['days'],len(tested_data))):
            unit = tested_data[i]

            if unit['breakfast'] and unit['breakfast'] != '-':
                name, city = get_valid_name_city(unit['breakfast'])
                if city == question['org']:
                    continue
                res = matching_records(restaurants.data, unit['breakfast'], 'Name', 'City')
                if len(res) > 0:       
                    for cuisine in question['local_constraint']['cuisine']:
                        if cuisine in res.iloc[0]['Cuisines']:
                            cuisine_set.add(cuisine)

            if unit['lunch'] and unit['lunch'] != '-':
                name, city = get_valid_name_city(unit['lunch'])
                if city == question['org']:
                    continue
                res = matching_records(restaurants.data, unit['lunch'], 'Name', 'City')
                if len(res) > 0:
                    for cuisine in question['local_constraint']['cuisine']:
                        if cuisine in res.iloc[0]['Cuisines']:
                            cuisine_set.add(cuisine)

            if unit['dinner'] and unit['dinner'] != '-':
                name, city = get_valid_name_city(unit['dinner'])
                if city == question['org']:
                    continue
                res = matching_records(restaurants.data, unit['dinner'], 'Name', 'City')
                if len(res) > 0:
                    for cuisine in question['local_constraint']['cuisine']:
                        if cuisine in res.iloc[0]['Cuisines']:
                            cuisine_set.add(cuisine)

        if len(cuisine_set) == len(question['local_constraint']['cuisine']):
            return True, None
        else:
            # judge which cuisine is not satisfied
            for cuisine in question['local_constraint']['cuisine']:
                if cuisine not in cuisine_set:
                    return False, f"The cuisine {cuisine} is not satisfied."
            # return False, f"The cuisine should be {question['local_constraint']['cuisine']}."
    else:
        return None,None
    

def is_valid_transportation(question, tested_data):
    if question['local_constraint']['transportation'] is None:
        return None,None
    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]
        if unit['transportation'] and unit['transportation'] != '-':
            value = unit['transportation']
            if question['local_constraint']['transportation'] == 'no flight' and 'Flight' in value:
                return False, f"The transportation should not be {question['local_constraint']['transportation']}."
            elif question['local_constraint']['transportation'] == 'no self-driving' and 'Self-driving'  in value:
                return False, f"The transportation should not be {question['local_constraint']['transportation']}."
            
    return True, None


def is_valid_room_type(question, tested_data):
    if question['local_constraint']['room type'] is None:
        return None,None
    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]
        if unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            res = matching_records(accommodation.data, unit['accommodation'], 'NAME', 'city')
            if len(res) > 0:
                if question['local_constraint']['room type'] == 'not shared room' and res['room type'].values[0] == 'Shared room':
                    return False, f"The room type should be {question['local_constraint']['room type']}."
                # "shared room", "not shared room", "private room", "entire room"
                elif question['local_constraint']['room type'] == 'shared room' and res['room type'].values[0] != 'Shared room':
                    return False, f"The room type should be {question['local_constraint']['room type']}."

                elif question['local_constraint']['room type'] == 'private room' and res['room type'].values[0] != 'Private room':
                    return False, f"The room type should be {question['local_constraint']['room type']}."

                elif question['local_constraint']['room type'] == 'entire room' and res['room type'].values[0] != 'Entire home/apt':
                    return False, f"The room type should be {question['local_constraint']['room type']}."

    return True, None


def evaluation(query_data, tested_data):
    return_info = {}
    return_info['valid_cuisine'] = is_valid_cuisine(query_data, tested_data)
    return_info['valid_room_rule'] = is_valid_room_rule(query_data, tested_data)
    return_info['valid_transportation'] = is_valid_transportation(query_data, tested_data)
    return_info['valid_room_type'] = is_valid_room_type(query_data, tested_data)
    return_info['valid_cost'] = (bool(get_total_cost(query_data, tested_data) <= query_data['budget']), None)
    return return_info

def boolean_evaluation(query_data, tested_data):
    return_info = {}
    return_info['valid_cuisine'] = is_valid_cuisine(query_data, tested_data)
    return_info['valid_room_rule'] = is_valid_room_rule(query_data, tested_data)
    return_info['valid_transportation'] = is_valid_transportation(query_data, tested_data)
    return_info['valid_room_type'] = is_valid_room_type(query_data, tested_data)
    return_info['valid_cost'] = (bool(get_total_cost(query_data, tested_data) <= query_data['budget']), None)
    for key in return_info:
        if return_info[key][0] == False:
            print(key)
            return False
    return True

