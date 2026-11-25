from utils.func import get_valid_name_city,extract_before_parenthesis,extract_numbers_from_filenames
from tools.flights.apis import Flights
from tools.accommodations.apis import Accommodations
from tools.restaurants.apis import Restaurants
from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
from tools.attractions.apis import Attractions
from evaluation.parsing import (
    city_route, extract_from_to, flight_records, matching_records,
    normalize_city, parse_attractions, travel_date,
)
import math
import json
import re   
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

city_state_set = open('../database/background/citySet_with_states.txt','r').read().split('\n')
city_state_map = {x:y for x,y in [unit.split('\t') for unit in city_state_set]}


def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            data.append(unit)
    return data


def count_consecutive_values(lst):
    if not lst:
        return []

    result = []
    current_string = lst[0]
    count = 1

    for i in range(1, len(lst)):
        if lst[i] == current_string:
            count += 1
        else:
            result.append((current_string, count))
            current_string = lst[i]
            count = 1

    result.append((current_string, count))  # Add the last group of values
    return result


def transportation_match(text: str):

    if 'taxi' in text.lower():
        return 'Taxi'
    
    elif 'self-driving' in text.lower():
        return 'Self-driving'
    
    elif 'flight' in text.lower():
        return 'Flight'


def is_valid_city_sequence(city_list):
    """A closed route may visit each destination in only one consecutive run."""
    if len(city_list) < 3 or city_list[0] != city_list[-1]:
        return False
    runs = count_consecutive_values(city_list)
    if len(runs) < 3:
        return False
    interior = [city for city, _ in runs[1:-1]]
    return (city_list[0] not in interior and
            len(interior) == len(set(interior)) and
            all(count >= 2 for _, count in runs[1:-1]))


def is_reasonable_visiting_city(question, tested_data):
    if not tested_data or len(tested_data) < question['days']:
        return False, "The route is incomplete."

    origin = normalize_city(question['org'])
    previous_city = origin
    city_list = []
    for i, unit in enumerate(tested_data[:question['days']]):
        start, end = city_route(unit.get('current_city'))
        if start is None or end is None:
            return False, f"The city in day {i+1} cannot be parsed."
        if start != previous_city:
            return False, f"The route in day {i+1} should start in {previous_city}."
        city_list.extend([start, end] if start != end else [start])
        previous_city = end

    if previous_city != origin:
        return False, "The trip should return to its origin."
    if not is_valid_city_sequence(city_list):
        return False, "The city sequence is invalid."

    for city in city_list:
        if city not in city_state_map:
            return False, f"{city} is not a valid city."
        if city != origin:
            if question['days'] == 3 and city != normalize_city(question['dest']):
                return False, f"{city} is not the requested destination."
            if question['days'] > 3 and city_state_map[city] != question['dest']:
                return False, f"{city} is not in {question['dest']}."
    return True, None


def is_valid_restaurants(question, tested_data):

    restaurants_list = []

    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]

        if 'breakfast' in unit and unit['breakfast'] and unit['breakfast'] != '-':
            if unit['breakfast'] not in restaurants_list:
                restaurants_list.append(unit['breakfast'])
            else:
                return False, f"The restaurant in day {i+1} breakfast is repeated."
        # elif 'breakfast' not in unit :
        #     return False, f"No Breakfast Info."
            
        if 'lunch' in unit and unit['lunch'] and unit['lunch'] != '-':
            if unit['lunch'] not in restaurants_list:
                restaurants_list.append(unit['lunch'])
            else:
                return False, f"The restaurant in day {i+1} lunch {unit['lunch']} is repeated."
        # elif 'lunch' not in unit:
        #     return False, f"No Lunch Info."
        
        if 'dinner' in unit and unit['dinner'] and unit['dinner'] != '-':
            if unit['dinner'] not in restaurants_list:
                restaurants_list.append(unit['dinner'])
            else:
                return False, f"The restaurant in day {i+1} dinner is repeated."
        # elif 'dinner' not in unit:
        #     return False, f"No Dinner Info."

    return True, None
            
def is_valid_attractions(question, tested_data):

    attractions_list = []

    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]

        if 'attraction' in unit and unit['attraction'] and unit['attraction'] != '-':
            for attraction in parse_attractions(unit['attraction']):
                identity = get_valid_name_city(attraction)
                if identity not in attractions_list:
                    attractions_list.append(identity)
                else:
                    return False, f"The attraction '{attraction}' in day {i+1} is repeated."
                
        # elif 'attraction' not in unit:
        #     return False, f"No Attraction Info."
        
    return True, None

def is_valid_transportation(question, tested_data):
    
    if tested_data and tested_data[0].get('transportation') not in (None, '', '-'):
        transportation_list = [transportation_match(tested_data[0]['transportation'])]
    
    else:
        return False, "The transportation in day 1 should not be empty."

    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]

        if 'transportation' in unit and unit['transportation'] and unit['transportation'] != '-':
            transportation_list.append(transportation_match(unit['transportation']))
        # elif 'transportation' not in unit:
        #     return False, f"No Transportation Info."
    
    if (('Self-driving' in transportation_list) and ('Flight' in transportation_list)) or (('Taxi' in transportation_list) and ('Self-driving' in transportation_list)):
        return False, "The transportation is conflicting."

    return True, None

def is_valid_information_in_current_city(question, tested_data):
    for i, unit in enumerate(tested_data[:question['days']]):
        start, end = city_route(unit.get('current_city'))
        if start is None or end is None:
            return False, f"The city in day {i+1} cannot be parsed."
        value = unit.get('transportation')
        if value and value != '-':
            if extract_from_to(value) != (start, end):
                return False, f"The transportation in day {i+1} is invalid city choice."
        for meal in ('breakfast', 'lunch', 'dinner'):
            value = unit.get(meal)
            if value and value != '-' and get_valid_name_city(value)[1] not in (start, end):
                return False, f"The {meal} in day {i+1} is invalid city choice."
        for attraction in parse_attractions(unit.get('attraction')):
            if get_valid_name_city(attraction)[1] not in (start, end):
                return False, f"The attraction in day {i+1} is invalid city choice."
        value = unit.get('accommodation')
        if value and value != '-' and get_valid_name_city(value)[1] != end:
            return False, f"The accommodation in day {i+1} is invalid city choice."
    return True, None

# hallucination 
def is_valid_information_in_sandbox(question, tested_data):
    
    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]
        
        if unit.get('transportation') and unit['transportation'] != '-':
            value = unit['transportation']
            org_city, dest_city = extract_from_to(value)
            if org_city == None or dest_city == None:
                org_city, dest_city = extract_from_to(unit.get('current_city'))
            if 'flight number' in value.lower():
                if len(flight_records(flight.data, value, org_city, dest_city, travel_date(question, i))) != 1:
                    return False, f"The flight number in day {i+1} is invalid in the sandbox."

            elif 'self-driving' in value.lower() or 'taxi' in value.lower():
                try:
                    org_city = extract_before_parenthesis(org_city)
                    dest_city = extract_before_parenthesis(dest_city)
                except TypeError:
                    org_city = '-'
                    dest_city = '-'
                    print("The transportation {} in day {} can not be parsed and '-' will be used instead.".format(value,i+1))
                
                if 'self-driving' in value.lower():
                    if googleDistanceMatrix.run_for_evaluation(org_city, dest_city, mode='self-driving')['cost'] == None:
                        return False, f"The self-driving in day {i+1} is invalid in the sandbox."
                else:
                    if googleDistanceMatrix.run_for_evaluation(org_city, dest_city, mode='taxi')['cost'] == None:
                        return False, f"The taxi in day {i+1} is invalid in the sandbox."

        if 'breakfast' in unit and unit['breakfast'] and unit['breakfast'] != '-':
            if len(matching_records(restaurants.data, unit['breakfast'], 'Name', 'City')) != 1:
                return False, f"The breakfast in day {i+1} is invalid in the sandbox."
        # elif 'breakfast' not in unit:
        #     return False, f"No Breakfast Info."
        
        if 'lunch' in unit and unit['lunch'] and unit['lunch'] != '-':
            if len(matching_records(restaurants.data, unit['lunch'], 'Name', 'City')) != 1:
                return False, f"The lunch in day {i+1} is invalid in the sandbox."
        # elif 'lunch' not in unit:
        #     return False, f"No Lunch Info."
        
        if 'dinner' in unit and unit['dinner'] and unit['dinner'] != '-':
            if len(matching_records(restaurants.data, unit['dinner'], 'Name', 'City')) != 1:
                return False, f"The dinner in day {i+1} is invalid in the sandbox."
        # elif 'dinner' not in unit:
        #     return False, f"No Dinner Info."
            
        if 'attraction' in unit and unit['attraction'] and unit['attraction'] != '-':
            attractions_list = parse_attractions(unit['attraction'])
            for attraction in attractions_list:
                if len(matching_records(attractions.data, attraction, 'Name', 'City')) < 1:
                    return False, f"The attraction {attraction} in day {i+1} is invalid in the sandbox."
        # elif 'attraction' not in unit:
        #     return False, f"No Attraction Info."
                
        if 'accommodation' in unit and unit['accommodation'] and unit['accommodation'] != '-':
            if len(matching_records(accommodation.data, unit['accommodation'], 'NAME', 'city')) != 1:
                return False, f"The accommodation in day {i+1} is invalid in the sandbox."
        # elif 'accommodation' not in unit:
        #     return False, f"No Accommodation Info."
        
    return True, None


def is_valid_accommodaton(question, tested_data):
    # Missing nights break a stay; whitespace and city annotations do not.
    stays = []
    for unit in tested_data[:question['days']]:
        if 'accommodation' not in unit:
            return False, "No Accommodation Info."
        value = unit['accommodation']
        if not value or value.strip() in ('', '-'):
            stays.append(None)
            continue
        records = matching_records(accommodation.data, value, 'NAME', 'city')
        if len(records) != 1:
            return False, f"The accommodation {value} cannot be uniquely identified."
        record = records.iloc[0]
        try:
            minimum = float(record['minimum nights'])
        except (TypeError, ValueError):
            return False, f"The accommodation {value} has invalid minimum nights."
        if not math.isfinite(minimum) or minimum <= 0:
            return False, f"The accommodation {value} has invalid minimum nights."
        stays.append((record['NAME'], record['city'], minimum))

    for stay, nights in count_consecutive_values(stays):
        if stay is not None and nights < stay[2]:
            return False, f"The accommodation {stay[0]}, {stay[1]} does not obey the minimum nights rule."
    return True, None


def is_valid_visiting_city_number(question, tested_data):
    city_set = set()
    for i, unit in enumerate(tested_data[:question['days']]):
        start, end = city_route(unit.get('current_city'))
        if start is None or end is None:
            return False, f"The city in day {i+1} cannot be parsed."
        city_set.update((start, end))
    city_set.discard(normalize_city(question['org']))
    if len(city_set) != question['visiting_city_number']:
        return False, f"The number of visiting cities should be {question['visiting_city_number']}."
    return True, None

def is_valid_days(question, tested_data):
    lens = 0
    for i in range(min(question['days'],len(tested_data))):
        if tested_data[i] != {} and tested_data[i].get('current_city') not in (None, '') and tested_data[i]['current_city'] != "You don't need to fill in the information for this or later days.":
            lens += 1
        
    if lens != question['days']:
        # print(lens)
        return False, f"The number of days should be {question['days']}."
    else:
        return True, None

def is_not_absent(question, tested_data):
    needed_info = 6 * question['days']
    total_valid_info = 0

    if not is_valid_days(question, tested_data)[0]:
        return False, "Invalid Days"
    
    if not is_valid_visiting_city_number(question, tested_data)[0]:
        return False, "Invalid City Number"

    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]

        if 'transportation' not in unit:
            return False, f"No Transportation Info."
        
        if 'breakfast' not in unit:
            return False, f"No Breakfast Info."
        
        if 'lunch' not in unit:
            return False, f"No Lunch Info."
        
        if 'dinner' not in unit:
            return False, f"No Dinner Info."
        
        if 'attraction' not in unit:
            return False, f"No Attraction Info."
        
        if 'accommodation' not in unit:
            return False, f"No Accommodation Info."
        
        start, end = city_route(unit['current_city'])
        traveling = start != end
        if traveling and unit['transportation'] in ['', '-']:
            return False, f"No transportation in day {i+1} is not allowed."
        
        if not traveling and not parse_attractions(unit['attraction']):
            return False, f"No attaction in day {i+1} is not allowed."

        if i != question['days'] - 1 and unit['accommodation'] in ['','-']:
            return False, f"No accommodation in day {i+1} is not allowed."

        if (unit['breakfast'] in ['','-'] or unit['lunch'] in ['','-'] or unit['dinner'] in ['','-']) and not traveling:
            return False, f"No meal in day {i+1} is not allowed."
        

        for key in unit:
            if unit[key] and unit[key] != '-':
                total_valid_info += 1


    if total_valid_info * 1.0 / needed_info < 0.5:
        return False, f"The absent information is more than 50%."
    
    return True, None


def evaluation(query_data, tested_data):
    return_info = {}
    return_info['is_reasonable_visiting_city'] = is_reasonable_visiting_city(query_data, tested_data)
    return_info['is_valid_restaurants'] = is_valid_restaurants(query_data, tested_data)
    return_info['is_valid_attractions'] = is_valid_attractions(query_data, tested_data)
    return_info['is_valid_accommodation'] = is_valid_accommodaton(query_data, tested_data)
    return_info['is_valid_transportation'] = is_valid_transportation(query_data, tested_data)
    return_info['is_valid_information_in_current_city'] = is_valid_information_in_current_city(query_data, tested_data)
    return_info['is_valid_information_in_sandbox'] = is_valid_information_in_sandbox(query_data, tested_data)
    return_info['is_not_absent'] = is_not_absent(query_data, tested_data)
    return return_info

def boolean_evaluation(query_data, tested_data):
    return_info = {}
    return_info['is_reasonable_visiting_city'] = is_reasonable_visiting_city(query_data, tested_data)
    return_info['is_valid_restaurants'] = is_valid_restaurants(query_data, tested_data)
    return_info['is_valid_accommodation'] = is_valid_accommodaton(query_data, tested_data)
    return_info['is_valid_attractions'] = is_valid_attractions(query_data, tested_data)
    return_info['is_valid_transportation'] = is_valid_transportation(query_data, tested_data)
    return_info['is_valid_information_in_current_city'] = is_valid_information_in_current_city(query_data, tested_data)
    return_info['is_valid_information_in_sandbox'] = is_valid_information_in_sandbox(query_data, tested_data)
    return_info['is_not_absent'] = is_not_absent(query_data, tested_data)
    for key in return_info:
        if return_info[key][0] == False:
            print(return_info[key][1])
            return False
    return True
