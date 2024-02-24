import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
from tools.flights.apis import Flights
from tools.accommodations.apis import Accommodations
from tools.restaurants.apis import Restaurants
from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
from tools.attractions.apis import Attractions
import math
from tqdm import tqdm
import re
import random
import argparse
from datasets import load_dataset


flight = Flights()
accommodations = Accommodations()
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

def get_city_list(days, deparure_city, destination):
    city_list = []
    city_list.append(deparure_city)
    if days == 3:
        city_list.append(destination)
    else:
        city_set = open('../database/background/citySet_with_states.txt').read().split('\n')
        state_city_map = {}
        for unit in city_set:
            city, state = unit.split('\t')
            if state not in state_city_map:
                state_city_map[state] = []
            state_city_map[state].append(city)
        for city in state_city_map[destination]:
            if city != deparure_city:
                city_list.append(city + f"({destination})")
    return city_list

def extract_before_parenthesis(s):
    match = re.search(r'^(.*?)\([^)]*\)', s)
    return match.group(1) if match else s

def get_transportation(org,dest,date):
    transportation_price_info = {'Flight':1e9,'Self-driving':1e9,'Taxi':1e9}
    # get the flight information
    flight_info = flight.run(org, dest, date)
    if type(flight_info) != str and len(flight_info) > 0:
        flight_cost = int(flight_info.sort_values(by=['Price'],ascending=True).iloc[0]['Price'])
        transportation_price_info['Flight'] = flight_cost 
    # get the self-driving information
    self_driving_info = googleDistanceMatrix.run_for_evaluation(org, dest, mode='driving')
    if self_driving_info['cost'] != None:
        transportation_price_info['Self-driving'] = self_driving_info['cost'] *  math.ceil(1.0 / 5)
    # get the taxi information
    taxi_info = googleDistanceMatrix.run_for_evaluation(org, dest, mode='taxi')
    if taxi_info['cost'] != None:
        transportation_price_info['Taxi'] = taxi_info['cost'] *  math.ceil(1.0 / 4)
    sorted_dict = dict(sorted(transportation_price_info.items(), key=lambda item: item[1]))
    transportation = list(sorted_dict.keys())[0]
    if transportation_price_info[transportation] == 1e9:
        return False, None
    if transportation == 'Flight':
        transportation = f"Flight Number: {flight_info.sort_values(by=['Price'],ascending=True).iloc[0]['Flight Number']}"
    return True, transportation
    
def get_meal(city):
    restaurant = restaurants.run(city)
    if type(restaurant) == str:
        return False, None
    restaurant = restaurant.sort_values(by=["Average Cost"], ascending=True)

    for idx in range(len(restaurant)):
        # if f"{restaurant.iloc[idx]['Name']}, {city}" not in restaurant_data_list:
            return True, f"{restaurant.iloc[idx]['Name']}, {city}"
    return False, None

def get_attraction(city):
    attraction = attractions.run(city)
    if type(attraction) == str:
        return False, None
    idx = random.choice([i for i in range(len(attraction))])
    return True, f"{attraction.iloc[idx]['Name']}, {city}"
    # return False, None

def get_accommodation(city):
    accommodation = accommodations.run(city)

    if type(accommodation) == str:
        return False, None
    accommodation = accommodation.sort_values(by=["price"], ascending=True)
    if len(accommodation) == 0:
        return False, None
    
    return True, f"{accommodation.iloc[0]['NAME']}, {city}"



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--output_dir", type=str, default="./")
    args = parser.parse_args()
    if args.set_type == 'validation':
        query_data_list  = load_dataset('osunlp/TravelPlanner','validation')['validation']
    elif args.set_type == 'test':
        query_data_list  = load_dataset('osunlp/TravelPlanner','test')['test']
    for idx, query in enumerate(tqdm(query_data_list)):
        plan_list = [{'finished':[False,set()]}]
        restaurant_list = []
        attraction_list = []
        finished = False
        city_list = get_city_list(query['days'], query['org'], query['dest'])

        for i, unit in enumerate(city_list):
            city_list[i] = extract_before_parenthesis(unit)
        

        for current_day in range(1,query['days']+1):
            plan = {key:"-" for key in ['day','current_city','transportation','breakfast','lunch','dinner','attraction','accommodation']}
            plan['day'] = current_day
            current_city = None
            if current_day == 1:
                plan['current_city'] = f'from {city_list[0]} to {city_list[1]}'
                # get the transportation information
                flag, transportation = get_transportation(city_list[0],city_list[1],query['date'][0])
                if flag:
                    plan['transportation'] = f'{transportation}, from {city_list[0]} to {city_list[1]}'
                else:
                    plan_list[0]['finished'][0] = False
                    plan_list[0]['finished'][1].add('No valid transportation information.')

            elif current_day==3 and current_day == query['days']:
                plan['current_city'] = f'from {city_list[1]} to {city_list[0]}'
                # get the transportation information
                flag, transportation = get_transportation(city_list[1],city_list[0],query['date'][2])
                if flag:
                    plan['transportation'] = f'{transportation}, from {city_list[1]} to {city_list[0]}'
                else:
                    plan_list[0]['finished'][0] = False
                    plan_list[0]['finished'][1].add('No valid transportation information.')

            elif current_day == 3:
                plan['current_city'] = f'from {city_list[1]} to {city_list[2]}'
                # get the transportation information
                flag, transportation = get_transportation(city_list[1],city_list[2],query['date'][2])
                if flag:
                    plan['transportation'] = f'{transportation}, from {city_list[1]} to {city_list[2]}'
                else:
                    plan_list[0]['finished'][0] = False
                    plan_list[0]['finished'][1].add('No valid transportation information.')

            elif current_day == 5 and current_day == query['days']:
                plan['current_city'] = f'from {city_list[2]} to {city_list[0]}'
                # get the transportation information
                flag, transportation = get_transportation(city_list[2],city_list[0],query['date'][4])
                if flag:
                    plan['transportation'] = f'{transportation}, from {city_list[2]} to {city_list[0]}'
                else:
                    plan_list[0]['finished'][0] = False
                    plan_list[0]['finished'][1].add('No valid transportation information.')
            
            elif current_day == 5:
                plan['current_city'] = f'from {city_list[2]} to {city_list[3]}'
                # get the transportation information
                flag, transportation = get_transportation(city_list[2],city_list[3],query['date'][4])
                if flag:
                    plan['transportation'] = f'{transportation}, from {city_list[2]} to {city_list[3]}'
                else:
                    plan_list[0]['finished'][0] = False
                    plan_list[0]['finished'][1].add('No valid transportation information.')
            
            elif current_day == 7:
                plan['current_city'] = f'from {city_list[3]} to {city_list[0]}'
                # get the transportation information
                flag, transportation = get_transportation(city_list[3],city_list[0],query['date'][6])
                if flag:
                    plan['transportation'] = f'{transportation}, from {city_list[3]} to {city_list[0]}'
                else:
                    plan_list[0]['finished'][0] = False
                    plan_list[0]['finished'][1].add('No valid transportation information.')
            
            if plan['current_city'] == '-':
                plan['current_city'] = plan_list[-1]['current_city'].split(' to ')[1]
                current_city = plan['current_city']
            else:
                current_city = plan['current_city'].split(' to ')[1]


            # print(current_city)
            for key in ['breakfast','lunch','dinner']:
                flag, meal = get_meal(current_city)
                if flag:
                    plan[key] = f'{meal}'
                    restaurant_list.append(meal)
                else:
                    plan_list[0]['finished'][0] = False
                    plan_list[0]['finished'][1].add('No valid meal information.')
            
            flag, attraction = get_attraction(current_city)
            if flag:
                plan['attraction'] = f'{attraction}'
            else:
                plan_list[0]['finished'][0] = False
                plan_list[0]['finished'][1].add('No valid attraction information.')

            if current_day != query['days']:
                flag, accommodation = get_accommodation(current_city)
                if flag:
                    plan['accommodation'] = f'{accommodation}'
                else:
                    plan_list[0]['finished'][0] = False
                    plan_list[0]['finished'][1].add('No valid accommodation information.')

            plan_list.append(plan)
        
        if plan_list[0]['finished'][1] == set():
            plan_list[0]['finished'] = (True,[])
        
        # write to json file
        if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}')):
            os.makedirs(os.path.join(f'{args.output_dir}/{args.set_type}'))
        if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{idx+1}.json')):
            generated_plan = [{}]
        else:
            generated_plan = json.load(open(f'{args.output_dir}/{args.set_type}/plan_{idx+1}.json'))
        generated_plan[-1]['greedy_search_plan_success'] = [plan_list[0]['finished'][0],list(plan_list[0]['finished'][1])]
        generated_plan[-1]['greedy_search_plan'] = plan_list[1:]
        # print(generated_plan[-1])
        with open(f'{args.output_dir}/{args.set_type}/plan_{idx+1}.json', 'w') as f:
            json.dump(generated_plan, f)