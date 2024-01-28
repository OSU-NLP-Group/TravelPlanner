import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import random
from utils.budget_estimation import budget_calc
import json
from datetime import datetime, timedelta
from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
import numpy as np

google_distance = GoogleDistanceMatrix()

city_set = open('../database/background/citySet_with_states.txt').read().strip().split('\n')

state_city_map = {}

for city in city_set:   
    state = city.split('\t')[1]
    if state not in state_city_map:
        state_city_map[state] = [city.split('\t')[0]]
    else:
        state_city_map[state].append(city.split('\t')[0])

visiting_city_map = {3:1,5:2,7:3}

def round_to_hundreds(num):
    return round(num / 100) * 100

def select_consecutive_dates(num_days, start_date=datetime(2022, 3, 1), end_date=datetime(2022, 4, 1)):
    """
    Selects consecutive dates within the given range.
    """
    # Generate a list of all possible dates within the range
    delta = end_date - start_date
    all_dates = [start_date + timedelta(days=i) for i in range(delta.days)]
    
    # Get the latest possible starting date for the consecutive days
    latest_start = len(all_dates) - num_days
    
    # Randomly select a starting point
    start_index = random.randint(0, latest_start)
    
    # Extract the consecutive dates
    consecutive_dates = all_dates[start_index:start_index+num_days]
    
    return consecutive_dates


def get_org_dest(days:int):
    if days == 3:
        city_set = open('../database/background/citySet_with_states.txt').read().strip().split('\n')

        org = random.choice(city_set)

        while True:
            dest = random.choice(city_set)
            if dest.split('\t')[1] != org.split('\t')[1]:
                break

        final_org = org.split('\t')[0]
        final_des = dest.split('\t')[0]

    elif days in [5,7]:
    
        state_set = open('../database/background/citySet_with_states.txt').read().strip().split('\n')
        org = random.choice(state_set)

        while True:
            dest = random.choice(state_set)
            if dest != org and "None" not in dest and dest.split('\t')[1] != org.split('\t')[1] and len(state_city_map[dest.split('\t')[1]]) > 3:
                break
        final_org = org.split('\t')[0]
        final_des = dest.split('\t')[1]

    return final_org, final_des


def easy_level_element_selection(day_list):
    """Selects the element to be used in the easy level query."""
    days = random.choice(day_list)
    query_dict = None
    date  = [date.strftime('%Y-%m-%d') for date in select_consecutive_dates(days)]
    final_org, final_des = get_org_dest(days)
    budget = budget_calc(final_org, final_des, date=date, days=days )
    local_constraint_list = ["house rule", "cuisine","room type",'transportation']
    local_constrain_record = {key:None for key in local_constraint_list}
    if days == 3:
        final_budget = round_to_hundreds((budget["average"]+budget["lowest"])/2)
    elif days == 5:
        final_budget = round_to_hundreds(budget["average"])
    elif days == 7:
        final_budget = round_to_hundreds(round_to_hundreds((budget["average"]+budget["highest"])/2))

    query_dict = {"org": final_org, "dest": final_des, "days": days, "visiting_city_number":visiting_city_map[days] ,"date":date, "people_number": 1, "local_constraint": local_constrain_record ,"budget": final_budget,"query": None, "level":"easy"}
    return query_dict



def medium_level_element_selection(day_list):
    days = random.choice(day_list)
    date  = [date.strftime('%Y-%m-%d') for date in select_consecutive_dates(days)]
    people_number = random.choice(random.choice([[2],[3,4,5,6,7,8]]))
    local_constraint_list = ["house rule", "cuisine","room type"]
    local_constrain_record = {key:None for key in local_constraint_list}
    local_constrain_record['transportation'] = None
    final_org, final_des = get_org_dest(days)

    local_constraint_type = random.choice(local_constraint_list)

    if local_constraint_type == "flight time":
        local_constraint = random.choice(["morning", "afternoon", "evening"])
        local_constrain_record["flight time"] = local_constraint

    # elif local_constraint_type == "rating":
    #     local_constraint = random.choice([3, 3,5,4,4.5])
    #     local_constrain_record["rating"] = local_constraint

    elif local_constraint_type == "room type":
        if people_number <= 2:
            local_constraint = random.choice(["shared room", "not shared room", "private room", "entire room"])
        else:
            local_constraint = random.choice(["private room", "entire room"])
        local_constrain_record["room type"] = local_constraint

    elif local_constraint_type == "house rule":
        local_constraint = random.choice(["parties","smoking","children under 10","visitors","pets"])
        local_constrain_record["house rule"] = local_constraint

    elif local_constraint_type == "cuisine":
        # choice_number = random.choice([2,3,4,5])
        local_constraint = random.sample(["Chinese", "American", "Italian", "Mexican", "Indian","Mediterranean","French"], 2)
        local_constrain_record["cuisine"] = local_constraint
    
    budget = budget_calc(final_org, final_des, days=days, date=date, people_number=people_number)

    if days == 3:
        final_budget = round_to_hundreds((budget["average"]+budget["lowest"])/2 * people_number * 0.75)
    elif days == 5:
        final_budget = round_to_hundreds(budget["average"] * people_number * 0.75)
    elif days == 7:
        final_budget = round_to_hundreds(round_to_hundreds((budget["average"]+budget["highest"])/2) * people_number * 0.75)

    query_dict = {"org": final_org, "dest": final_des, "days": days, "visiting_city_number":visiting_city_map[days], "date":date, "people_number": people_number, "local_constraint": local_constrain_record ,"budget": final_budget,"query": None, "level":"medium"}
    return query_dict



def hard_level_element_selection(day_list):
    days = random.choice(day_list)
    date  = [date.strftime('%Y-%m-%d') for date in select_consecutive_dates(days)]
    people_number = random.choice(random.choice([[2],[3,4,5,6,7,8]]))
    # local_constraint_list = ["flight time", "house rule", "cuisine","room type", "transportation"]
    local_constraint_list = ["house rule", "cuisine","room type","transportation"]
    probabilities = [0.3, 0.1, 0.3, 0.3] 
    final_org, final_des = get_org_dest(days)
    # result = google_distance.run(final_org, final_des)

    # if result != {} and 'day' not in result["duration"]:
    #     local_constraint_list.append()

    local_constrain_record = {key:None for key in local_constraint_list}

    local_constraint_type_list = np.random.choice(local_constraint_list, size=3, replace=False, p=probabilities).tolist()

    for local_constraint_type in local_constraint_type_list:
        if local_constraint_type == "flight time":
            local_constraint = random.choice(["morning", "afternoon", "evening"])
            local_constrain_record["flight time"] = local_constraint

        elif local_constraint_type == "transportation":
            local_constraint = random.choice(["no flight", "no self-driving"])
            local_constrain_record["transportation"] = local_constraint

        elif local_constraint_type == "room type":
            if people_number <= 2:
                local_constraint = random.choice(["shared room", "not shared room", "private room", "entire room"])
            else:
                local_constraint = random.choice(["private room", "entire room"])
            local_constrain_record["room type"] = local_constraint

        elif local_constraint_type == "house rule":
            local_constraint = random.choice(["parties","smoking","children under 10","visitors","pets"])
            local_constrain_record["house rule"] = local_constraint

        elif local_constraint_type == "cuisine":
            # choice_number = random.choice([2,3,4,5])
            local_constraint = random.sample(["Chinese", "American", "Italian", "Mexican", "Indian","Mediterranean","French"], 4)
            local_constrain_record["cuisine"] = local_constraint
    
    budget = budget_calc(final_org, final_des, days=days, date=date, people_number=people_number,local_constraint=local_constrain_record)

    if days == 3:
        final_budget = round_to_hundreds((budget["average"]+budget["lowest"])/2 * people_number * 0.5)
    elif days == 5:
        final_budget = round_to_hundreds(budget["average"] * people_number * 0.5)
    elif days == 7:
        final_budget = round_to_hundreds(round_to_hundreds((budget["average"]+budget["highest"])/2) * people_number * 0.5)

    query_dict = {"org": final_org, "dest": final_des, "days": days, "visiting_city_number":visiting_city_map[days], "date":date, "people_number": people_number, "local_constraint": local_constrain_record ,"budget": final_budget, "query": None,"level":"hard"}

    return query_dict


def generate_elements(number:int, level="easy", day_list=[3,5,7]):
    """Generate the elements for the easy level query."""
    query_list = []
    while len(query_list) < number:
        print(len(query_list))
        try:
            if level == "easy":
                query = easy_level_element_selection(day_list)
                if query not in query_list:
                    query_list.append(query)
            elif level == "medium":
                query = medium_level_element_selection(day_list)
                if query not in query_list:
                    query_list.append(query)
            elif level == "hard":
                query = hard_level_element_selection(day_list)
                if query not in query_list:
                    query_list.append(query)
        except ValueError:
            continue
    return query_list


if __name__ == "__main__":

    """Generate the elements for the different level query."""

    # save query_list as jsonl file
    for num, day_list in zip([160,160,160], [[3],[5],[7]]):
        query_list = generate_elements(num,"medium",day_list=day_list)

        with open('../data/query/final_annotation_medium.jsonl', 'a+') as f:
            for query in query_list:
                # print(query)
                json.dump(query, f)
                f.write('\n')
            f.close()