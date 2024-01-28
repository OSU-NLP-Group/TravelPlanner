import json
import re
import gradio as gr
import os

def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            data.append(unit)
    return data

def save_file(data, path):
    with open(path,'w',encoding='utf-8') as w:
        for unit in data:
            output = json.dumps(unit)
            w.write(output + "\n")
        w.close()

def extract_query_number(query_string):
    """
    Extract the number from a query string formatted as "Query X" or "Query X --- Done".
    
    Args:
    - query_string (str): The input string.
    
    Returns:
    - int: The extracted number if found, else None.
    """
    pattern = r"Query (\d+)"
    match = re.search(pattern, query_string)
    return int(match.group(1)) if match else None

def create_data_display(css_content,data,annotation_idx):
    return f"""
    <style>
    {css_content}
    </style>
    <div>
        <span class="query-highlighted"><strong>Query {annotation_idx}:</strong> {data[annotation_idx-1]['query']}</span><br>
        <span class="highlighted"><strong>Day:</strong> {data[annotation_idx-1]['days']}</span>
        <span class="highlighted"><strong>Visiting City Number:</strong> {data[annotation_idx-1]['visiting_city_number']}</span>
        <span class="highlighted"><strong>Date:</strong> {data[annotation_idx-1]['date']}</span>
        <span class="highlighted"><strong>Departure:</strong> {data[annotation_idx-1]['org']}</span>
        <span class="highlighted"><strong>Destination:</strong> {data[annotation_idx-1]['dest']}</span><br>
        <span class="highlighted-alt"><strong>People Number:</strong> {data[annotation_idx-1]['people_number']}</span>
        <span class="highlighted-alt"><strong>Budget:</strong> {data[annotation_idx-1]['budget']}</span>
        <span class="highlighted-alt"><strong>Hotel Rule:</strong> {data[annotation_idx-1]['local_constraint']['house rule']}</span>
        <span class="highlighted-alt"><strong>Cuisine:</strong> {data[annotation_idx-1]['local_constraint']['cuisine']}</span>
        <span class="highlighted-alt"><strong>Room Type:</strong> {data[annotation_idx-1]['local_constraint']['room type']}</span>
        <span class="highlighted-alt"><strong>Transportation:</strong> {data[annotation_idx-1]['local_constraint']['transportation']}</span><br>
    </div>
    """

def judge_valid_info(info):
    if info == "" or not info or info == "You don't need to fill in the information for this or later days." :
        return False
    return True

def judge_submit_info(info, current_day, label, annotation_data, *tested_data):
    if info == "" or not info:
        raise gr.Error("Day {} {} is empty!".format(current_day, label))
    if info != "-":
        if label == "transportation":
            if not judge_valid_transportation(info, annotation_data):
                raise gr.Error("Day {} {} is invalid! Please note the transportation.".format(current_day, label))
        elif label == "accommodation":
            if not judge_valid_room_type(info, annotation_data, tested_data[0]):
                raise gr.Error("Day {} {} is invalid! Please note the room type.".format(current_day, label))
            
            if not  judge_valid_room_rule(info, annotation_data, tested_data[0]):
                raise gr.Error("Day {} {} is invalid! Please note the house rules.".format(current_day, label))
        
    return True


def judge_valid_transportation(info, annotation_data):
    if  annotation_data['local_constraint']['transportation'] == 'no flight' and 'Flight' in info:
        return False
    elif annotation_data['local_constraint']['transportation'] == 'no self-driving' and 'Self-driving'  in info:
        return False
    return True

def judge_valid_room_type(info, annotation_data, accommodation_data_all):
    accommodation_data_filtered = get_filtered_data(info, accommodation_data_all)
    if annotation_data['local_constraint']['room type'] == 'not shared room' and accommodation_data_filtered['room type'].values[0] == 'Shared room':
        return False
    # "shared room", "not shared room", "private room", "entire room"
    elif annotation_data['local_constraint']['room type'] == 'shared room' and accommodation_data_filtered['room type'].values[0] != 'Shared room':
        return False

    elif annotation_data['local_constraint']['room type'] == 'private room' and accommodation_data_filtered['room type'].values[0] != 'Private room':
        return False

    elif annotation_data['local_constraint']['room type'] == 'entire room' and accommodation_data_filtered['room type'].values[0] != 'Entire home/apt':
        return False

    return True

def judge_valid_room_rule(info, annotation_data, accommodation_data_all):
    accommodation_data_filtered = get_filtered_data(info, accommodation_data_all)
    if annotation_data['local_constraint']['house rule'] == 'smoking' and 'No smoking' in str(accommodation_data_filtered['house_rules'].values[0]):
        return False
    if annotation_data['local_constraint']['house rule'] == 'parities' and 'No parties' in str(accommodation_data_filtered['house_rules'].values[0]):
        return False
    if annotation_data['local_constraint']['house rule'] == 'children under 10' and 'No children under 10' in str(accommodation_data_filtered['house_rules'].values[0]):
        return False
    if annotation_data['local_constraint']['house rule'] == 'visitors' and 'No visitors' in str(accommodation_data_filtered['house_rules'].values[0]):
        return False
    if annotation_data['local_constraint']['house rule'] == 'pets' and 'No pets' in str(accommodation_data_filtered['house_rules'].values[0]):
        return False
    
    return True

def judge_valid_cuisine(info, annotation_data, restaurant_data_all, cuisine_set: set):
    if info != "-" and annotation_data['local_constraint']['cuisine'] is not None and annotation_data['org'] not in info:
        restaurant_data_filtered = get_filtered_data(info, restaurant_data_all,('Name','City'))
        for cuisine in annotation_data['local_constraint']['cuisine']:
            if cuisine in restaurant_data_filtered.iloc[0]['Cuisines']:
                cuisine_set.add(cuisine)
    return cuisine_set




def get_valid_name_city(info):
    # Modified the pattern to preserve spaces at the end of the name
    pattern = r'(.*?),\s*([^,]+)(\(\w[\w\s]*\))?$'
    match = re.search(pattern, info)
    if match:
        return match.group(1).strip(), extract_before_parenthesis(match.group(2).strip()).strip()
    else:
        print(f"{info} can not be parsed, '-' will be used instead.")
        return "-","-"

    
def extract_numbers_from_filenames(directory):
    # Define the pattern to match files
    pattern = r'annotation_(\d+).json'

    # List all files in the directory
    files = os.listdir(directory)

    # Extract numbers from filenames that match the pattern
    numbers = [int(re.search(pattern, file).group(1)) for file in files if re.match(pattern, file)]

    return numbers

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

def get_filtered_data(component,data, column_name=('NAME','city')):
    name, city = get_valid_name_city(component)
    return data[(data[column_name[0]] == name) & (data[column_name[1]] == city)]

def extract_before_parenthesis(s):
    match = re.search(r'^(.*?)\([^)]*\)', s)
    return match.group(1) if match else s

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