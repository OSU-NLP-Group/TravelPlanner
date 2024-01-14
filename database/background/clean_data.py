with open('database/background/citySet.txt','r') as f:
    city_set = f.read().strip().split('\n')

with open('database/background/citySet_with_states.txt','r') as f:
    lines = f.read().strip().split('\n')
    data = []
    for unit in lines:
        if unit.split('\t')[0] in city_set:
            data.append(unit)

with open('database/background/citySet_with_states.txt','w') as f:
    for unit in data:
        f.write(unit + '\n')
    f.close()