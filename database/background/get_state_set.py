import os
# print now directory
print(os.getcwd())
state_set = set()
city_set = set()
with open('database/background/citySet_with_states.txt','r') as f:
    city_set = f.read().strip().split('\n')
    for city in city_set:
        city_name = city.split('\t')[0]
        state_name = city.split('\t')[1]
        state_set.add(state_name)
        city_set.add(city_name)
        # write to new file
    f.close()
# with open('database/background/stateSet.txt', 'a') as f:
#     for state_name in state_set:
#         f.write(state_name.split('\\')[0] + '\n')
#     f.close()
with open('database/background/citySet_2.txt', 'a') as f:
    for city_name in city_set:
        f.write(city_name.split('\\')[0] + '\n')
    f.close()