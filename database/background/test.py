
f = open('/home/xj/toolAugEnv/code/toolConstraint/database/background/citySet.txt','r').read().strip().split('\n')
citySet = []
for line in f:
    if line not in citySet:
        citySet.append(line.strip())
    else:
        print(line)