import csv

reader = csv.reader(open("C:\\Users\\HUY\\Downloads\\results.csv"))
lines = list(reader)
for line in lines:
    v = line[1:]
    print(v)
    v_mean = sum(v)/len(v)
    v_var = sum([(x-v_mean)**2 for x in v])/(len(v)-1)
    v_std = v_var**0.5
    line.extend([v_mean, v_var, v_std, f'{v_mean}±{v_std}'])

writer = csv.writer(open('results.csv', 'w'))
writer.writerows(lines)