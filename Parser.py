file = open("Coverage.txt" , 'r')
count = 1
for line in file:
    line = line.rstrip()
    warr_group, line = line.split('-', 1)
    prod_codes, line = line.split('_', 1)
    
    if (line.count(',') > 0):
        level, veh_class = line.split(',', 1)
    else:
        level = line
        veh_class = 'NULL'
        
    if (level.count('/') > 0): level, discard = level.split('/', 1) 
    
    if (level.count('\\') > 0): # Need to escape \ in Python 
        level, add_class = level.split('\\', 1)
        if (veh_class == 'NULL'):
            veh_class = add_class
        else: veh_class += add_class

    while (prod_codes.count(',') > 0):
        prod_code, prod_codes = prod_codes.split(',', 1)
        print(str(count) + ' ' + warr_group + ' ' + prod_code + ' ' + level + ' ' + veh_class)        
    print(str(count) + ' ' + warr_group + ' ' + prod_codes + ' ' + level + ' ' + veh_class)

    count += 1
file.close()