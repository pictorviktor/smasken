from itertools import permutations
def attribfunc():
    combination = []
    attributes = ['Number words female','Total words','Number of words lead',
    'Difference in words lead and co-lead','Number of male actors',
    'Year','Number of female actors','Number words male','Gross',
    'Mean Age Male','Mean Age Female','Age Lead','Age Co-Lead']
    # combination.append(permutations(attributes,r=4))
    # print(combination)
    print(permutations(attributes,r=4))

combination = []
print(combination)
numbers = [1,2,3,4,5,6,7,8,9]
attributes = ['Number words female','Total words','Number of words lead',
'Difference in words lead and co-lead','Number of male actors',
'Year','Number of female actors','Number words male','Gross',
'Mean Age Male','Mean Age Female','Age Lead','Age Co-Lead']
#print(len(attributes))
for i in range(1,13):
    
    attributes.pop()
    combination.append(attributes.copy())
    
    
    
    
print(combination)
        



