
# coding: utf-8

# In[1]:

import math as m
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

#Define dictionary for x, f(x) values
starting_values = {
    0.4 : [0.3894],
    0.5 : [0.4794],
    0.6 : [0.5646],
    0.7 : [0.6442],
    0.8 : [0.7174],
}

#Let h = 0.1
h=0.1
length = len(starting_values)

#If  length >= position in dictionary > 0, BD
#If  length > position in dictionary >= 0, FD
#If length > position in dictionary > 0, CD
#Note, value[0] = f(x)
keyList=sorted(starting_values.keys())
for index, (key, value) in enumerate(starting_values.items()):
    prev=starting_values[keyList[index-1]]
    curr=starting_values[keyList[index]]
    
    
    if (index >= 0 and index < length-1): #add forward difference
        next=starting_values[keyList[index+1]]
        value.append((next[0]-curr[0])/h)
    if index == 0: #If index = 0, zero for backward difference
        value.append(0)
    if index > 0: #add the backward difference if index > 0
        value.append((curr[0] - prev[0])/h)
    if (index > 0 and index < length-1):#add central difference
        next=starting_values[keyList[index+1]]
        value.append((next[0]-prev[0])/(2))
    if index == length-1: #can't do forward/central difference on last item
        value.append(0)
        value.append(0)
    if index == 0: #can't do central difference on first item
        value.append(0)
    #print(value)

print("x|f(x)|f'(x)(FD)|f'(x)(BD)|f'(x)(CD)")
for index, (key, value) in enumerate(starting_values.items()):
    print(key,value)
    






# In[ ]:




# In[76]:




# In[ ]:




# In[ ]:



