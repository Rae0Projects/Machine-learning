# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 22:49:27 2021

@author: olatu
"""

Colours = ['Black','Pink','Blue','Red']
print(Colours)

# Printing specific elements
print(Colours[0])
print(Colours[0].title())

print(Colours[1].title())

print(Colours[-1].title())

# Curly brace
Age = {"Faye":21}
print(Age["Faye"])

# Using individual values from a list
Colours = ['Black','Pink','Blue','Red']
message = f" The two colours represented in Blackpink are {Colours[0],Colours[1].title()}."
print(message)

# Changing, adding and removing elements
Colours = ['Black','Pink','Blue','Red']
print(Colours)
Colours[0] = 'Green'
print(Colours)

# Appending elements to the end of a list:
Colours.append('Purple')
print(Colours)

Cities = []
Cities.append('Paris')
Cities.append('New York')
print(Cities)

# Inserting elements into a list
Colours = ['Black','Pink','Blue','Red']
Colours.insert(0,'Silver')
print(Colours)

# Removing an item with the del method
del Colours[1]
print(Colours)



    
    
    
    
