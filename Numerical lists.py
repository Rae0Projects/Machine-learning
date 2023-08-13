# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 19:39:53 2021

@author: olatu
"""

# Range to print a series of numbers
for value in range(3,8):
    print(value)
print("\n")
    
for value in range(3,9):
    print(value)
print("\n")

for value in range(5):
    print(value)
    
# Using Range() and list() to make a list of numbers
print("\n")
Numbers = list(range(2,8))
print(Numbers)

# Choosing step size and range() when generating numbers
print("\n")
odd_numbers = list(range(1,12,2))
print(odd_numbers)

# Creating sets of numbers using the range() function
print("\n")
Squares = []
for number in range(1,12):
    square = number ** 2
    Squares.append(square)
print(Squares)

Squares = []
for number in range(1,12):
    Squares.append(number**2)
print(Squares)
    
# Statistics
print("\n")
values = [2,4,6,8,10,12,14,16,18,20]
print(min(values))
print(max(values))
print(sum(values))

# List comprehension
squares = [value**2 for value in range(2,12)]
print(squares)

# Slicing a list
print("\n")
Blackpink_members = ['Jisoo', 'Jennie', 'Rosé', 'Lisa']
print(Blackpink_members[0:3])
print(Blackpink_members[3:])
print(Blackpink_members[-2:]) 
print("Here are the members with their own solo:")
for solo in Blackpink_members[1:3]:
    print(solo.title())

# Copying a list
print("\n")
Blackpink_members = ['Jisoo', 'Jennie', 'Rosé', 'Lisa']
BP_Members = Blackpink_members[:]
print("The members of BlackPink are:")
print(Blackpink_members)
print("New list of members copied:")
print(BP_Members)
Blackpink_members.append('Blinks')
BP_Members.append('Hello')
print(Blackpink_members)
print(BP_Members)

# Tuples
print("\n")
dimensions = (100,30)
print(dimensions[0])
print(dimensions[1])
for dimension in dimensions:
    print(dimension)
dimensions = (200,80)
print("\nModified dimensions:")
for dimension in dimensions:
    print(dimension)




    


    
    
      
