# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 20:34:30 2021

@author: olatu
"""

# Simple dictionary
print('\n')
Client_1 = {'Age':43, 'Nationality': 'Indian'}
print(Client_1['Age'])
print(Client_1['Nationality'])

# Working with dictionaries
print('\n')
Charlie = {'Colour':'Blue', 'Age':14}
print(Charlie)

# One key-value pair
Charlie = {'Colour':'Blue'}
print(Charlie)

# Accessing values in a dictionary
print('/n')
Charlie = {'Colour':'Blue', 'Age':14,'Occupation':'Doctor'}
User = Charlie['Occupation']
print(f"The users occupation is {User}")

# Adding new key-value pairs
print('/n')
Charlie = {'Colour':'Blue', 'Age':14,'Occupation':'Doctor'}
print(Charlie)
Charlie['Nationality'] = 'English'
Charlie['Height in cm'] = 184
print(Charlie)

# Beginning with an empty dictionary
print('\n')
Roseanne = {}
Roseanne['Eye colour'] = 'Brown'
Roseanne['Hair Colour'] = 'Blonde'
print(Roseanne)

# Modifying values in a dictionary
print('\n')
Jennie = {'Eye colour': 'Brown'}
print(Jennie)
print(f"Jennie's eye colour is {Jennie['Eye colour']}.")

Jennie['Eye colour'] = 'Green'
print(f"She now has {Jennie['Eye colour']} contact lenses.")

# Removing key-value pairs
print('\n')
Charlie = {'Colour':'Blue', 'Age':14,'Occupation':'Doctor'}
del Charlie['Occupation']
print(Charlie)

# Dictionary with similar objects
print('\n')
Favourite_fruit = {
    'Alice':'Apple',
    'Sarah':'Grapes',
    'Eliza':'Blueberries',
    'Millie':'Raspberries',
    }
favourite_fruit = Favourite_fruit['Alice'].title()
print(f"Alice's favourite fruit is an {favourite_fruit}.")
    
# Using get() to access values
print('\n')
Jennie = {'Eye colour': 'Brown', 'Height in cm': 163}
# print(Jennie['Nationality'])

Nationality_Ans = Jennie.get('Nationality','No answer was given')
print(Nationality_Ans)

# Looping through a dictionary
print('\n')
Favourite_fruit = {
    'Alice':'Apple',
    'Sarah':'Grapes',
    'Eliza':'Blueberries',
    'Millie':'Raspberries',
    }
for key, value in Favourite_fruit.items():
    print(f"\nKey: {key}")
    print(f"Value: {value}")
    
for k, v in Favourite_fruit.items():
    print(f"\nKey: {key}")
    print(f"Value: {value}")
    
# Looping through all the keys in a dictionary
print('\n')
Favourite_fruit = {
    'Alice':'Apple',
    'Sarah':'Grapes',
    'Eliza':'Blueberries',
    'Millie':'Raspberries',
    }
for fruits in Favourite_fruit.keys():
    print(fruits.title())
   for fruits in Favourite_fruit:
    print(fruits.title()) 

    


















