# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:51:16 2021

@author: olatu
"""
# if statement
Directions = ['north','south','east','west','ne']
for direction in Directions:
    if direction == 'ne':
        print(direction.upper())
    else:
        print(direction.title())
        
# Conditional tests
direction = 'NE'
print(direction == 'NE')
print(direction == 'NW')

# Ignoring case 
print('\n')
direction = 'NE'
print(direction == 'ne')
print(direction.lower()== 'ne')

# Checking for inequality
print('\n')
Ice_cream = 'strawberry'

if Ice_cream != 'chocolate':
    print("I want strawberry!")
    
# Numerical comparison
print('\n')
Height = 160
print(Height < 168)
print(Height <= 168)
print(Height>168)
print(Height>=168)

# Checking multiple conditions
print('\n')
Height_M = 168
Height_R = 170
print(Height_M >= 167 and Height_R >= 173)
print(Height_M >= 160 or Height_R >= 160)
print(Height_M >= 173 or Height_R >= 173)

# Checking if a value is in a list
print('\n')
Ice_cream_Flavours = ['Strawberry','Chocolate','Vanilla','Caramel','Cookie Dough']
print('Vanilla' in Ice_cream_Flavours)
print('Bubblegum' in Ice_cream_Flavours)

# Checking if a value is not in a list
print('\n')
Ice_cream_Flavours = ['Strawberry','Chocolate','Vanilla','Caramel','Cookie Dough']
Flavour = 'Bubblegum'

if Flavour not in Ice_cream_Flavours:
    print(f"{Flavour.title()}, is not in stock.")
    
# Boolean expressions
print('\n')
a = True
b = False
c = 0
print(bool(c))
d = 2
print(bool(d))

# Basic If statements
print('\n')
Score = 70
if Score >= 69:
    print("Congratulations you've passed!")
    print("Would you like to try again for a higher mark?")

# if-else statement
print('\n')
Score = 70
if Score >= 69:
    print("Congratulations you've passed!")
    print("Would you like to try again for a higher mark?")
else:
    print( "Unfortunately you failed")
    print(" Are you taking extra class?")
    
# if-else chain
print('\n')
Grade = 70

if Grade < 40:
    Mark = ('fail')
elif Grade < 50:
    Mark =(' Pass')
elif Grade <  60:
     Mark = (' Good pass')
else:
    Mark = ('Success')
    
print(f"Your overall results are {Mark}.")

# Testing Multiple conditions
print('\n')
Ice_cream_Flavours = ['Strawberry','Chocolate']
if 'Strawberry' in Ice_cream_Flavours:
    print("Strawberry is added")
if'Chocolate' in Ice_cream_Flavours:
    print("Chocolate is added")
if 'Vanilla' in Ice_cream_Flavours:
    print('Adding Vanilla')
print("\nHere is your ice cream!")

print('\n')
Ice_cream_Flavours = ['Strawberry','Chocolate']
if 'Strawberry' in Ice_cream_Flavours:
    print("Strawberry is added")
elif'Chocolate' in Ice_cream_Flavours:
    print("Chocolate is added")
elif 'Vanilla' in Ice_cream_Flavours:
    print('Adding Vanilla')
print("\nHere is your ice cream!")

# Checking for special items
print('\n')
Ice_cream_Flavours = ['Strawberry','Chocolate','Vanilla']
for Ice_cream_flavour in Ice_cream_Flavours:
    if Ice_cream_flavour == 'Strawberry':
        print("Sorry, the strawberry is out of stock.")
    else:
        print(f"Adding {Ice_cream_flavour}.")
print("Here is your ice cream, enjoy!")

# Checking that a list is not empty
print('\n')
Drink_list = []
if Drink_list:
    for drink_list in Drink_list:
        print(f"{Drink_list} has been added to your cart.")
        print("\nBasket ready for check out")
    else:
        print("Your basket is empty")
        

















