# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Variables
message = "Hello Python world!"
print("Hello world")

# Strings
"This is a string"
'This is also a string'
'I told my friend,"Python is my favourite language!"'
"The language 'Python' is named after Monty Python, not the snake."

# Changing case in a string
name = "jennie kim"
print(name.title())
print(name.upper())
print(name.lower())

# Using variables in strings
first_name = "BLACKPINK"
last_name = " IN YOUR AREA"
full_name = f"{first_name}{last_name}"
print(full_name)

# Using variables in strings part 2
First_name = "Ho-"
Last_name = "How you like that?"
Full_Name = f"{First_name}{Last_name}"
message = f"Blinks!,{Full_Name.title()}!"
print(message)

# Adding whitespace to strings with Tabs or Newlines
print (" Blackpink is the revolution ")
print("\t Blackpink is the revolution")
print("Positions:\n\t Vocalist\n\t Dancer \n\t Visuals \n\t Rapper")

# Stripping whitespace
On_The_Ground = "             Everything I need is on the ground                      "
#remove spaces from right
print(On_The_Ground.rstrip())
#remove spaces from left
print(On_The_Ground.lstrip())
#remove spaces from both side
print(On_The_Ground.strip())






