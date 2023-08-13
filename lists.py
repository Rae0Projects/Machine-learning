# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 21:37:00 2021

@author: olatu
"""

# Removing an item using the pop() method
BlackPink_Songs = ['Boombayah','Whistle','Playing With Fire','Stay','As If It’s Your Last']
print(BlackPink_Songs)

Popped_BP_Songs = BlackPink_Songs.pop()
print(BlackPink_Songs)
print(f"I enjoyed listening to {Popped_BP_Songs.title()}.")


# Popping items from any position in a list
First_Song_I_Listened_To = BlackPink_Songs.pop(0)
print(f"The first ever song of blackpink that I listened to was {First_Song_I_Listened_To.title()}.")

# Removing an item by value
Blackpink_members = ['Jisoo', 'Jennie', 'Rosé', 'Lisa','Blinks']
print(Blackpink_members)
Blackpink_members.remove('Blinks')
print(Blackpink_members)
# Explaning removal of item from list
Blackpink_members = ['Jisoo', 'Jennie', 'Rosé', 'Lisa','Blinks']
print(Blackpink_members)
Name_BP_Fanbase = 'Blinks'
Blackpink_members.remove(Name_BP_Fanbase)
print(Blackpink_members)
print(f"\n {Name_BP_Fanbase.title()} is actually the name of their fanbase.")

# Sorting a list prmanently in alphabetical order with sort() method
Blackpink_members = ['Jisoo', 'Jennie', 'Rosé', 'Lisa','Blinks']
print(Blackpink_members)
Blackpink_members.remove('Blinks')
print(Blackpink_members)
BP_Members = Blackpink_members
print(BP_Members)
BP_Members.sort()
print(BP_Members)
# Reverse alphabetical order sort() method (permenant) 
BP_Members.sort(reverse=True)
print(BP_Members)

# Sorting a list temporarily with the sorted() function
Other_BP_Songs = ['How you like that','Lovesick girls', 'Pretty savage','As if it\'s your last']
print("Here are some other songs by BlackPink:")
print(Other_BP_Songs)
print("\nHere is the sorted list:")
print(sorted(Other_BP_Songs))
print("\nHere is the original list again:*")
print(Other_BP_Songs)

# Printing a list in reverse order
BP_Members = Blackpink_members
print(BP_Members)
BP_Members.reverse()
print(BP_Members)

# Finding the length of a list
How_Many_Members = len(BP_Members)
print(How_Many_Members)






