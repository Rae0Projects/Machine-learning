#!/usr/bin/env python
# coding: utf-8

# In[23]:


import sqlite3
import pandas as pd
import matplotlib.pyplot as plt


# In[8]:


# Establish a connection to SQLite database
Soccer_db_path = 'C:\\Users\\olatu\\Documents\\SQL\\soccer_database.sqlite'
connection = sqlite3.connect(Soccer_db_path)


# In[11]:


cursor=connection.cursor()


# In[12]:


# Execute SQL queries and load data into a pandas DataFrame
Soccer_dataframe=pd.read_sql_query('''SELECT *
                                FROM Match
                                LIMIT 10;''', connection)
Soccer_dataframe


# In[13]:


# Selecting specific variable names of the columns from the database
countries=cursor.execute('''SELECT COUNT(DISTINCT name)
                                FROM League;''').fetchall()
leagues=cursor.execute('''SELECT COUNT(DISTINCT name)
                            FROM League;''').fetchall()
teams=cursor.execute('''SELECT COUNT(DISTINCT team_long_name)
                            FROM Team;''').fetchall()
players=cursor.execute('''SELECT COUNT(DISTINCT player_name)
                            FROM Player;''').fetchall()


# In[15]:


# Printing table count individually.

print("Countries: ", countries[0][0])
print("Leagues: ", leagues[0][0])
print("Teams: ", teams[0][0])
print("Players: ", players[0][0])


# In[16]:


cursor.execute("BEGIN TRANSACTION;")    #   Beginning a transaction to improve the execution time of the queries.

#   Creating corresponding columns in the table to store the names of the leagues and countries.

cursor.execute('''ALTER TABLE Match
                    ADD COLUMN "Leagues" TEXT;''')
cursor.execute('''ALTER TABLE Match
                    ADD COLUMN "Countries" TEXT;''')
cursor.execute('''ALTER TABLE Match
                    ADD COLUMN "Home_Team" TEXT;''')
cursor.execute('''ALTER TABLE Match
                    ADD COLUMN "Away_Team" TEXT;''')


# In[17]:


#   Updating the corresponding columns with the names of the leagues and countries after joining the tables.

cursor.execute('''UPDATE Match
                    SET Leagues=(
                        SELECT League.name
                        FROM League
                        WHERE Match.league_id=League.id),
                    Countries=(
                        SELECT Country.name
                        FROM Country
                        WHERE Match.country_id=Country.id),
                    Home_Team=(
                        SELECT Team.team_long_name
                        FROM Team
                        WHERE Match.home_team_api_id=Team.team_api_id),
                    Away_Team=(
                        SELECT Team.team_long_name
                        FROM Team
                        WHERE Match.away_team_api_id=Team.team_api_id)
                    WHERE league_id IN (
                        SELECT id
                        FROM League)
                    AND country_id IS NOT NULL
                    AND home_team_api_id IN (
                        SELECT team_api_id
                        FROM Team)
                    AND away_team_api_id IN (
                        SELECT team_api_id
                        FROM Team);''')

cursor.execute("END TRANSACTION;")  #   Ending the transaction.


# In[19]:


SoccerJoin=pd.read_sql_query('''SELECT *
                                FROM Match
                                    LIMIT 10;''', connection)
SoccerJoin


# In[20]:


# Wins, losses, number of goals scored and lost for each team
SoccerJoin=pd.read_sql_query('''SELECT Home_Team AS "Team",
                                season AS "Season",
                                Leagues AS "League",
                                COUNT(*) AS "Matches_Played",
                                SUM(home_team_goal) AS "Goals_Scored",
                                SUM(away_team_goal) AS "Goals_Conceded",
                                SUM(
                                    CASE WHEN home_team_goal>away_team_goal
                                    THEN 1
                                    ELSE 0
                                    END) AS "Wins",
                                SUM(
                                        CASE WHEN home_team_goal=away_team_goal
                                        THEN 1
                                        ELSE 0
                                        END) AS "Draws",
                                SUM(
                                        CASE WHEN home_team_goal<away_team_goal
                                        THEN 1
                                        ELSE 0
                                        END) AS "Losses",
                                SUM(
                                        CASE WHEN home_team_goal>away_team_goal
                                        THEN 1
                                        ELSE 0
                                        END)*100/COUNT(*) AS "Win_Percentage"
                                FROM Match
                                GROUP BY Team, season, Leagues
                                UNION
                                SELECT Away_Team AS "Team",
                                season AS "Season",
                                Leagues AS "League",
                                COUNT(*) AS "Matches_Played",
                                SUM(away_team_goal) AS "Goals_Scored",
                                SUM(home_team_goal) AS "Goals_Conceded",
                                SUM(
                                        CASE WHEN away_team_goal>home_team_goal
                                        THEN 1
                                        ELSE 0
                                        END) AS "Wins",
                                SUM(
                                        CASE WHEN away_team_goal=home_team_goal
                                        THEN 1
                                        ELSE 0
                                        END) AS "Draws",
                                SUM(
                                        CASE WHEN away_team_goal<home_team_goal
                                        THEN 1
                                        ELSE 0
                                        END) AS "Losses",
                                SUM(
                                        CASE WHEN away_team_goal>home_team_goal
                                        THEN 1
                                        ELSE 0
                                        END)*100/COUNT() AS "Win_Percentage"
                                FROM Match
                                GROUP BY Team, season, Leagues
                                ORDER BY Win_Percentage
                                DESC;''', connection)
SoccerJoin


# In[25]:


# Top ten team wins across all leagues and seasons 
wins=SoccerJoin.groupby(["Team", "League"])["Wins"].sum().reset_index()  #   Grouping the pandas.DataFrame by team and league, and summing the wins.
wins=wins.sort_values(by="Wins", ascending=False).reset_index(drop=True)    #   Sorting the pandas.DataFrame by the wins in descending order.

#   Plotting a bar graph for the top ten teams with the most wins across all seasons and leagues.

plt.figure(figsize=(10, 5))
plt.bar(wins["Team"][:10], wins["Wins"][:10], color="#A133FF")
plt.bar(wins["Team"][0], wins["Wins"][0], color="#FFF833")
plt.xticks(rotation=90)
plt.xlabel("Football Teams")
plt.ylabel("Aggregate Wins")
plt.title("Football Teams With the Most Aggregated Wins in Europe (2008 ‒ 2016)")
for i in range(10):
    plt.text(x=i, y=wins["Wins"][i]+3, s=wins["Wins"][i], ha="center")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




