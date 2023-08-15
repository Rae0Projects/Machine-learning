#!/usr/bin/env python
# coding: utf-8

# In[48]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[63]:


import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


# Establish a connection to SQLite database
Soccer_db_path = 'C:\\Users\\olatu\\Documents\\SQL\\soccer_database.sqlite'
connection = sqlite3.connect(Soccer_db_path)


# In[3]:


cursor=connection.cursor()


# In[6]:


# printing all the tables present in database.sqlite 
for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';"):
    print(list(row))


# In[8]:


# Execute SQL queries and load data into a pandas DataFrame
Soccer_dataframe=pd.read_sql_query('''SELECT *
                                FROM Match
                                LIMIT 10;''', connection)
Soccer_dataframe


# In[11]:


# Create DataFrames of the available tables

DF_PlayerAttribute = pd.read_sql("Select * from Player_Attributes",connection)

DF_player = pd.read_sql("Select * from Player",connection)

DF_Match = pd.read_sql("Select * from Match",connection)

DF_League = pd.read_sql("Select * from League",connection)

DF_Country = pd.read_sql("Select * from Country",connection)

DF_Team = pd.read_sql("Select * from Team",connection)

DF_TeamAttributes = pd.read_sql("Select * from Team_Attributes",connection)


# In[20]:


# Identifying missing values from the dataset


# In[21]:


# The number of teams per country
query = pd.read_sql_query(
    '''
        SELECT 
              c.name AS Country,
              COUNT(DISTINCT(team_long_name)) AS 'No. of Teams'
              FROM Match AS m
              LEFT JOIN Country AS c
              ON m.country_id = c.id
              LEFT JOIN Team AS t 
              ON m.home_team_api_id = t.team_api_id
              GROUP BY Country
    ''', connection
)
query


# In[23]:


# Home teams number of scored goals per country for each season
Scored_homeGoals = pd.read_sql_query(
    '''
        SELECT c.name AS Country,
               m.season AS Season,
               SUM(m.home_team_goal) AS 'Home Goal',
               SUM(m.away_team_goal) AS 'Away Goal'
        FROM Match as m 
        LEFT JOIN country AS c
        ON m.country_id = c.id
        GROUP BY Country, Season    
        ORDER BY Country     
    ''', connection
)
Scored_homeGoals


# In[25]:


# The number of goals scored per team for each season
Goals_perTeamSeason = pd.read_sql_query(
    '''
        SELECT t.team_long_name AS Team, 
               m.season as Season,
               SUM(m.home_team_goal) AS 'Home Goal',
               SUM(m.away_team_goal) AS 'Away Goal'
        FROM Match AS m
        LEFT JOIN Team AS t
        ON m.home_team_api_id = t.team_api_id 
        GROUP BY Team, Season
        ORDER BY Team
    ''', connection
)
Goals_perTeamSeason


# In[26]:


Goals_perTeamSeason2 = pd.read_sql_query(
    '''
        SELECT c.name AS Country,
               t.team_long_name AS Team,
               m.season AS Season,
               SUM(m.home_team_goal) AS 'Home Goal',
               SUM(m.away_team_goal) AS 'Away Goal'
        FROM Match as m
        LEFT JOIN Country AS c
        ON m.country_id = c.id
        LEFT JOIN Team AS t
        ON m.home_team_api_id = t.team_api_id
        GROUP BY Country, Team, Season
        ORDER BY Country
    ''', connection
)
Goals_perTeamSeason2


# In[27]:


# Overall number of goals scored at home per country by each team
Total_Homegoals = pd.read_sql_query(
    '''
        SELECT name AS Name,
            team_long_name AS Team,
            --STRFTIME('%Y', date) AS Year, 
            SUM(home_team_goal) AS Goal
        FROM Match AS m
        LEFT JOIN Country as c
        ON m.country_id = c.id
        LEFT JOIN Team AS t
        ON m.home_team_api_id = t.team_api_id  
        GROUP BY Name, Team
        ORDER BY Goal DESC
        
    ''', connection
)
Total_Homegoals


# In[29]:


# Dataframe for Week, Month, Year and day
Time_df = pd.read_sql_query(
    '''
        SELECT date AS Date,
               STRFTIME('%Y', date) AS Year,
               STRFTIME('%m', date) AS Month,
               STRFTIME('%w', date) AS Week,
               STRFTIME('%d', date) AS Day
        FROM Match
    ''', connection
)
Time_df


# In[30]:


# The number of matches won, tied and lost for each country
Score = pd.read_sql_query(
    '''
    WITH sub_q AS (
        SELECT     
                   c.name AS Country,
                   season AS Season,
                   t.team_long_name AS Team,
                   --m.home_team_goal AS home_goal,
                   --m.away_team_goal AS away_goal,
                   COUNT(CASE WHEN m.home_team_goal > away_team_goal THEN 'Win' END) AS Won,
                   COUNT(CASE WHEN m.home_team_goal < away_team_goal THEN 'Lost' END) AS Lost,
                   COUNT(CASE WHEN m.home_team_goal = away_team_goal THEN 'Draw' END) AS Draw
         FROM Match AS m
         LEFT JOIN Country AS c
         ON m.country_id = c.id
         LEFT JOIN Team as t
         ON m.home_team_api_id = t.team_api_id
         GROUP BY Country, Season
         ORDER BY Country
         )
         SELECT ROW_NUMBER() OVER(ORDER BY Won DESC) AS 'Row Number',
                Country, 
                Season,
                Won, 
                Lost,
                Draw
        FROM sub_q
    ''', connection, index_col='Row Number'
    )
Score


# In[35]:


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


# In[36]:


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


# In[49]:


# Calculate Average Goals per League

TeamGoalLoss = SoccerJoin.groupby(["Team", "League"])["Goals_Conceded"].sum().reset_index()
TeamGoalLoss=TeamGoalLoss.sort_values(by="Goals_Conceded", ascending=False).reset_index(drop=True) 

#Plot average goals scored per league
sns.set_style('white')
a=sns.catplot(kind = 'bar', x = 'Goals_Conceded', y = 'League', data = TeamGoalLoss, edgecolor='k')
a.fig.set_size_inches(12,8)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Average Goals Conceded', fontsize=15)
plt.ylabel('League', fontsize = 15)
plt.xlim(0,4)
plt.title('Avg. Goals conceded per League', fontsize=20)
for i in range(len(TeamGoalLoss)):
    plt.text(TeamGoalLoss['Goals_Conceded'].iloc[i]+0.15,i+0.15, round(TeamGoalLoss['Goals_Conceded'].iloc[i],2), fontsize = 12)
    
    


# In[61]:


#
plt.figure(figsize=(10,6))
sns.set_style('whitegrid')
sns.regplot(x='Losses',y='Goals_Conceded',data=SoccerJoin,scatter_kws={'s':10})
plt.title('Goals Conceded vs Losses')
plt.xlabel('Losses')
plt.ylabel('Goals Conceded')
plt.show()


# In[66]:


# Read Messi's Data from the database
messi = pd.read_sql("""SELECT player_name,
                                  date,overall_rating,
                                  attacking_work_rate,
                                  crossing,
                                  finishing,
                                  shot_power,
                                  heading_accuracy,
                                  free_kick_accuracy,
                                  sprint_speed,
                                  dribbling,
                                  agility
                                    
                      FROM Player 
                      LEFT JOIN Player_Attributes
                      ON Player.player_api_id = Player_Attributes.player_api_id
                    
                      WHERE player_name = 'Lionel Messi'
                        
                      ORDER by date
                      """, connection)
# Covert date column
messi["date"] = pd.to_datetime(messi["date"])


# In[67]:


messi.head()


# In[73]:


cris = pd.read_sql("""SELECT player_name,
                                  date,overall_rating,
                                  attacking_work_rate,
                                  crossing,
                                  finishing,
                                  shot_power,
                                  heading_accuracy,
                                  free_kick_accuracy,
                                  sprint_speed,
                                  dribbling,
                                  agility
                                    
                      FROM Player 
                      LEFT JOIN Player_Attributes
                      ON Player.player_api_id = Player_Attributes.player_api_id
                    
                      WHERE player_name = 'Cristiano Ronaldo'
                        
                      ORDER by date
                      """, connection)


# In[74]:


cris.head()


# In[ ]:





# In[ ]:








# In[ ]:





# In[ ]:




