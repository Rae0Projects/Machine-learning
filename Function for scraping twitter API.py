#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tweepy


# In[4]:


# Add your API keys and tokens here
consumer_key = 'YOUR_CONSUMER_KEY'
consumer_secret = 'YOUR_CONSUMER_SECRET'
access_token = 'YOUR_ACCESS_TOKEN'
access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'

# Authenticate with the Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# Define the username of the user whose timeline you want to scrape
username = 'twitter_username'

# Fetch tweets from the user's timeline
tweets = api.user_timeline(screen_name=username, count=10, tweet_mode='extended')

# Process and print fetched tweets
for tweet in tweets:
    print(f"Username: {tweet.user.screen_name}")
    print(f"Tweet: {tweet.full_text}")
    print("-" * 40)


# In[ ]:




