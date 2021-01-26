import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import re
import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from API_KEYS import api_key, api_secret_key, access_token, access_token_secret


# Authentication
auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Search parameters
keyword = "Cyberpunk"
noOfTweet = 50

# Set the end date as X amount of days before today
today = datetime.date.today()
end = today - datetime.timedelta(days = 7)
end_verbose = end.strftime('%A') + ', ' + end.strftime("%B") + ' ' + end.strftime("%d") + ', ' + end.strftime("%Y")

# Search for tweets
tweets = tweepy.Cursor(api.search, q = keyword, until = end).items(noOfTweet)

# Get the text values of the tweets for later analysis
tweet_list = []
for tweet in tweets:
    tweet_list.append(tweet.text)    

# Remove duplicates
tweet_list = pd.DataFrame(tweet_list)
tweet_list.drop_duplicates(inplace = True)

tw_list = pd.DataFrame(tweet_list)
tw_list["text"] = tw_list[0]

# Clean the text data of the tweets (retweets, punctuation etc)
remove_rt = lambda x: re.sub('RT @\w+: '," ",x)
rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x)
tw_list["text"] = tw_list.text.map(remove_rt).map(rt)
tw_list["text"] = tw_list.text.str.lower()

# Do sentiment analysis on the tweets
for index, row in tw_list['text'].iteritems():
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    if neg > pos:
        tw_list.loc[index, 'sentiment'] = "negative"
    elif pos > neg:
        tw_list.loc[index, 'sentiment'] = "positive"
    else:
        tw_list.loc[index, 'sentiment'] = "neutral"
    tw_list.loc[index, 'neg'] = neg
    tw_list.loc[index, 'neu'] = neu
    tw_list.loc[index, 'pos'] = pos
    tw_list.loc[index, 'compound'] = comp


# Create new data frames for all sentiments
tw_list_negative = tw_list[tw_list["sentiment"] == "negative"]
tw_list_positive = tw_list[tw_list["sentiment"] == "positive"]
tw_list_neutral = tw_list[tw_list["sentiment"] == "neutral"]


# Function to count the values in the data frame
def count_values_in_column(data, feature):
    total = data.loc[:,feature].value_counts(dropna = False)
    percentage = round(data.loc[:,feature].value_counts(dropna = False, normalize = True) * 100, 2)
    return pd.concat([total,percentage], axis = 1, keys = ['Total', 'Percentage'])


# Create the data for the pie chart
piechart = count_values_in_column(tw_list, "sentiment")
names = piechart.index
size = piechart["Percentage"]

# Create  the pie chart with an empty middle area
my_circle = plt.Circle( (0,0), 0.7, color = 'white')
plt.pie(size, labels = names, colors=['green', 'blue', 'red'])
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.title("Sentiment Analysis for " + keyword + " on " + end_verbose)
plt.show()

# Print the numerical data to the console
print("Sentiment Analysis for " + keyword + " on " + end_verbose)
print("total number: ", len(tw_list))
print("positive number: ", len(tw_list_positive))
print("negative number: ", len(tw_list_negative))
print("neutral number: ", len(tw_list_neutral))
