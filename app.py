import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
import csv
import tweepy
import nltk

# initialize api instance
consumer_key='oZxkH1UMtNKedU8j33x5uXKVM'
consumer_secret='cqS0dpdjz4SM3z4kgScso7xXt843K0WQcD59WFgsb9UG9XBqo0'
access_token='1635271567-2LLevLP40482bAK43vFdej6SOSlP5qPW8gZ90Jq'
access_token_secret='maEDVrBh0cBqnJN0urWb8pyFYlp1eMjzofi7d0BH75of6'

# test authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
twitter_api = tweepy.API(auth,wait_on_rate_limit =True,wait_on_rate_limit_notify= True)
#print(twitter_api.verify_credentials())

def buildTestSet(search_keyword):
    try:
        tweets_fetched = twitter_api.search(search_keyword, count=100, lang="en")
        #print(tweets_fetched)
        print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + search_keyword)
        return [(status.text) for status in tweets_fetched]
    except:
        print("Unfortunately, something went wrong..")
        return None

search_term = input("Enter a search keyword: ")
testDataSet = buildTestSet(search_term)

trainingData=[]
with open("/train.csv",'r') as csvfile:
    lineReader = csv.reader(csvfile, delimiter=',', quotechar="\"")
    for row in lineReader:
            trainingData.append({"text":row[2],"label":row[1]})


class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL'])

    def processTweets(self, list_of_tweets):
        processedTweets = []
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet["text"]), tweet["label"]))
        return processedTweets

    def processTweets1(self, list_of_tweets):
        processedTweets = []
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet)))
        return processedTweets

    def _processTweet(self, tweet):
        tweet = tweet.lower()  # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)  # remove URLs
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet)  # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  # remove the # in #hashtag
        tweet = word_tokenize(tweet)  # remove repeated characters (helloooooooo into hello)
        return [word for word in tweet if word not in self._stopwords]

tweetProcessor = PreProcessTweets()
preprocessedTrainingSet = tweetProcessor.processTweets(trainingData)
preprocessedTestSet = tweetProcessor.processTweets1(testDataSet)


def buildVocabulary(preprocessedTrainingData):
    all_words = []

    for (words, sentiment) in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()

    return word_features

def extract_features(tweet):
    tweet_words=set(tweet)
    features={}
    for word in word_features:
        features['contains(%s)' % word]=(word in tweet_words)
    return features

# Now we can extract the features and train the classifier
word_features = buildVocabulary(preprocessedTrainingSet)
trainingFeatures=nltk.classify.apply_features(extract_features,preprocessedTrainingSet[:50000])

testing_set=nltk.classify.apply_features(extract_features,preprocessedTrainingSet[50000:])

NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)

NBResultLabels = [NBayesClassifier.classify(extract_features(tweet)) for tweet in preprocessedTestSet]
print(NBResultLabels)

print("Classifier accuracy percent:",(nltk.classify.accuracy(NBayesClassifier, testing_set))*100)

NBResultLabels = [NBayesClassifier.classify(extract_features(tweet)) for tweet in preprocessedTestSet]
print(NBResultLabels)

print("Classifier accuracy percent:",(nltk.classify.accuracy(NBayesClassifier, testing_set))*100)