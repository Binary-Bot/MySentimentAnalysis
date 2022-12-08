import re
import string
import numpy as np
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt



class MySentimentModel:
    def __init__(self):
        self.__trainX, self.__trainY, self.__testX, self.__testY = self.__getTrainAndTestData()
        self.__freqDict = self.__populateFrequencies()
        self.__matrix = self.__createMatrix()
        self.__J, self.__theta = self.__gradientDescent_algo(1e-9, 1500)

    # returns training and testing data
    @staticmethod
    def __getTrainAndTestData():
        positiveTweets = twitter_samples.strings('positive_tweets.json')
        negativeTweets = twitter_samples.strings('negative_tweets.json')

        trainPos = positiveTweets[:4000]
        trainNeg = negativeTweets[:4000]
        testPos = positiveTweets[4000:]
        testNeg = negativeTweets[4000:]

        trainX = trainPos + trainNeg
        testX = testPos + testNeg
        trainY = np.append(np.ones((len(trainPos), 1)), np.zeros((len(trainNeg), 1)), axis=0)
        testY = np.append(np.ones((len(testPos), 1)), np.zeros((len(testNeg), 1)), axis=0)
        return trainX, trainY, testX, testY

    # Cleaning, tokenizing and stemming the data
    def __processText(self, text):
        text = re.sub(r'^RT[\s]+', '', text)
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
        text = re.sub(r'@\S*', '', text)
        tokenizedText = TweetTokenizer().tokenize(text)
        stopWords = stopwords.words('english')
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in tokenizedText if word not in stopWords and
                         word not in string.punctuation]

    def __populateFrequencies(self):
        # Gets all the positive words
        posWords = [word for sentence in self.__trainX[:4000] for word in self.__processText(sentence)]
        posFreq = {}
        for word in posWords:
            if (word, 1) not in posFreq:
                posFreq[(word, 1)] = 1
            else:
                posFreq[(word, 1)] = posFreq[(word, 1)] + 1
        # Gets all the negative words
        negWords = [word for sentence in self.__trainX[4000:] for word in self.__processText(sentence)]
        negFreq = {}
        for word in negWords:
            if (word, 0) not in negFreq:
                negFreq[(word, 0)] = 1
            else:
                negFreq[(word, 0)] = negFreq[(word, 0)] + 1

        frequencies = dict(posFreq)
        frequencies.update(negFreq)
        return frequencies


    def __features_extraction(self, text):
        word_l = self.__processText(text)
        x = np.zeros((1, 3))
        x[0,0] = 1
        for word in word_l:
            try:
                x[0,1] += self.__freqDict[(word, 1)]
            except:
                x[0,1] += 0
            try:
                x[0,2] += self.__freqDict[(word, 0.0)]
            except:
                x[0,2] += 0
        assert(x.shape == (1, 3))
        return x

    @staticmethod
    def __sigmoid(x):
        h = 1/(1+np.exp(-x))
        return h

    def __gradientDescent_algo(self, alpha, num_iters):
        theta = np.zeros((3, 1))
        m = self.__matrix.shape[0]
        for i in range(0, num_iters):
            z = np.dot(self.__matrix, theta)
            h = self.__sigmoid(z)
            J = -1/m*(np.dot(self.__trainY.T,np.log(h))+np.dot((1-self.__trainY).T,np.log(1-h)))
            theta = theta-(alpha/m)*np.dot(self.__matrix.T, h - self.__trainY)
        return float(J), theta

    def __createMatrix(self):
        X = np.zeros((len(self.__trainX), 3))
        for i in range(len(self.__trainX)):
            X[i, :] = self.__features_extraction(self.__trainX[i])
        return X

    def compareWithNLTK(self):
        sia = SentimentIntensityAnalyzer()
        myScores = []
        nltkScores = []
        for tweet in self.__testX:
            yPred = self.sentimentAnalysis(tweet)
            myScores.append(yPred[0])
            nltkScores.append(sia.polarity_scores(tweet)['compound'])
        plt.figure()
        plt.plot(myScores[1000:1100], '.', label='My Model')
        plt.plot(nltkScores[1000:1100], '.', label="NLTK model")
        # plt.plot(nltkScores)
        plt.xlabel("No. of tweets")
        plt.ylabel("Sentiment Score")
        plt.title("Models performance on negative tweets")
        plt.legend()
        plt.draw()
        plt.show()



    def sentimentAnalysis(self, text):
        x = self.__features_extraction(text)
        sent = self.__sigmoid(np.dot(x, self.__theta))
        return equalize(sent[0])



positiveTweets = twitter_samples.strings('positive_tweets.json')
negativeTweets = twitter_samples.strings('negative_tweets.json')

trainPos = positiveTweets[:4000]
trainNeg = negativeTweets[:4000]
testPos = positiveTweets[4000:]
testNeg = negativeTweets[4000:]


#################################################################################
############################# HELPER FUNCTIONS ##################################
#################################################################################

# Removes punctuation from a string
def stripPunctuation(s, all=False):
    punctuationRegex = re.compile('[{0}]'.format(re.escape(string.punctuation)))
    return punctuationRegex.sub('', s.strip())

def equalize(score):
    if score < 0.5:
        score = score - 1
    return score
