#!usr/bin/python3

import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
#import enchant

class FunctionWords(TransformerMixin):
    def __init__(self, top=2500):
        self.top = top

    def fit(self, X, y=None):
        functionwords = {}
        labels = list(set(y))
        self.f = [[] for i in labels]
        for i in range(len(X)):
            label = y[i]
            for doc in X[i]:
                for token in doc:
                    if len(token)>3 and not token[0]=='#':
                        if token.lower() in functionwords:
                            functionwords[token.lower()][labels.index(label)]+=1
                        else:
                            functionwords[token.lower()]=[1]*len(labels)
        for token, counts in functionwords.items():
            for i in range(len(counts)):
                self.f[i].append((token, counts[i]/sum(counts)))
        for j in range(len(self.f)):
            self.f[j] = [k[0] for k in sorted(self.f[j], key=lambda x: x[1], reverse=True)[:self.top]]
        return self

    def transform(self, X):
        newX = []
        for x in X:
            newx = [0]*len(self.f)
            tokens = list(itertools.chain(*x))
            tokens = [i.lower() for i in tokens]
            for i in range(len(self.f)):
                newx[i]+=len(set(self.f[i]).intersection(tokens))
            newX.append(newx)
        return newX
   
class PosVec(TransformerMixin):
    def __init__(self, Xtrain, pos_train, pos_test):
        self.Xtrain = Xtrain
        self.pos_train = pos_train
        self.pos_test = pos_test

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        newX = []
        if len(self.pos_test)==0 or len(X)==len(self.pos_train):
            for x in X:
                postags = self.pos_train[self.Xtrain.index(x)]
                newX.append(' '.join(postags))
        else:
            for x in X:
                postags = self.pos_test[X.index(x)]
                newX.append(' '.join(postags))
        return newX

class FirstOrder(TransformerMixin):
    '''Parent class for second order real valued features'''
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        newX = [[self.calculate(x)] for x in X]
        return newX

class SecondOrderReal(TransformerMixin):
    '''Parent class for second order real valued features'''
    def fit(self, X, y=None):
        labels = list(set(y))
        values = {key: 0 for key in labels}
        counts = {key: 0 for key in labels}
        for i in range(len(X)):
            value = self.calculate(X[i])
            values[y[i]] += value
            counts[y[i]] += 1
        for key in values: # get value relative to count
            values[key] = values[key] / counts[key]
        for key in values: # get value relative to other labels
            values[key] = (values[key]*len(labels)) / sum(values.values())
        self.dist = values.values()
        return self

    def transform(self, X):
        newX = []
        values = [self.calculate(x) for x in X] # get values
        avg = sum(values)/len(values)
        for value in values:
            avgvalue = value/avg
            newX.append([avgvalue - i for i in self.dist])
        return newX

class AverageWordLength(SecondOrderReal):
    '''Calculates average token length for author'''
    def calculate(self, x):
        total = 0
        tokens = 0
        for tweet in x:
            for token in tweet:
                if len(token)>1:
                    total+=len(token)
                    tokens+=1
        try:
            result = total/tokens
        except:
            result = 0
        return result

class AverageSentenceLength(SecondOrderReal):
    '''Calculates average sentence length (in tokens) for author'''
    def calculate(self, x):
        sentences = 0
        tokens = 0
        for tweet in x:
            sentences+=1
            tokens+=len(tweet)
        result = tokens/sentences
        return result

class StartsWithCapital(SecondOrderReal):
    '''Percentage of tweets that start with capital letter'''
    def calculate(self, user_tweets):
        starts_with_capital = 0
        for user_tweet in user_tweets:
            if len(user_tweet) > 0:
                if user_tweet[0][0].isupper():
                    starts_with_capital += 1
        result = starts_with_capital/len(user_tweets)
        return result

class EndsWithPunctuation(SecondOrderReal):
    '''Percentage of tweets that end with punctuation'''
    def calculate(self, user_tweets):
        ends_with_punctuation = 0
        punctuation = ['!', '.', '?']
        for user_tweet in user_tweets:
            if len(user_tweet) > 0:
                if user_tweet[-1][-1] in punctuation:
                    ends_with_punctuation += 1
        result = ends_with_punctuation/len(user_tweets)
        return result

class Misspell(SecondOrderReal):
    def __init__(self, language):
        self.language = language
        if self.language == 'english':
            self.d = enchant.Dict('en')
        if self.language == 'spanish':
            self.d = enchant.Dict('es')
        if self.language == 'dutch':
            self.d = enchant.Dict('nl')

    def calculate(self, user_tweets):    
        all_tokens_user = 0
        for user_tweet in user_tweets:
            mspl = 0
            for token in user_tweet:
                if self.d.check(token) == False:
                        mspl += 1
            all_tokens_user += len(user_tweet)
        result = mspl/all_tokens_user
        return result
        
class PunctuationByTweet(SecondOrderReal):
    '''Average of percentage of punctuation characters per tweet'''
    def calculate(self, user_tweets):
        average_count = 0
        punctuation = [',', '.', ';', '!', '?', '-']
        for user_tweet in user_tweets:
            if len(user_tweet) > 0:
                punctuation_count = 0
                number_of_characters = 0
                for tok in user_tweet:
                    number_of_characters += len(tok)
                    if tok[-1] in punctuation:
                        punctuation_count += 1
                average_count += punctuation_count / number_of_characters
                
        result = average_count/len(user_tweets)
        return result

        
class CapitalizedTokens(SecondOrderReal):
    '''Average of percentage of capitalized tokens per tweet'''
    def calculate(self, user_tweets):
        average_count = 0
        for user_tweet in user_tweets:
            if len(user_tweet) > 0:
                capitalized_count = 0
                for tok in user_tweet:
                    if tok[0].isupper():
                        if tok not in ['AT_USER', 'URL', 'NUMBER']:
                            capitalized_count += 1
                average_count += capitalized_count / len(user_tweet)
                
        result = average_count/len(user_tweets)
        return result

class CapitalLetters(SecondOrderReal):
    '''Average of percentage of capitalized letters per tweet'''
    def calculate(self, user_tweets):
        average_count = 0
        
        for user_tweet in user_tweets:
            if len(user_tweet) > 0:
                capital_letter_count = 0
                number_of_characters = 0
                for tok in user_tweet:
                    number_of_characters += len(tok)
                    for character in tok:
                        if character.isupper():
                            capital_letter_count += 1
                average_count += capital_letter_count / number_of_characters
                
        result = average_count/len(user_tweets)
        return result

class EmoticonNoses(SecondOrderReal):
	'''Average of percentage of emoticons with noses, out of total emoticons, per tweet'''
	def calculate(self, user_tweets):
		average_count = 0
		num_emoticon_tweets = 0
		for user_tweet in user_tweets:
			nose_count = 0
			emoticon_count = 0
			for tok in user_tweet:
				if ':-' in tok: # nose emoticon
					nose_count += 1
				if len(tok) > 1 and tok[0] in [':', ';', '=']: # any emoticon
					emoticon_count += 1
#					print(tok)
#					print(len(tok))
#			print(emoticon_count)
			if emoticon_count > 0:
				num_emoticon_tweets += 1
#				print('num_emoticon_Tweets: %s' % num_emoticon_tweets)
				average_count += nose_count / emoticon_count
		try:
			result = average_count / num_emoticon_tweets
		except ZeroDivisionError:
			result = 0.5 # Impute 0.5 for emoticon-less users
#		print('result: %.2f' % result)
		return result

class EmoticonReverse(SecondOrderReal):
	'''Average of percentage of reverse emoticons (:, out of total emoticons, per tweet'''
	def calculate(self, user_tweets):
		average_count = 0
		num_emoticon_tweets = 0
		for user_tweet in user_tweets:
			reverse_count = 0
			emoticon_count = 0
			for tok in user_tweet:
				if len(tok) > 1 and tok[0] in [':', ';', '=']: # any emoticon
					emoticon_count += 1
				if len(tok) > 1 and tok[-1] in [':', ';', '=']: # any reverse emoticon
					reverse_count += 1 
					emoticon_count += 1
			if emoticon_count > 0:
				num_emoticon_tweets += 1
				average_count += reverse_count / emoticon_count
		try:
			result = average_count / num_emoticon_tweets
		except ZeroDivisionError:
			result = 0.5 # Impute 0.5 for emoticon-less users
		return result

class EmoticonCount(SecondOrderReal):
	'''Average of percentage of tokens that are emoticons, per tweet'''
	def calculate(self, user_tweets):
		average_count = 0
		for user_tweet in user_tweets:
			emoticon_count = 0
			for tok in user_tweet:
				if len(tok) > 1 and (tok[0] in [':', ';', '='] or tok[-1] in [':', ';', '=']):
					emoticon_count += 1
			try:
				average_count += emoticon_count / len(user_tweet)
			except ZeroDivisionError:
				average_count += 0

		result = average_count / len(user_tweets)
		return result

class EmoticonEmotion(SecondOrderReal):
	'''Average of percentage of happy emoticons, out of total emoticons, per tweet'''
	def calculate(self, user_tweets):
		average_count = 0
		num_emoticon_tweets = 0
		for user_tweet in user_tweets:
			happy_count = 0
			emoticon_count = 0
			for tok in user_tweet:
				if len(tok) > 1 and tok[0] in [':', ';', '=']: # any emoticon
					emoticon_count += 1
					if tok[-1] in [')', 'D', 'P', ']']:
						happy_count += 1
				if len(tok) > 1 and tok[-1] in [':', ';', '=']:
					emoticon_count += 1 
					if tok[0] in [')', 'D', 'P', ']']:
						happy_count += 1
			if emoticon_count > 0:
				num_emoticon_tweets += 1
				average_count += happy_count / emoticon_count
		try:
			result = average_count / num_emoticon_tweets
		except ZeroDivisionError:
			result = 0.5 # Impute 0.5 for emoticon-less users
		return result

class VocabularyRichness(SecondOrderReal):
    '''Percentage of unique words per user'''
    
    def calculate(self, user_tweets):
        dict = {}
        tokens = 0
        for user_tweet in user_tweets:
            if len(user_tweet) > 0:
                for tok in user_tweet:
                    dict[tok]= dict.get(tok, 0) + 1
                    tokens+=1
        unique_words = [k for (k, v) in dict.items() if v == 1]
        count = len(unique_words)
                
        result = count/tokens
        return result
