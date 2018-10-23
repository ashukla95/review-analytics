"""

@author: AIshwary Shukla
The final output of the proram is stored in two files namely: positive.csv and negative.csv
"""


""" importing modules """
#importing pandas framework to start analytics
import pandas as pd

#import CountVectorizer to convert collection of texts into matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer

#importing The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification)
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfTransformer



"""processing part """


#part1 : setting up the processing part
#loading the reviews(to be processed) for processing: to be used later on in part2
rev = pd.read_csv('reviews.csv')


#loading the training set called review_corpus
f = pd.read_csv('review_corpus.csv', sep=',', names=['Text', 'Sentiment'], dtype=str, header=0)




""" count vectorizer does the work of word seperation. 
Also, it provides the total count of words rpesent in the passed data.
In the function, line 1 is used to set Countvectorizer with some required parameters,
line 2 does the work of tokenization (word seperation), but with a condition that the word must be of atleast 2 letters; 
in this case letters like 'a', etc. are ignored.
line 3 returns the tokenized list
here min_df =1 means that countvectorizer will ignore those words who have a frequency of occurance <1. 
ngram_range(1,3) will produce n_grams -> unigram,bigram and trigram as well
"""

def split_into_lemmas(sent):
    ngram_vectorizer = CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)
    analyse = ngram_vectorizer.build_analyzer()
    return analyse(sent)

#Bag of words: converting tweets into a list of individual words, removing stop words, stemming.
#Tf-idf transformation: computing how important a word is based on number of times it appears in the corpus.
bowd_transformer = CountVectorizer(analyzer=split_into_lemmas, stop_words='english', strip_accents='ascii').fit(f['Text'])









#create a document term matrix using transform function.
text_bowd = bowd_transformer.transform(f['Text'])



"""creating a tf-idf term, so that the words who although have a less frequency but, can have 
more weightage can be extracted
Tf means term-frequency while tf-idf means term-frequency times inverse document-frequency. This is a common term weighting scheme in 
 information retrieval, that has also found good use in document classification.
The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document is to scale down the impact of
tokens that occur very frequently in a given corpus and that are hence empirically
less informative than features that occur in a small fraction of the training corpus.

"""
tfidf_transformer = TfidfTransformer().fit(text_bowd)


#again calculating the DTM
tfidf = tfidf_transformer.transform(text_bowd)

text_tfidf = tfidf_transformer.transform(text_bowd)












#performing classification
classifier_nby = MultinomialNB(class_prior=[0.30, 0.70]).fit(text_tfidf, f['Sentiment'])


#creating a sentiments dataframe for storing final output.
sentiments = pd.DataFrame(columns=['text', 'class', 'prob'])



#part2: iteration on each row form review dataframe starts form here
#all the above fuctions will thereby be called from here.
i = 0
for _, sent in rev.iterrows(): #iterrows is used to iterate through rows of a dataframe
    i += 1
    try:
        bowds_sent = bowd_transformer.transform(sent)
        tfidf_sent = tfidf_transformer.transform(bowds_sent)
        sentiments.loc[i-1, 'text'] = sent.values[0]
        sentiments.loc[i-1, 'class'] = classifier_nby.predict(tfidf_sent)[0]
        sentiments.loc[i-1, 'prob'] = round(classifier_nby.predict_proba(tfidf_sent)[0][1], 2)
    except Exception as e:
        sentiments.loc[i-1, 'text'] = sent.values[0]


"""output module """
#saving the final output
sentiments.to_csv('sentiments.csv', encoding='utf-8')

#print sentiments


#print type(sentiments)
#positive comments
sent1 = sentiments.loc[~sentiments['class'].str.contains('neg')]
#negative comments
sent2 = sentiments.loc[~sentiments['class'].str.contains('pos')]                   
#print sent1                
#print sent2

#subsetting only reviews
sent1_text = sent1['text']       
sent2_text = sent2['text']

#saving positive tweets
sent1_text.to_csv('positive.csv', encoding='utf-8')
#saving negative tweets
sent2_text.to_csv('negative.csv', encoding='utf-8')
