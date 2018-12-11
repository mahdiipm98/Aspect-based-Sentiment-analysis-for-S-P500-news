## Sentiment analysis of news on S&P 500
import pandas as pd
import numpy as np

dataset = pd.read_csv('News.csv', sep='^', header = None, prefix = 'X')
dataset2 = dataset.X0.str.split('-', 2 ,expand = True)

dataset2.columns = ['entity', 'headline', 'summary']

## Replace mising values coded as 'NA,,,,,' with null value
cols = ['headline','summary']
for column in cols:
    dataset2.loc[dataset2[column] == 'NA,,,,,,', column] = str(np.nan)

dataset2[cols].dtypes

## in this phase of data cleaning we should consider that in some news, parts of headline has 
# been moved to summary because of multiple delimiters '-' in source file, so I
# split summry column by another '-' and the combine first part with headline

### Note: in many cases headlines and summary are separated correctly, so when we split the
#column 'summary' by '-' to two columns of 'summary2' and 'headline_2', in this cases 'summary2'
# will be empty and 'headline_2' will be consist of our summary
dataset2[['headline_2','summary2']] = dataset2['summary'].str.split('-', 1, expand = True)

col = ['headline','summary','headline_2', 'summary2'] ## convert all columns to string
for column in col:
    dataset2[column] = dataset2[column].astype(str)


'''' to solve the problem mentioned in Note, I create a loop to to create new 
column of 'summary3' that keeps values of summary2 and new values from column 'headline_2
which was misclassified to hadline'''''
for i in range(0, 3000):
    if dataset2.loc[i,'summary2'] == 'None':
        dataset2.loc[i,'summary3'] = dataset2.loc[i,'headline_2']
        dataset2.loc[i,'headline_2'] = 'None'
    elif dataset2.loc[i, 'summary2'] != 'None':
        dataset2.loc[i, 'summary3'] = dataset2.loc[i, 'summary2']
        dataset2.loc[i,'headline_2'] = dataset2.loc[i,'headline_2']
        
       
dataset2[col].dtypes

## converting 'None' to empty string 
dataset2.loc[dataset2['headline_2'] == 'None', 'headline_2'] = '' 
   
### now combine two columns of headline to have the main hedline
dataset2['headline_new'] = dataset2[['headline', 'headline_2']].apply(lambda x: ' '.join(x), axis=1)

''' now that the splitting part is completed, two columns of 'headline_new' and 'summary3' 
are our columns for news, so we do not need other columns and will remove them'''

dataset2.drop(labels = ['headline', 'summary', 'headline_2', 'summary2'],
                         axis = 1, inplace = True)

dataset2.rename(columns = {'entity':'entity', 'summary3':'summary','headline_new':'headline'},
                inplace = True)

dataset2.to_csv('News_cleaned.csv')



#### Data Preparation
''' let's remove URL, html tags, handle negation words, convert words to lower cases,
remove non-letter characters. these elements do not provide enough semantic information for 
task'''
import re

pat_1 = r"(?:\@|https?\://)\S+"
pat_2 = r'#\w+ ?'
combined_pat = r'|'.join((pat_1, pat_2))
www_pat = r'www.[^ ]+'
html_tag = r'<[^>]+>'
negations_ = {"isn't":"is not", "can't":"can not","couldn't":"could not", "hasn't":"has not",
                "hadn't":"had not","won't":"will not",
                "wouldn't":"would not","aren't":"are not",
                "haven't":"have not", "doesn't":"does not","didn't":"did not",
                 "don't":"do not","shouldn't":"should not","wasn't":"was not", "weren't":"were not",
                "mightn't":"might not",
                "mustn't":"must not"}
negation_pattern = re.compile(r'\b(' + '|'.join(negations_.keys()) + r')\b')

from nltk.tokenize import WordPunctTokenizer
tokenizer1 = WordPunctTokenizer()
tokenizer2 = WordPunctTokenizer()
    
    

corpus_summary = []
for i in range(0, 3000):
        stripped = re.sub(combined_pat, '', dataset2['summary'][i])
        stripped = re.sub(www_pat, '', stripped)
        cleantags = re.sub(html_tag, '', stripped)
        #lower_case = cleantags.lower()
        neg_handled = negation_pattern.sub(lambda x: negations_[x.group()], cleantags)
        letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
        tokens = tokenizer1.tokenize(letters_only)
        tokens = ' '.join(tokens)
        corpus_summary.append(tokens)
        
corpus_headline = []
for i in range(0, 3000):
        s = re.sub(combined_pat, '', dataset2['headline'][i])
        s = re.sub(www_pat, '', s)
        cleant = re.sub(html_tag, '', s)
        #lowercase = cleant.lower()
        neghandled = negation_pattern.sub(lambda y: negations_[y.group()], cleant)
        lettersonly = re.sub("[^a-zA-Z]", " ", neghandled)
        tokenss = tokenizer2.tokenize(lettersonly)
        tokenss = ' '.join(tokenss)
        corpus_headline.append(tokenss)
    
'''corpus_entity = []
for i in range(0,3000):
    lower = dataset2['entity'][i].lower()
    corpus_entity.append(lower)'''
        

####### creating new dataframe with processed featueres
        
headline_prepared = pd.DataFrame(corpus_headline)
headline_prepared.rename(columns = {0:'headline'}, inplace = True)
summary_prepared = pd.DataFrame(corpus_summary)
summary_prepared.rename(columns = {0:'summary'} , inplace = True)
'''entity_prepared = pd.DataFrame(corpus_entity)
entity_prepared.rename(columns = {0:'entity'}, inplace = True)'''


dataset_prepared = pd.concat([headline_prepared, summary_prepared, dataset2['entity']],
                             axis = 1)

dataset_prepared.loc[dataset_prepared['summary'] == 'nan', 'summary'] = ' '
dataset_prepared.loc[dataset_prepared['summary'] == 'na', 'summary'] = ' '
dataset_prepared.to_csv('dataset_prepared.csv')

##### Data Visualization
'''dataset_prepared = pd.read_csv('dataset_prepared.csv')
dataset_prepared.drop(labels = ['Unnamed: 0'],
                         axis = 1, inplace = True)'''

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from wordcloud import WordCloud, STOPWORDS

#### visualization for news summary
summary_visualization = []
for t in dataset_prepared.summary:
    summary_visualization.append(t)
summary_visualization = pd.Series(summary_visualization).str.cat(sep=' ')


from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(summary_visualization)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig('wordcloud_summary.png')
plt.show()


#### visualization for news headlines

headline_visualization = []
for t in dataset_prepared.headline:
    headline_visualization.append(t)
headline_visualization = pd.Series(headline_visualization).str.cat(sep=' ')
wordcloud = WordCloud(width=1600, height=800,max_font_size=200,colormap='magma').generate(headline_visualization) 
plt.figure(figsize=(12,10)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off") 
plt.savefig('wordcloud_headline.png')
plt.show()


######### determinine which entities exist in each news
for column in ['entity','summary','headline']:
    dataset_prepared[column] = dataset_prepared[column].astype(str)

dataset_prepared['news'] = dataset_prepared[['headline', 'summary']].apply(lambda x: ' '.join(x), axis=1)





########### performing sentiment analysis
import nltk
nltk.download('vader_lexion')

def nltk_sentiment(sentence):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    nltk_sentiment = SentimentIntensityAnalyzer()
    score = nltk_sentiment.polarity_scores(sentence)
    return score 

nltk_results = [nltk_sentiment(row) for row in dataset_prepared['news']]
results_df = pd.DataFrame(nltk_results)

completed_classification = pd.concat([dataset_prepared, results_df], axis = 1)


for i in range(0,3000):
    if completed_classification.loc[i, 'compound'] > 0.1:
        completed_classification.loc[i, 'sentiment'] = 1
    elif -0.1 <= completed_classification.loc[i, 'compound'] <= 0.1:
        completed_classification.loc[i, 'sentiment'] = 0
    elif completed_classification.loc[i, 'compound'] < -0.1 :
        completed_classification.loc[i, 'sentiment'] = -1
        

completed_classification['sentiment'].value_counts()

completed_classification.to_csv('Completed_classification.csv')

import pandas as pd
import numpy as np
completed = pd.read_csv('completed_classification.csv')


############### 

# removing stopwords from news
import re
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

news_no_stopwords = []
for i in range(0, 3000):
    ch = re.sub('[^a-zA-Z]', ' ', completed['news'][i])
    ch = ch.split()
    ps = PorterStemmer()
    ch = [ps.stem(word) for word in ch if not word in set(get_stop_words('english'))]
    ch = ' '.join(ch)
    news_no_stopwords.append(ch)

news_no_stopwords = pd.DataFrame(news_no_stopwords)
completed = pd.concat([completed, news_no_stopwords], axis = 1)
completed.rename(columns = {0:'news_cleared'},
                inplace = True)


################################################################################
#### define my model for predicting the sentiment analysis
# Part 1: building supervide machine learning models based on results from 
# Vader Sentiment Analyser

#Spliting The Data
from sklearn.cross_validation import train_test_split
SEED = 2000

x_train, x_validation, y_train, y_validation = train_test_split(completed.loc[:,'news_cleared'],
                                                                completed.loc[:,'sentiment'], test_size=.1, random_state=SEED)



#### Feature Extraction
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
import numpy as np
from time import time

def acc_summary(pipeline, x_train, y_train, x_test, y_test):
    t0 = time()
    sentiment_fit = pipeline.fit(x_train, y_train)
    y_pred = sentiment_fit.predict(x_test)
    train_test_time = time() - t0
    accuracy = accuracy_score(y_test, y_pred)
    print ("accuracy score: {0:.2f}%".format(accuracy*100))
    print ("train and test time: {0:.2f}s".format(train_test_time))
    print ("-"*80)
    return accuracy, train_test_time


from sklearn.feature_extraction.text import TfidfVectorizer
tvec = TfidfVectorizer()

from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFromModel

names = ["Logistic Regression", "Linear SVC", "LinearSVC with L1-based feature selection","Multinomial NB", 
         "Bernoulli NB", "Ridge Classifier", "AdaBoost", "Perceptron","Passive-Aggresive", "Nearest Centroid"]
classifiers = [
    LogisticRegression(),
    LinearSVC(),
    Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
  ('classification', LinearSVC(penalty="l2"))]),
    MultinomialNB(),
    BernoulliNB(),
    RidgeClassifier(),
    AdaBoostClassifier(),
    Perceptron(),
    PassiveAggressiveClassifier(),
    NearestCentroid()
    ]
zipped_clf = zip(names,classifiers)

tvec = TfidfVectorizer()
def classifier_comparator(vectorizer=tvec, n_features=10000, stop_words=None, ngram_range=(1, 1), classifier=zipped_clf):
    result = []
    vectorizer.set_params(stop_words=stop_words, max_features=n_features, ngram_range=ngram_range)
    for n,c in classifier:
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', c)
        ])
        print ("Validation result for {}".format(n))
        print (c)
        clf_acc,tt_time = acc_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
        result.append((n,clf_acc,tt_time))
    return result

'''the function below will fit all classifiers that defined to our data and masure accuracy for 
each classifier, in this function we can determine number of maximum features that
relate to frequency of the words in the news, we could define that our algorithm disregard the
words that  repeate more than this parameter, if we set this parameter to None, it will conclude
all words (about 6566), I limited it to 5000'''
trigram_result = classifier_comparator(n_features=5000,ngram_range=(1,3))

''' due to trigram_result (Ridge Classifier) has the best accuracy of 74%'''


####################### Part 2: building unsupervised deep learning models:
''' our models consist:
Word2vec
DBOW (Distributed Bag of Words)
DMC (Distributed Memory Concatenated)
DMM (Distributed Memory Mean)
DBOW + DMC
DBOW + DMM'''

# Build our word2vec model
import os
import sys
import gensim
import pandas as pd
from gensim.models.doc2vec import LabeledSentence

'''we can label each news with unique ID using Gensim’s LabeledSentence function,
and then concatenate the training and validation set for word representation, 
that's because word2vec and doc2vec training are completely unsupervised
and thus there is no need to hold out any data, as it is unlabelled.'''

def labelize_text(text,label):
    result = []
    prefix = label
    for i, t in zip(text.index, text):
        result.append(LabeledSentence(t.split(), [prefix + '_%s' % i]))
    return result
  
all_x = pd.concat([x_train,x_validation])

all_x_w2v = labelize_text(all_x, 'ALL')
x_train = labelize_text(x_train, 'TRAIN')
x_validation = labelize_text(x_validation, 'TEST')

##### train our first model word2vec from our corpus
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
from sklearn import utils
import numpy as np

model_w2v = Word2Vec(size=200, min_count=10) ### put the size of output vector = 200
model_w2v.build_vocab([x.words for x in tqdm(all_x_w2v)])
model_w2v.train([x.words for x in tqdm(all_x_w2v)], total_examples=len(all_x_w2v), epochs=1)

'''After training our model, I it now to convert words to vectors like the example below:'''
model_w2v['stock']

'''We  use the result of the training to extract the similarities of a given word as well'''
model_w2v.most_similar('amazon')


### we build our a Tf-IDF matrix, then define the function that creates an averaged review vector
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in all_x_w2v])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

def build_Word_Vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: 
            
            continue
    if count != 0:
        vec /= count
    return vec

'''We  convert our training and validation set into a list of vectors 
using the previous function. We also scale each column to have zero mean and 
unit standard deviation. After that, we feed our neural network with the resulted vectors
 Then, after the training, we will evaluate it on the validation set.'''
 
from sklearn.preprocessing import scale
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

train_vecs_w2v = np.concatenate([build_Word_Vector(z, 200) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_w2v = scale(train_vecs_w2v)
val_vecs_w2v = np.concatenate([build_Word_Vector(z, 200) for z in tqdm(map(lambda x: x.words, x_validation))])
val_vecs_w2v = scale(val_vecs_w2v)


model = Sequential()
model.add(Dense(256, activation='relu', input_dim=200))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_vecs_w2v, y_train, epochs=100, batch_size=32, verbose=2)

score = model.evaluate(val_vecs_w2v, y_validation, batch_size=128, verbose=2)

print(score[1]) 
'''the accuracy of the model using word2vec is 62.67%'''


##### now let's build DBOW model
from gensim.models import Doc2Vec
import multiprocessing

cores = multiprocessing.cpu_count()
model_dbow = Doc2Vec(dm=0, size=100, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_dbow.build_vocab([x for x in tqdm(all_x_w2v)])
model_dbow.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)

def build_doc_Vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_dbow[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: 
            continue
    if count != 0:
        vec /= count
    return vec

train_vecs_dbow = np.concatenate([build_doc_Vector(z, 100) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_dbow = scale(train_vecs_dbow)
val_vecs_dbow = np.concatenate([build_doc_Vector(z, 100) for z in tqdm(map(lambda x: x.words, x_validation))])
val_vecs_dbow = scale(val_vecs_dbow)

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=100))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_vecs_dbow, y_train, epochs=100, batch_size=32, verbose=2)
score = model.evaluate(val_vecs_dbow, y_validation, batch_size=128, verbose=2)

print (score[1])
'''the accuracy of the model using word2vec is 60.67% which is less than word2vec'''


#### Build Distributed Memory Concatenation model (DMC)
cores = multiprocessing.cpu_count()
model_dmc = Doc2Vec(dm=1, dm_concat=1, size=100, window=2, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_dmc.build_vocab([x for x in tqdm(all_x_w2v)])
model_dmc.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)

def build_doc_Vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_dmc[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: 
            continue
    if count != 0:
        vec /= count
    return vec

  
train_vecs_dmc = np.concatenate([build_doc_Vector(z, 100) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_dmc = scale(train_vecs_dmc)


val_vecs_dmc = np.concatenate([build_doc_Vector(z, 100) for z in tqdm(map(lambda x: x.words, x_validation))])
val_vecs_dmc = scale(val_vecs_dmc)

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=100))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(train_vecs_dmc, y_train, epochs=100, batch_size=32, verbose=2)
score = model.evaluate(val_vecs_dmc, y_validation, batch_size=128, verbose=2)

print (score[1])
'''the accuracy for DMC model is 57%'''

### Build Distributed Memory Mean - DMM model
cores = multiprocessing.cpu_count()
model_dmm = Doc2Vec(dm=1, dm_mean=1, size=100, window=4, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_dmm.build_vocab([x for x in tqdm(all_x_w2v)])
model_dmm.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)

def build_doc_Vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_dmm[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: 
            continue
    if count != 0:
        vec /= count
    return vec

  
train_vecs_dmm = np.concatenate([build_doc_Vector(z, 100) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_dmm = scale(train_vecs_dmm)

val_vecs_dmm = np.concatenate([build_doc_Vector(z, 100) for z in tqdm(map(lambda x: x.words, x_validation))])
val_vecs_dmm = scale(val_vecs_dmm)

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=100))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(train_vecs_dmm, y_train, epochs=100, batch_size=32, verbose=2)
score = model.evaluate(val_vecs_dmm, y_validation, batch_size=128, verbose=2)
print (score[1])













