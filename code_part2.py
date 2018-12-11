######### S&P500 sentiment analysis part2
# in first part we built some models to predict sentiments for the whole news, 
#but in this part we first determine which entities exist in each news and then try to build 
# aspect-based sentiment analysis model

############### dtermine existance of entities in each news
import pandas as pd
import numpy as np

news = pd.read_csv('Completed_classification.csv')
entity = pd.read_csv('Entities1.csv')

entity.drop_duplicates(subset = 'Company', keep = 'last', inplace = True) #### removing duplicate entities
entity.reset_index(inplace = True)
entity.drop(columns = ['index'], inplace = True)


##### in Entity.csv file, in some cases the 'Word' and 'Company' column are diffierent
# in this cases 'Word' contains complete form of company name, so we keep abbreviated 
#form of the names from 'Company' column and then add those completed form to them by
# creating new list of Entities to make sure our model capture all kind of entities
non_abbreviation = entity.loc[14:, 'Company']
non_abbreviation = non_abbreviation.to_frame()
non_abbreviation.reset_index(inplace = True)
non_abbreviation.rename(columns = {0:'Entities'}, inplace = True)
non_abbreviation.drop(columns = ['index'], inplace = True)
non_abbreviation.rename(columns = {'Company':'Word'}, inplace = True)


entity_words = pd.DataFrame(entity.loc[:,'Word'])

complete_set_of_entities = pd.concat([entity_words, non_abbreviation], axis = 0)
complete_set_of_entities.reset_index(inplace = True)
complete_set_of_entities.drop(columns = ['index'], inplace = True)


'''col = ['Entities'] ## convert all columns to string
for column in col:
    complete_set_of_entities[column] = complete_set_of_entities[column].astype(str)'''

import re
from nltk.tokenize import WordPunctTokenizer
tokenizer1 = WordPunctTokenizer()

pat_1 = r"(?:\@|https?\://)\S+"
pat_2 = r'#\w+ ?'
combined_pat = r'|'.join((pat_1, pat_2))
www_pat = r'www.[^ ]+'
html_tag = r'<[^>]+>'

corpus_entities = []
for i in range(0, 1003):
        #stripped = re.sub(combined_pat, '', complete_set_of_entities['Entities'][i])
        #stripped = re.sub(www_pat, '', stripped)
        stripped = re.sub("Inc$", '', complete_set_of_entities['Word'][i])
        stripped = re.sub("Corp$", '', stripped)
        stripped = re.sub("Corp.$", '', stripped)
        stripped = re.sub("Inc.$", '', stripped)
        stripped = re.sub("Co$", '', stripped)
        stripped = re.sub("Co.$", '', stripped)
        stripped = re.sub("Ltd.$", '', stripped)
        stripped = re.sub("&$", '', stripped)
        #cleantags = re.sub(html_tag, '', stripped)
        #lower_case = cleantags.lower()
        #neg_handled = negation_pattern.sub(lambda x: negations_[x.group()], cleantags)
        #letters_only = re.sub("[^a-zA-Z]", " ", stripped)
        #tokens = tokenizer1.tokenize(letters_only)
        #tokens = ' '.join(tokens)
        corpus_entities.append(stripped)
        
Entity_cleaned = pd.DataFrame(corpus_entities)
Entity_cleaned.rename(columns = {0:'entity'}, inplace = True)

corpus_entities_2 = []
for i in range(0, 1003):
        stripped = re.sub(combined_pat, '', Entity_cleaned['entity'][i])
        #stripped = re.sub(www_pat, '', stripped)
        #stripped = re.sub("Inc$", '', complete_set_of_entities['Entities'][i])
        #stripped = re.sub("Corp$", '', stripped)
        #stripped = re.sub("Corp.$", '', stripped)
        #stripped = re.sub("Inc.$", '', stripped)
        #stripped = re.sub("Co$", '', stripped)
        #stripped = re.sub("Co.$", '', stripped)
        stripped = re.sub("Ltd.$", '', stripped)
        stripped = re.sub(".com$", '', stripped)
        stripped = re.sub("&$", '', stripped)
        #cleantags = re.sub(html_tag, '', stripped)
        #lower_case = cleantags.lower()
        #neg_handled = negation_pattern.sub(lambda x: negations_[x.group()], cleantags)
        #letters_only = re.sub("[^a-zA-Z]", " ", stripped)
        #tokens = tokenizer1.tokenize(letters_only)
        #tokens = ' '.join(tokens)
        corpus_entities_2.append(stripped)

Entity_cleaned_2 = pd.DataFrame(corpus_entities_2)
Entity_cleaned_2.rename(columns = {0:'entity'}, inplace = True)
Entity_cleaned_2.reset_index(inplace = True)
Entity_cleaned_2.drop(columns = ['index'], inplace = True)



list_1 = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W'
          ,'X','Y','Z']

for w in list_1:
    Entity_cleaned_2 = Entity_cleaned_2[Entity_cleaned_2.entity != w]


Entity_cleaned_2.reset_index(inplace = True)
Entity_cleaned_2.drop(columns = ['index'], inplace = True)

for w in list_1:
    Entity_cleaned_2 = Entity_cleaned_2[Entity_cleaned_2.entity != ' ']


Entity_cleaned_2.reset_index(inplace = True)
Entity_cleaned_2.drop(columns = ['index'], inplace = True)

'''Entity_cleaned_2 is complete list of entities(include abbreviations and complete name)
and pre-processed carefully'''

x =[]
y = []
for j in range(0,3000):
    #x = []
    for i in range(0, 991):
        if Entity_cleaned_2.loc[i, 'entity'] in news.loc[j, 'news']:
            x.append(Entity_cleaned_2.loc[i, 'entity'])
    y.append(x)
    x = []
    

y_cl = pd.DataFrame({'col':y})
y_cl.to_csv('list_of_entities_cleaned.csv')

######### check if our model could predict each news contains which entities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Convert the multi-labels into arrays
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y_cl.col)
X = news.news


# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.1, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import LabelPowerset
import numpy as np


# LabelPowerset allows for multi-label classification
# Build a pipeline for multinomial naive bayes classification
text_clf = Pipeline([('vect', CountVectorizer(stop_words = "english",ngram_range=(1, 1))),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                     ('clf', LabelPowerset(MultinomialNB(alpha=1e-1))),])
text_clf = text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)

# Calculate accuracy
np.mean(predicted == y_test)

    
###############################################################################
news_with_entities = pd.concat([news, y_cl], axis = 1)
news_with_entities.to_csv('news_with_entities.csv')

####


import os
import pandas as pd
import numpy as np
import pickle
from collections import Counter, defaultdict
import re

# import sklearn models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import LabelPowerset


# nlp libraries/api
import en_core_web_sm
from spacy import displacy
import gensim
#from neuralcoref import Coref


spacy = en_core_web_sm.load()
#coref = Coref(nlp=spacy)

# Load opinion lexicon
neg_file = open("neg_words.txt",encoding = "ISO-8859-1")
pos_file = open("pos_words.txt",encoding = "ISO-8859-1")
neg = [line.strip() for line in neg_file.readlines()]
pos = [line.strip() for line in pos_file.readlines()]
opinion_words = neg + pos


#Setup nltk corpora path and Google Word2Vec location
google_vec_file = r'C:\Users\kps\Desktop\tadbirgaran\processed_data\GoogleNews-vectors-negative300.bin.gz'
word2vec = gensim.models.KeyedVectors.load_word2vec_format(google_vec_file, binary = True)
pickle.dump(word2vec, open("word2vec_google.pkl", 'wb'))

# If above script has been run, load saved word embedding
word2vec = pickle.load(open("word2vec_google.pkl", 'rb'))

# load the Multi-label binarizer from previous notebook
mlb = pickle.load(open("mlb.pkl", 'rb'))

# load the fitted naive bayes model from previous notebook
naive_model1 = pickle.load(open("naive_model1.pkl", 'rb'))


################################################# the code below is taken from another source 
# on github for aspect based sentiment analysis, it was accurate with the source project,
# I wanted to develop this code for our problem but due to time limmitation I could not
def check_similarity(aspects, word):
    similarity = []
    for aspect in aspects:
        similarity.append(word2vec.n_similarity([aspect], [word]))
    # set threshold for max value
    if max(similarity) > 0.30:
        return aspects[np.argmax(similarity)]
    else:
        return None

def assign_term_to_aspect(aspect_sent, terms_dict, sent_dict, pred):
    '''
    function: takes in a sentiment dictionary and appends the aspect dictionary
    inputs: sent_dict is a Counter in the form Counter(term:sentiment value)
            aspect_sent is total sentiment tally
            terms_dict is dict with individual aspect words associated with sentiment
    output: return two types of aspect dictionaries: 
            updated terms_dict and aspect_sent
    '''
    aspects = Entity_cleaned_2.loc[:, 'entity']
    
    
    
    # First, check word2vec
    # Note: the .split() is used for the term because word2vec can't pass compound nouns
    for term in sent_dict:
        try:
            # The conditions for when to use the NB classifier as default vs word2vec
            if check_similarity(aspects, term.split()[-1]):
                terms_dict[check_similarity(aspects, term.split()[-1])][term] += sent_dict[term]
                if sent_dict[term] > 0:
                    aspect_sent[check_similarity(aspects, term.split()[-1])]["pos"] += sent_dict[term]
                else:
                    aspect_sent[check_similarity(aspects, term.split()[-1])]["neg"] += abs(sent_dict[term])
            elif (pred[0] == " "):
                continue
            elif (len(pred) == 1):
                terms_dict[pred[0]][term] += sent_dict[term]
                if sent_dict[term] > 0:
                    aspect_sent[pred[0]]["pos"] += sent_dict[term]
                else:
                    aspect_sent[pred[0]]["neg"] += abs(sent_dict[term])
            # if unable to classify via NB or word2vec, then put them in misc. bucket
            else:
                terms_dict["misc"][term] += sent_dict[term]
                if sent_dict[term] > 0:
                    aspect_sent["misc"]["pos"] += sent_dict[term]
                else:
                    aspect_sent["misc"]["neg"] += abs(sent_dict[term])
        except:
            print(term, "not in vocab")
            continue
    return aspect_sent, terms_dict
    
    
def feature_sentiment(sentence):
    '''
    input: dictionary and sentence
    function: appends dictionary with new features if the feature did not exist previously,
              then updates sentiment to each of the new or existing features
    output: updated dictionary
    '''

    sent_dict = Counter()
    sentence = spacy(sentence)
    debug = 0
    for token in sentence:
    #    print(token.text,token.dep_, token.head, token.head.dep_)
        # check if the word is an opinion word, then assign sentiment
        if token.text in opinion_words:
            sentiment = 1 if token.text in pos else -1
            # if target is an adverb modifier (i.e. pretty, highly, etc.)
            # but happens to be an opinion word, ignore and pass
            if (token.dep_ == "advmod"):
                continue
            elif (token.dep_ == "amod"):
                sent_dict[token.head.text] += sentiment
            # for opinion words that are adjectives, adverbs, verbs...
            else:
                for child in token.children:
                    # if there's a adj modifier (i.e. very, pretty, etc.) add more weight to sentiment
                    # This could be better updated for modifiers that either positively or negatively emphasize
                    if ((child.dep_ == "amod") or (child.dep_ == "advmod")) and (child.text in opinion_words):
                        sentiment *= 1.5
                    # check for negation words and flip the sign of sentiment
                    if child.dep_ == "neg":
                        sentiment *= -1
                for child in token.children:
                    # if verb, check if there's a direct object
                    if (token.pos_ == "VERB") & (child.dep_ == "dobj"):                        
                        sent_dict[child.text] += sentiment
                        # check for conjugates (a AND b), then add both to dictionary
                        subchildren = []
                        conj = 0
                        for subchild in child.children:
                            if subchild.text == "and":
                                conj=1
                            if (conj == 1) and (subchild.text != "and"):
                                subchildren.append(subchild.text)
                                conj = 0
                        for subchild in subchildren:
                            sent_dict[subchild] += sentiment

                # check for negation
                for child in token.head.children:
                    noun = ""
                    if ((child.dep_ == "amod") or (child.dep_ == "advmod")) and (child.text in opinion_words):
                        sentiment *= 1.5
                    # check for negation words and flip the sign of sentiment
                    if (child.dep_ == "neg"): 
                        sentiment *= -1
                
                # check for nouns
                for child in token.head.children:
                    noun = ""
                    if (child.pos_ == "NOUN") and (child.text not in sent_dict):
                        noun = child.text
                        # Check for compound nouns
                        for subchild in child.children:
                            if subchild.dep_ == "compound":
                                noun = subchild.text + " " + noun
                        sent_dict[noun] += sentiment
                    debug += 1
    return sent_dict

def classify_and_sent(sentence, aspect_sent, terms_dict):
    '''
    function: classify the sentence into a category, and assign sentiment
    note: aspect_dict is a parent dictionary with all the aspects
    input: sentence & aspect dictionary, which is going to be updated
    output: updated aspect dictionary
    '''
    # classify sentence with NB classifier
    predicted = naive_model1.predict([sentence])
    pred = mlb.inverse_transform(predicted)
    
    # get aspect names and their sentiment in a dictionary form
    sent_dict = feature_sentiment(sentence)
    
    # try to categorize the aspect names into the 4 aspects in aspect_dict
    aspect_sent, terms_dict = assign_term_to_aspect(aspect_sent, terms_dict, sent_dict, pred[0])
    return aspect_sent, terms_dict

def replace_pronouns(text):
    coref.one_shot_coref(text)
    return coref.get_resolved_utterances()[0]

def split_sentence(text):
    '''
    splits review into a list of sentences using spacy's sentence parser
    '''
    review = spacy(text)
    bag_sentence = []
    start = 0
    for token in review:
        if token.sent_start:
            bag_sentence.append(review[start:(token.i-1)])
            start = token.i
        if token.i == len(review)-1:
            bag_sentence.append(review[start:(token.i+1)])
    return bag_sentence

# Remove special characters using regex
def remove_special_char(sentence):
    return re.sub(r"[^a-zA-Z0-9.',:;?]+", ' ', sentence)

def review_pipe(review, aspect_sent, terms_dict={'ambience':Counter(), 'food':Counter(), 'price':Counter(), 'service':Counter(),'misc':Counter()}):
    review = replace_pronouns(review)
    sentences = split_sentence(review)
    for sentence in sentences:
        sentence = remove_special_char(str(sentence))
        aspect_sent, terms_dict = classify_and_sent(sentence.lower(), aspect_sent, terms_dict)
    return aspect_sent, terms_dict


        
    
        