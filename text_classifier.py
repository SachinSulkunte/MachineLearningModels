# Standard Libraries
import pandas as pd
import numpy as np
import json

# Data Preprocessing & NLP
import nltk
import re
import string

import sys
import gensim

from textblob import Word

import xgboost as xgb
from xgboost import XGBClassifier

from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')

# Models
from sklearn.svm import SVC, LinearSVC
from sklearn import model_selection
from sklearn.feature_selection import chi2
import joblib

# Performance metrics
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

from elasticsearch import Elasticsearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
import json

news_df = pd.read_csv('../bbc-text.csv')
news_df.info()
#print(news_df['category'].value_counts())

# index category names
news_df['category_id'] = news_df['category'].factorize()[0]

# dataframe for unique categories
category_id_df = news_df[['category', 'category_id']].drop_duplicates().sort_values('category_id')
# convert category names to index number from dictionary
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)

# Visual
# news_df.groupby('category').category_id.count().plot.bar(ylim=0)

news_df.drop_duplicates(subset=['category', 'text'], inplace=True)
# Cleaning
def clean_text(text):
    # remove everything except alphabets
    text = re.sub("[^a-zA-Z]", " ", text)
    # remove whitespaces
    text = ' '.join(text.split())
    text = text.lower()
    
    return text

# clean text feature
news_df['clean_text'] = news_df['text'].apply(clean_text).str.replace('bn bn ', '')

# other clean text feature
news_df['clean_text'] = news_df['text'].apply(clean_text).str.replace(' bn ', '')

# Lemmatize words
lemmatizer = WordNetLemmatizer()
def tokenize_and_lemmatize(text):
    # tokenization to ensure that punctuation is caught as its own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    lem = [lemmatizer.lemmatize(t) for t in filtered_tokens]
    return lem

# Count Vectorizer object
count_vec = CountVectorizer(stop_words='english', max_features=10000)
# Defining a TF-IDF Vectorizer
tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), tokenizer=tokenize_and_lemmatize, max_features=10000, use_idf=True)
features = tfidf_vec.fit_transform(news_df.clean_text).toarray()
labels = news_df.category_id

print(sorted(category_to_id.items()))

# Use chi-square analysis to find corelation between features (importantce of words) and labels(news category) 
N = 3  # top 3 categories
# highly corelated words for each category
for category, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])                                  # Sorts the indices of features_chi2[0] - the chi-squared stats of each feature
    feature_names = np.array(tfidf_vec.get_feature_names())[indices]            # Converts indices to feature names ( in increasing order)
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]         # List of unigram features ( in increasing order of chi-squared stat values)
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]          # List for bigram features ( in increasing order of chi-squared stat values)
    #print("# '{}':".format(category))
    #print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:]))) # Print 3 with highest Chi squared stat
    #print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:]))) # Print 3 with highest Chi squared stat

news_df.to_csv('out_df.csv', index=False)

X = news_df.loc[:,'clean_text']
y = news_df.loc[:,'category_id']

# Basic validation: splitting the data 80-20-20 train/test
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, news_df.index, test_size=0.2, random_state=55)

# Tf-Idf transformation 
xtrain_tfidf = tfidf_vec.fit_transform(X_train)
xtest_tfidf = tfidf_vec.transform(X_test)

# Count Vectorizer transformation
xtrain_cv = count_vec.fit_transform(X_train)
xtest_cv = count_vec.transform(X_test)

model = LinearSVC()
mdl = model.fit(xtrain_tfidf, y_train)

def convert(text):
    frame = {0:text}
    ser = pd.Series(data=frame)
    vec = tfidf_vec.transform(ser)
    return vec

############################### Classification ###################################

host = 'search-vast-db-h3mq23otgi6hffagiqjpjsnsey.us-east-1.es.amazonaws.com'
region = 'us-east-1' # e.g. us-west-1

service = 'es'
credentials = boto3.Session(profile_name="lambda").get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

es = Elasticsearch(
    hosts = [{'host': host, 'port': 443}],
    http_auth = awsauth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)

all_files = json.dumps(es.search(index="file_uploads", doc_type="_doc", q="*", filter_path=['hits.hits._source.text'], size=1000))
contents = json.loads(all_files)

# read in transcriptions from es into one string
transcription = ""
for hit in contents['hits']['hits']:
    try:
        transcription += hit['_source']['text']
    except KeyError as e:
        print("No Source")

vector = convert(transcription)
result = mdl.predict(vector)
print(result)

# Get range of time representing video creation for event timing
# get time of first video and normalize all audio timestamps to that time
    # find periods of strong overlap

