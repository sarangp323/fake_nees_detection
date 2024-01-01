import pandas as pd
import numpy as np
import pickle

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from nltk.stem import PorterStemmer
ps = PorterStemmer()

df = pd.read_csv("news.csv")

corpus = []
count=0
for mess in df['text']:
    count+=1
    print(count,end="\r")
    no_punc = [char for char in mess if char not in string.punctuation]
    no_punc = "".join(no_punc)
    corpus.append(" ".join([ps.stem(word) for word in no_punc.split() if word.lower() not in stopwords.words('english')]))



df['text'] = corpus

X = df["text"]
y = df["label"]


cv = CountVectorizer()
X = cv.fit_transform(X).toarray()
pickle.dump(cv, open('countvector.pkl', 'wb'))

tfidf = TfidfTransformer()
X = tfidf.fit_transform(X)
pickle.dump(tfidf, open('tfidftransformer.pkl', 'wb'))

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=.3)

rf = RandomForestClassifier()
rf.fit(X_train,y_train)
pickle.dump(rf, open('randomforest.pkl', 'wb'))