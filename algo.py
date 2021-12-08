import pickle

import nltk
import pandas as pd
import stopwords as stopwords
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, _check_stop_list
from sklearn.model_selection import train_test_split
from stop_words import get_stop_words
import re
from nltk.corpus import stopwords

nltk.download('stopwords')
df = pd.read_csv("labels.csv")
print(df)

df['tweet'] = df['tweet'].str.lower()
df['tweet'] = df['tweet'].apply(lambda x: re.sub("[^a-z\s]", "", x))
df['tweet'] = df['tweet'].str.replace("#", " ")
df['tweet'] = df['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
stopwords = set(stopwords.words("english"))
df['tweet'] = df['tweet'].apply(lambda x: " ".join(word for word in x.split() if word not in stopwords))

print(df)

y = df['class']
X = df['tweet']


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = make_pipeline(
    TfidfVectorizer(stop_words=get_stop_words('en')),
    OneVsRestClassifier(SVC(kernel='linear', probability=True))
)

# clf.fit(X_train, y_train)

# s = pickle.dumps(clf)
