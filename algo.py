import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, _check_stop_list
from sklearn.model_selection import train_test_split

df = pd.read_csv("labels.csv")

print(df)


def get_stop_words(self):
    """Build or fetch the effective stop words list.
            Returns
            -------
            stop_words: list or None
                    A list of stop words.
            """
    return _check_stop_list(self.stop_words)


clf = make_pipeline(
    TfidfVectorizer(stop_words=get_stop_words('en')),
    OneVsRestClassifier(SVC(kernel='linear', probability=True))
)

Y = df['class']
X = df.drop('class', axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

clf.fit(X_train, y_train)

s = pickle.dumps(clf)
