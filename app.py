import pandas as pd
import streamlit as st
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from joblib import load

st.title('API Twitter Classification')
user_input = st.text_area("Ecrivez un tweet")
st.markdown(f" Le tweet écrit est : {user_input}")


def getprob(txt):
    prediction = clf.predict([txt])
    print(prediction)
    return prediction[0]


clf = load('resultat.joblib')

prob = getprob(user_input)

if prob == 0:
    st.markdown(f" Ce tweet est classé à caractère haineux par l'algorithme")
elif prob == 1:
    st.markdown(f" Ce tweet est classé à langage offensant par l'algorithme")
else:
    st.markdown(f" Ce tweet est classé neutre par l'algorithme")
