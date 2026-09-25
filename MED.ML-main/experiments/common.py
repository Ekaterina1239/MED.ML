import os, numpy as np, pandas as pd, re
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(os.path.dirname(__file__), '..', 'data'))
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def load():
    df = pd.read_csv(os.path.join(DATA_DIR,'Symptom2Disease.csv'))[['label','text']]
    return df

def clean(df):
    t = df.text.str.lower().str.strip()
    return df.loc[~t.duplicated()].reset_index(drop=True)

def groups(df, th=0.8):
    V = TfidfVectorizer().fit_transform(df.text.str.lower())
    S = cosine_similarity(V) >= th
    n, g = connected_components(S, directed=False)
    return g

def word_tfidf(): return TfidfVectorizer(ngram_range=(1,2), min_df=2, stop_words='english', sublinear_tf=True)
def char_tfidf(): return TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), min_df=2, sublinear_tf=True)

MODELS = {
 'ComplementNB (word)':      lambda: make_pipeline(word_tfidf(), ComplementNB()),
 'kNN k=5 (word)':           lambda: make_pipeline(word_tfidf(), KNeighborsClassifier(5, metric='cosine')),
 'RandomForest (word)':      lambda: make_pipeline(word_tfidf(), RandomForestClassifier(300, random_state=0, n_jobs=-1)),
 'LogReg (word)':            lambda: make_pipeline(word_tfidf(), LogisticRegression(C=10, max_iter=3000)),
 'Cal. LinearSVC (word) [baseline]': lambda: make_pipeline(word_tfidf(), CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5000), cv=5)),
 'Cal. LinearSVC (char)':    lambda: make_pipeline(char_tfidf(), CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5000), cv=5)),
 'Cal. LinearSVC (word+char)': lambda: make_pipeline(make_union(word_tfidf(), char_tfidf()), CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5000), cv=5)),
}

def typo(text, rate, rng):
    out=[]
    for ch in text:
        if ch.isalpha() and rng.random() < rate:
            op = rng.integers(3)
            if op==0: continue                       # deletion
            elif op==1: out.append(chr(rng.integers(97,123)))  # substitution
            else: out.append(ch); out.append(ch)     # duplication
        else: out.append(ch)
    return ''.join(out)

def ece(conf, correct, bins=10):
    e=0; edges=np.linspace(0,1,bins+1)
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(conf>lo)&(conf<=hi)
        if m.any(): e += m.mean()*abs(correct[m].mean()-conf[m].mean())
    return e
