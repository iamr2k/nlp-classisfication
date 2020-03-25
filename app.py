from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
stop_words=set(stopwords.words("english"))
import time
def token(text):
    tokenized_word=word_tokenize(text)
    return tokenized_word

def lem(filtered_sent):
    lem_words = []
    lem = WordNetLemmatizer()
    
    for w in filtered_sent:
        lem_words.append(lem.lemmatize(w,"v"))
    return lem_words

def stop(tokenized_word):
    filtered_sent=[]
    for w in tokenized_word:
        if w not in stop_words:
            filtered_sent.append(w)
    return filtered_sent

from nltk.tokenize.treebank import TreebankWordDetokenizer
def detok(sent):
    
    return(TreebankWordDetokenizer().detokenize(sent))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    
    if request.method == 'POST':
        message = [[0]]
        message[0][0] = request.form['message']
        message = np.array(message)
        data = pd.DataFrame(message)
        data["content"] = message
        t0 = time.time()
        
        data["cleaned"] = data.content.apply(lambda x: re.sub(r'http\S+', '', x))
        data.cleaned = data.cleaned.apply(lambda x : re.sub("[^A-Za-z" "]+"," ",x).lower())
        data.cleaned = data.cleaned.apply(lambda x : re.sub("[0-9" "]+"," ",x))
        data.cleaned = data.cleaned.apply(lambda x : re.sub(r'(?:^| )\w(?:$| )', ' ', x).strip())

        data.cleaned = data.cleaned.str.replace('re',' ')
        data.cleaned = data.cleaned.str.replace('original message',' ')
        data.cleaned = data.cleaned.str.replace('from',' ')
        data.cleaned = data.cleaned.str.replace('excelr',' ')
        data.cleaned = data.cleaned.str.replace('subject',' ')
        data["tokenized"] = data.cleaned.apply(token)
        data['lemmatized'] = data.tokenized.apply(lem)
        data["detokenized"] = data.lemmatized.apply(detok)
        t1 = time.time()
        loaded_model = pickle.load(open("rfmodel.pkl", 'rb'))
        word_vectorizer = pickle.load(open("rfwvector2.pkl", 'rb'))
        char_vectorizer = pickle.load(open("rfcvector2.pkl", 'rb'))
        
        train_word_features = word_vectorizer.transform(data.detokenized)
        train_char_features = char_vectorizer.transform(data.detokenized)
        train_features = hstack([train_char_features, train_word_features])
        
        t2 = time.time()
        my_prediction = loaded_model.predict(train_features)
        prob = loaded_model.predict_proba(train_features)
        extime = round(t1-t0 ,2)
        prtime = round(t2-t1 ,2)

    return render_template('result.html',prediction = my_prediction[0],prob = prob,extime = extime  , prtime = prtime)



if __name__ == '__main__':
    app.run(debug=True)
