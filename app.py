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
        data["cleaned"] = data.content.apply(lambda x: re.sub(r'http\S+', '', x))
        data.cleaned = data.cleaned.apply(lambda x : re.sub("[^A-Za-z" "]+"," ",x).lower())
        data.cleaned = data.cleaned.apply(lambda x : re.sub("[0-9" "]+"," ",x))
        data.cleaned = data.cleaned.apply(lambda x : re.sub(r'(?:^| )\w(?:$| )', ' ', x).strip())

        data.cleaned = data.cleaned.str.replace('re',' ')
        data.cleaned = data.cleaned.str.replace('original message',' ')
        data.cleaned = data.cleaned.str.replace('from',' ')
        data.cleaned = data.cleaned.str.replace('excelr',' ')
        data.cleaned = data.cleaned.str.replace('subject',' ')

        loaded_model = pickle.load(open("model1.pkl", 'rb'))
        word_vectorizer = pickle.load(open("wvector.pkl", 'rb'))
        char_vectorizer = pickle.load(open("cvector1.pkl", 'rb'))
        
        train_word_features = word_vectorizer.transform(data.cleaned)
        train_char_features = char_vectorizer.transform(data.cleaned)
        train_features = hstack([train_char_features, train_word_features])
        
        
        my_prediction = loaded_model.predict(train_features)
        


    return render_template('result.html',prediction = my_prediction[0])



if __name__ == '__main__':
    app.run(debug=True)
