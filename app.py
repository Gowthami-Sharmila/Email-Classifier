import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 


app = Flask(__name__)
loaded_model = pickle.load(open("abusive.pkl", "rb"))
count_vector=  pickle.load(open("cv.pkl", "rb"))



def to_lower(text):
    result = str(text).lower()
    return result

# it removes special character and numbers 
import re
def remove_special_characters(text):
    #result = re.sub("[^A-Za-z0-9]+"," ", text)
    result =  re.sub(r'[^a-zA-Z]', ' ', text)
    return result



def removal_hyperlinks(text):
    result =  re.sub(r"http\\S+", " ", str(text))
    return result


def removal_whitespaces(text):
    result =  re.sub(' +', ' ', text)
    return result


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
#print(stopwords.words('english'))
stop_words = set(stopwords.words('english'))



def removal_stopwords(text):
    word_tokens = word_tokenize(text)  
    filtered_sentence = []
    a_row=""
    for a_word in word_tokens:
        if a_word not in stop_words:
            filtered_sentence.append(a_word)
            a_row = " ".join(filtered_sentence)
    return a_row




from nltk import WordNetLemmatizer
lemma = WordNetLemmatizer()
def lemmatization(text):
    word_tokens = word_tokenize(text) 
    a_array=[]
    a_string = ""
    for a_word in word_tokens:
               
        a_lemma = lemma.lemmatize(a_word,pos = "n")
        a_lemma1 = lemma.lemmatize(a_lemma, pos="v")
        a_lemma2 = lemma.lemmatize(a_lemma1, pos="a")
   
        a_array.append(a_lemma2)
        
        a_string = " ".join(a_array)
    return a_string



from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

from sklearn.feature_extraction.text import TfidfTransformer






def text_cleaning(new_email):
    new_email =new_email.replace('\n'," ")
    
    new_email2= to_lower(new_email)
    new_email3= remove_special_characters(new_email2)
    new_email4= removal_hyperlinks(new_email3)
    new_email5= removal_whitespaces(new_email4)
    new_email6= removal_stopwords(new_email5)
    text = lemmatization(new_email6)
    return text





@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
         
        sentence = request.form['sentence']
        cleaned_string =text_cleaning(sentence)    
        corpus =[cleaned_string]    
        new_X_test = count_vector.transform(corpus)    
        tfidf_transformer=TfidfTransformer().fit(new_X_test)    
        emails_tfidf = tfidf_transformer.transform(new_X_test)    
        new_y_pred = loaded_model.predict(new_X_test)   
        output = new_y_pred
        print(new_y_pred)
        return render_template("index.html",output=output)
    else:
         return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)