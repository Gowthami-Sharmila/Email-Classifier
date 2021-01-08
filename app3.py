

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
#




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


# #### 2. Lemmatization

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


import pickle
#new_email = input("Please enter a new email:  ")
loaded_model = pickle.load(open("abusive.pkl", "rb"))
count_vector=  pickle.load(open("cv.pkl", "rb"))

#import os
#os.chdir(r'D:\Project\EmailClassification\Deployment')
def new_email_predict(new_email):
    
    
    cleaned_string =text_cleaning(new_email)    
    corpus =[cleaned_string]    
    new_X_test = count_vector.transform(corpus)    
    tfidf_transformer=TfidfTransformer().fit(new_X_test)    
    emails_tfidf = tfidf_transformer.transform(new_X_test)    
    new_y_pred = loaded_model.predict(new_X_test)   
    return new_y_pred


import streamlit as st 

#from PIL import Image

def main():
    st.title("Mail Classification")
    
    html_temp = """
  
    <div style="background-color:Black;padding:10px">
    <h2 style="color:white;text-align:center;">Describe Abusive & Non-Abusive text </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Sentence = st.text_input("","Please Enter a valid sentence")
    result=""
    
   
    a_email= new_email_predict(Sentence)[0]
    
   
    if st.button("Predict"):
        
        a_email= new_email_predict(Sentence)[0]
        if a_email==0:
            result = 'Abusive'
        else :
            result = "Non-Abusive"
            
    st.success('The output is :  {}'.format(result.upper()))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")

    


if __name__=='__main__':
    main()
    

