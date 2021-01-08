

# import libraries

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

email_data =pd.read_csv("G:\\Project-2\\emails.csv")


email_data.head()
email_data.shape


email_data.dtypes

email_data.info()


email_data.columns


email_data ["Class"].value_counts().head()


email_data_ab = email_data[["content","Class"]]


email_data_ab.head()

print(email_data_ab["Class"].value_counts())
sns.countplot(email_data_ab["Class"])


email_data_ab.groupby('Class').describe()


email_data_ab['content_count']=email_data_ab['content'].apply(lambda x: len(str(x)))
email_data_ab.head()


email_data_ab['content_count'].describe()

email_data_ab[email_data_ab['content_count']==272036]['content'].iloc[0]

# Shortest mail
email_data_ab[email_data_ab['content_count']==1]['content'].iloc[0]


import re

email_data_ab["content_w_space"]=email_data_ab["content"].replace('\n'," ",regex=True)

email_data_ab["content_w_space"].head(10)


def to_lower(text):
    result = str(text).lower()
    return result


email_data_ab["content_low"]=email_data_ab["content_w_space"].apply(lambda x: to_lower(x))


email_data_ab["content_low"].head()



# it removes special character and numbers 

def remove_special_characters(text):
    #result = re.sub("[^A-Za-z0-9]+"," ", text)
    result =  re.sub(r'[^a-zA-Z]', ' ', text)
    return result

email_data_ab["content_wsch"]=email_data_ab["content_low"].apply(lambda x: remove_special_characters(x))


email_data_ab["content_wsch"].head()

def removal_hyperlinks(text):
    result =  re.sub(r"http\\S+", " ", str(text))
    return result

email_data_ab["content_whl"]=email_data_ab["content_wsch"].apply(lambda x: removal_hyperlinks(x))


email_data_ab["content_whl"].head()


def removal_whitespaces(text):
    result =  re.sub(' +', ' ', text)
    return result


email_data_ab["content_wws"]=email_data_ab["content_whl"].apply(lambda x: removal_whitespaces(x))


email_data_ab["content_wws"].head()


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


email_data_ab["content_w_sw"]=email_data_ab.content_wws.apply(lambda x: removal_stopwords(x))


email_data_ab["content_w_sw"].head()


# Text normalization

# 1.Stemming

from nltk.stem import PorterStemmer

stemer = PorterStemmer()
def stemming(text):
    word_tokens = word_tokenize(text)  
    a_array=[]
    a_string = ""
    for a_word in word_tokens:
   
       a_stem = stemer.stem(a_word)
   
       a_array.append(a_stem)
    
       a_string = " ".join(a_array)
    return a_string

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


email_data_ab["content_lemma"]=email_data_ab.content_w_sw.apply(lambda x: lemmatization(x))


email_data_ab["content_lemma"].head()


email_data_ab["content_cleaned"]=email_data_ab["content_lemma"]


email_data_ab["length"] = email_data_ab["content_cleaned"].apply(lambda x: len(x))


email_data_ab["length"].describe()


email_data_final = email_data_ab[["content_cleaned","Class","length"]]


email_data_final.head()

len(email_data_final)

duplicate_records = email_data_final[email_data_final.duplicated()] 
duplicate_records.head(5)


len(duplicate_records)


email_data_final =  email_data_final.drop_duplicates() # keeping the first value
email_data_final.head()


email_data_final.shape

email_data_final.to_csv("email_data_final.csv")


# Target Variable Class
    
print(email_data_final["Class"].value_counts())
sns.countplot("Class", data = email_data_final)
                       

email_data_final['length'].plot(bins=50,kind='hist')


 #email_data_final.to_csv("email_data_final.csv")


email_data_final['length'].describe()

# Longest mail

email_data_final[email_data_final['length']==253499]['content_cleaned'].iloc[0]

# Shortest mail

email_data_final[email_data_final['length']==0]['content_cleaned'].iloc[0]

email_abusive= email_data_final[(email_data_final["Class"]=="Abusive")]
email_abusive.shape


email_abusive.content_cleaned[0:5]


email_non_abusive= email_data_final[(email_data_final["Class"]=="Non Abusive")]
email_non_abusive.shape


# #### Abusive Mails
final_email_abusive=""
abusive_email =[]
for text in email_abusive["content_cleaned"]:
    abusive_email.append(text)
    final_email_abusive =  "".join(abusive_email)
final_email_abusive


# #### Word Clouds for abusive emails
from wordcloud import WordCloud,STOPWORDS


stopwords = set(STOPWORDS) 

wordcloud_abusive_words = WordCloud(
        background_color='white',
        height = 4000,
        width=4000,
        stopwords = stopwords,
        min_font_size = 10
   ).generate(final_email_abusive)

#plt.figure(figsize = (40,40))
plt.figure(figsize = (8, 8), facecolor = None) 
plt.axis("off") 
plt.tight_layout(pad = 0)  
plt.imshow(wordcloud_abusive_words,interpolation="bilinear")


# #### Non Abusive

# #### Building Text Corpus

final_email_nonabusive=""
nonabusive_email =[]
for text in email_non_abusive["content_cleaned"]:
    nonabusive_email.append(text)
    final_email_nonabusive =  "".join(nonabusive_email)
final_email_nonabusive



wordcloud_nonabusive_words = WordCloud(
        background_color='white',
        height = 4000,
        width=4000,
        stopwords = stopwords,
        min_font_size = 10
   ).generate(final_email_nonabusive)



#plt.figure(figsize = (40,40))
plt.figure(figsize = (8, 8), facecolor = None) 
plt.axis("off") 
plt.tight_layout(pad = 0)  
plt.imshow(wordcloud_nonabusive_words,interpolation="bilinear")



#### N Gram Visualization

email_data_final.columns




from nltk.tokenize import word_tokenize


email_data_final["content_tokenized"]= email_data_final["content_cleaned"].apply(lambda x: word_tokenize(x) )


email_data_final["content_tokenized"].head()


email_data_final.columns


from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer()


X1=cv.fit(email_data_final["content_cleaned"])

print(len(X1.vocabulary_))
a_email=email_data_final['content_cleaned'][4]
a_email

a_email_vector=X1.transform([a_email])
print(a_email_vector)
print(a_email_vector.shape)


print(X1.get_feature_names()[24004])
print(X1.get_feature_names()[43987])


emails= X1.transform(email_data_final['content_cleaned']) # Transformig the entire corpus

emails.shape
print('Shape of Sparse Matrix: ',emails.shape)
print('Amount of non-zero occurences:',emails.nnz)


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer=TfidfTransformer().fit(emails)

tfidf_a_email = tfidf_transformer.transform(a_email_vector)
print(tfidf_a_email.shape)

print(tfidf_transformer.idf_[X1.vocabulary_['gamble']])
print(tfidf_transformer.idf_[X1.vocabulary_['asshole']])
print(tfidf_transformer.idf_[X1.vocabulary_['excelr']])
print(tfidf_transformer.idf_[X1.vocabulary_['ect']])
print(tfidf_transformer.idf_[X1.vocabulary_['make']])
print(tfidf_transformer.idf_[X1.vocabulary_['lavorato']])
print(tfidf_transformer.idf_[X1.vocabulary_['problem']])
print(tfidf_transformer.idf_[X1.vocabulary_['go']])

emails_tfidf = tfidf_transformer.transform(emails) # transforming the entire corpus
print(emails_tfidf.shape)


emails_tfidf.shape

# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
email_data_final['Class']= label_encoder.fit_transform(email_data_final['Class']) 
  
email_data_final['Class'].unique()


email_data_final["Class"].value_counts()



from sklearn.naive_bayes import MultinomialNB
abusive_detect_model = MultinomialNB().fit(emails_tfidf,email_data_final['Class'])
#abusive_detect_model = MultinomialNB().fit(emails_tfidf,email_data_final['Class'])

print('predicted:',abusive_detect_model.predict(tfidf_a_email)[0])
#print('predicted:',abusive_detect_model.predict(X_feature)[0])

print('expected:',email_data_final.Class[0])


# Model Evaluation

all_predictions = abusive_detect_model.predict(emails_tfidf)
print(all_predictions)


comapare_df =pd.DataFrame({"predicted":all_predictions,"Actual":email_data_final["Class"] })


comapare_df.head(20)


from sklearn.metrics import classification_report,confusion_matrix, accuracy_score


print(classification_report(email_data_final['Class'],all_predictions))
print(confusion_matrix(email_data_final['Class'],all_predictions))

print(accuracy_score(email_data_final['Class'],all_predictions))


from sklearn.model_selection import train_test_split



# Feature data 
X_data = emails_tfidf
#X_data = X_feature

y =email_data_final["Class"]


X_data.shape

y.shape


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=0)


X_train.shape, y_train.shape, X_test.shape,  y_test.shape


print(y_train.value_counts())
sns.countplot(y_train)


from imblearn.over_sampling import SMOTE


sm=SMOTE(random_state=0)

X_train_bal, y_train_bal = sm.fit_resample(X_train,y_train)

X_train_bal.shape, y_train_bal.shape

print(y_train_bal.value_counts())
sns.countplot(y_train_bal)


y_train_bal


# #### 2. Underspamling


from imblearn.under_sampling import RandomUnderSampler



us = RandomUnderSampler(random_state=42)


X_bal,y_bal = us.fit_sample(X_feature,y)



from collections import Counter


print("oringinal dataset shape {}".format(Counter(y)))
print("oringinal dataset shape {}".format(Counter(y_bal)))


from sklearn.model_selection import train_test_split


from sklearn.naive_bayes import MultinomialNB


nb_model =MultinomialNB()
nb_model.fit(X_train_bal,y_train_bal)


from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, f1_score, recall_score, precision_score

# Train accuracy

y_train_pred = nb_model.predict(X_train_bal)


train_accuarcy_nb = accuracy_score(y_train_bal,y_train_pred)
train_accuarcy_nb


# Test Accuracy

y_test_pred = nb_model.predict(X_test)
#print(y_predicted)
#print(np.array(y_test))

test_accuarcy_nb = accuracy_score(y_test,y_test_pred)
test_accuarcy_nb


train_accuarcy_nb,test_accuarcy_nb


recall_score_nb = recall_score(y_test,y_test_pred)
recall_score_nb


precision_score_nb = precision_score(y_test,y_test_pred)
precision_score_nb


f1_score_nb = f1_score(y_test,y_test_pred)
f1_score_nb


print(classification_report(y_test,y_test_pred))
print(confusion_matrix(y_test,y_test_pred))


# #### 2.SVM Classifier

from sklearn.svm import SVC

sv_model =SVC()
sv_model.fit(X_train_bal,y_train_bal)


# Train Accuracy
y_train_pred = sv_model.predict(X_train_bal)
train_accur_svm =accuracy_score(y_train_bal,y_train_pred) 


# Test accuracy
y_test_pred = sv_model.predict(X_test)
test_accu_svm =accuracy_score (y_test,y_test_pred)

train_accur_svm, test_accu_svm


recall_score_svm = recall_score(y_test,y_test_pred)
recall_score_svm


precision_score_svm = precision_score(y_test,y_test_pred)
precision_score_svm

f1_score_svm = f1_score(y_test,y_test_pred)
f1_score_svm


print(classification_report(y_test,y_test_pred))
print(confusion_matrix(y_test,y_test_pred))


# #### 3.KNN Classifier

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_bal,y_train_bal )
 
y_train_pred = knn_model.predict(X_train_bal)
train_accur_knn =accuracy_score(y_train_bal,y_train_pred)

y_test_pred = knn_model.predict(X_test)
test_accu_knn =accuracy_score (y_test,y_test_pred)

train_accur_knn, test_accu_knn

f1_score_knn = f1_score(y_test,y_test_pred)
f1_score_knn

recall_score_knn = recall_score(y_test,y_test_pred)
recall_score_knn

precision_score_knn = precision_score(y_test,y_test_pred)
precision_score_knn

print(classification_report(y_test,y_test_pred))
print(confusion_matrix(y_test,y_test_pred))


# #### 4. Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier



gb_model = GradientBoostingClassifier()
gb_model.fit(X_train_bal,y_train_bal)


# Train Accuracy
y_train_pred = gb_model.predict(X_train_bal)
train_accur_gb =accuracy_score(y_train_bal,y_train_pred)

# Test accuracy
y_test_pred = gb_model.predict(X_test)
test_accu_gb =accuracy_score (y_test,y_test_pred)

train_accur_gb, test_accu_gb

recall_score_gb = recall_score(y_test,y_test_pred)
recall_score_gb
precision_score_gb = precision_score(y_test,y_test_pred)
precision_score_gb

f1_score_gb = f1_score(y_test,y_test_pred)
f1_score_gb

print(classification_report(y_test,y_test_pred))
print(confusion_matrix(y_test,y_test_pred))


# #### 6. XGBoost Classifier

from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_model.fit(X_train_bal,y_train_bal)
y_train_pred = xgb_model.predict(X_train_bal)
train_accur_xgb =accuracy_score(y_train_bal,y_train_pred)


# Test accuracy
y_test_pred = xgb_model.predict(X_test)
test_accu_xgb =accuracy_score (y_test,y_test_pred)

train_accur_xgb, test_accu_xgb



recall_score_xgb= recall_score(y_test,y_test_pred)
recall_score_xgb

precision_score_xgb = precision_score(y_test,y_test_pred)

precision_score_xgb

f1_score_xgb = f1_score(y_test,y_test_pred)
f1_score_xgb


print(classification_report(y_test,y_test_pred))
print(confusion_matrix(y_test,y_test_pred))


# ## Model Evaluation

# #### # Model Comparison






Evaluation_Scores =  {"Models":["Naive Bayes","KNN","GBM","XGB"],
                    "Train Accuracy":[train_accuarcy_nb,train_accur_knn,train_accur_gb, train_accur_xgb],
                    "Test Accuracy":[test_accuarcy_nb,test_accu_knn,test_accu_gb,test_accu_xgb],
                    "Recall" :[recall_score_nb,recall_score_knn,recall_score_gb,recall_score_xgb],
                    "Precision":[precision_score_nb,precision_score_knn,precision_score_gb,precision_score_xgb],
                    "F1 Score":[f1_score_nb,f1_score_knn,f1_score_gb,f1_score_xgb]}




Evaluation_Scores = pd.DataFrame(Evaluation_Scores)
Evaluation_Scores


# #### Predictions



type(pd.DataFrame(X_train_bal))




import os
os.chdir("D:\\Project\\EmailClassification")





import pickle
pickle.dump(cv, open('cv.pkl', 'wb'))

pickle.dump(xgb_model, open("xgb.pkl", "wb"))


"\n Helllo Dear How are You! 2 45@ Runnig".replace('\n'," ")




def text_cleaning(new_email):
    new_email =new_email.replace('\n'," ")
    
    new_email2= to_lower(new_email)
    new_email3= remove_special_characters(new_email2)
    new_email4= removal_hyperlinks(new_email3)
    new_email5= removal_whitespaces(new_email4)
    new_email6= removal_stopwords(new_email5)
    text = lemmatization(new_email6)
    return text


# = "\n Helllo Dear How are You! 2 45@ Running "


new_email = input("Please enter a new email:  ")
loaded_model = pickle.load(open("xgb.pkl", "rb"))

def new_email_predict(new_email):
    cleaned_string =text_cleaning(new_email)
    corpus =[cleaned_string]
    new_X_test = cv.transform(corpus)
    tfidf_transformer=TfidfTransformer().fit(new_X_test)
    emails_tfidf = tfidf_transformer.transform(new_X_test)
    new_y_pred = loaded_model.predict(new_X_test)
    return new_y_pred
a_email= new_email_predict(new_email)[0]
if a_email==0:
  print("Abusive")
else :
  print("Non-Abusive")

