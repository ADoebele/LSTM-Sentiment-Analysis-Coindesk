########################################################################################
########################################################################################
###############         Sentiment prediction and             ###########################
###############       Sentiment Index construction           ###########################
###############                                              ###########################
########################################################################################
########################################################################################

from __future__ import division
import pickle
from math import log
import pandas as pd
import os
from pandas import Series
from matplotlib import pyplot
from keras.models import load_model
from keras.preprocessing.text import Tokenizer




os.chdir("/Users/alexd/PycharmProjects/LSFM/")


#Load articles:
article_list = pd.read_csv('Coindesk_articles.csv',sep = '\t', header = None ,encoding = 'ISO-8859-1')


#Define metrics precision and recall
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


###Calculate sentiment for each sentence in an article:
#(Note: Calculation might take a while, suppress output for larger dataset)

#Load one of the trained models
filepath_a ='SMOTE_model.hdf5'
model = load_model(filepath_a,custom_objects={"precision": precision, "recall":recall})

article_sent = []
max_fatures = len(list(data["text"].str.split(' ', expand=True).stack().unique())) + 1 #Number of unique words in the dictionary
tokenizer = Tokenizer(num_words=max_fatures, split=' ',filters='!"#&()*+,-./:;<=>?@[\]^_`{"}~\t\n',lower = True)

for i in range(len(article_list)):
    count = 0
    sentence_sent = []
    for sentence in article_list[i]:
        print(sentence)
        twt = [sentence]
        tokenizer.fit_on_texts(twt)
        X_pred = tokenizer.texts_to_sequences(twt)
        #padding the sentence to have exactly the same shape as `embedding_2` input
        X_pred = pad_sequences(X_pred, maxlen=56, dtype='int32', value=0)   #MAXLEN from before is 81/56 automize!
        print(X_pred)
        sentiment = model.predict(X_pred,batch_size=100,verbose = 2)[0]
        if(np.argmax(sentiment) == 1):
            print('neutral')
            sentence_sent.insert(count,np.argmax(sentiment))
            count += 1
        elif(np.argmax(sentiment) == 0):
            sentence_sent.insert(count, np.argmax(sentiment))
            count += 1
            print('negative')
        elif(np.argmax(sentiment) == 2):
            sentence_sent.insert(count,np.argmax(sentiment))
            count += 1
            print('positive')
    print(i)
    article_sent.insert(i,sentence_sent)


#Save results temporarly:
with open('article_sent', 'wb') as g:
        pickle.dump(article_sent, g)


#Load results from temporarly storage:
with open('article_sent', 'rb') as g:
    article_sent = pickle.load(g)



###Calculate sentiment for each article based on the prediction above and the notation introduced:
article_sent_full = []

for article in range(len(article_list)):#len(article_list):
    count = 0
    Pos = 0
    Neg = 0
    for sentence in range(len(article_sent[article])):
        count += 1
        if(article_sent[article][sentence]==0):
            Neg += 1
        elif(article_sent[article][sentence]==2):
            Pos += 1
    print(Pos,Neg)
    PF = Pos/count
    NF = Neg/count
    B_a = log(1+PF) - log(1+NF)
    #Transform to [-1,1] scale for better interpretation
    B   = (log(2)**(-1))*B_a

    article_sent_full.insert(article,B)


##Combine and weight articles from the same day:

#Open time_list with date.time of each article(from Text extraction_cleaning.py)
with open('time_list', 'rb') as g:
    time_list = pickle.load(g)

#Exclude hours,minutes and seconds from timestamp
#time_list_2 = time_list[0:50]
for date in range(len(time_list)):
    time_list[date] = datetime.date(time_list[date].year, time_list[date].month, time_list[date].day)

#Combine timestamp and sentiment in dataframe:
d = {'Date':time_list,'Sentiment':article_sent_full}
df = pd.DataFrame(d)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

#Average over sentiment on the same day/week/month/year:
df_sent = df.set_index('Date').resample('W').mean()


#Save dataframe in csv file:
df_sent.to_csv("Pred_Sentiment.csv")


###Graphical illustraten of sentiment results:
df_sent.plot() #default style are lines
plt.savefig('Sentiment_year.png')
plt.show()


#Plot number of articles:
time_list_df = pd.DataFrame(time_list,columns = ["Date"])

s = pd.to_datetime(time_list_df["Date"])
df = s.groupby(s.dt.floor('d')).size().reset_index(name='count')
print (df)


plt.plot( 'Date', 'count', data=df, linestyle='', marker='o', markersize=2.5)
plt.xlabel('Date')
plt.ylabel('#Articles')
