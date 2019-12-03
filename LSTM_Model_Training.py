########################################################################################
########################################################################################
###############         LSTM Model Training                  ###########################
###############        (Naive,SMOTE,Word2Vec)                ###########################
###############                                              ###########################
########################################################################################
########################################################################################


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.chdir("/Users/alexd/PycharmProjects/LSFM/")


import numpy as np
import pandas as pd
import seaborn as sns
import collections
import matplotlib
#matplotlib.use('PS')
import matplotlib.pyplot as plt
from matplotlib import style as style
import pickle



from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout, Flatten
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
import keras.backend as K
import re



###Read in the data:
##txt files:

#data = pd.read_table("Training_50agree.txt",encoding = "ISO-8859-1",header = None)
#data.rename(columns={0:"text"},inplace = True)

# Extract sentiment into new column
#data = data["text"].str.rsplit("@", expand = True) #only necessary for the default txt file
#data.columns = ["text", "sentiment"]
#save new file as csv
#data.to_csv("Training_50agree.csv",encoding = "ISO-8859-1")#Training_allagree_random.csv


##CSV files:
data = pd.read_csv("Training_allagree.csv",sep=",", header = None ,encoding = "ISO-8859-1")
data = data.drop(index = 0)
data = data.drop(data.columns[[0]],axis = 1)
data.columns = ["text", "sentiment"]


#Randomize data
data = data.reindex(np.random.permutation(data.index))


#Determine the maximal length of a statement(number of words)
data['token_length'] = [len(x.split(" ")) for x in data.text]
data.loc[data.token_length.idxmax(),"text"]  #Check for outliers
data = data[data["text"] != data.loc[data.token_length.idxmax(),"text"] ] #Adjust the data manually
max(data.token_length)


#Graphical distribution of word frequency in articles
data['token_length'] = [len(x.split(" ")) for x in data.text]

n, bins, patches = plt.hist(x=data["token_length"], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('#Words per sentence')
plt.ylabel('Frequency')
plt.title('Word frequency Histogram')
maxfreq = n.max()
#Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()



#Clean data(LSTM can handle most unstructured data)
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z\s]', '', x)))#0-9



#Overview of each sentiment group in the training data (unbalanced data)
data.sentiment.value_counts()

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

#Train and test dataset
Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(data["text"],data["sentiment"], test_size = 0.1, random_state = 42)
print('Training Data:', X_train.shape[0])
print('Test Data:', X_test.shape[0])


#Tokenization
max_fatures = len(list(data["text"].str.split(' ', expand=True).stack().unique())) + 1 #Number of unique words
tokenizer = Tokenizer(num_words=max_fatures, split=' ',filters='[^0-9]!"#&()*+,-./:;<=>?@[\]^_`{"}~\t\n',lower = True)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
print('Top 5 most common words are:', collections.Counter(tokenizer.word_counts).most_common(5))
print('Found %s unique tokens.' % len(word_index))

#After having created the dictionary we can convert the text to a list of integer indexes.
#This is done with the text_to_sequences method of the Tokenizer.
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

seq_lengths = X_train.apply(lambda x: len(x.split(' ')))
seq_lengths.describe()

MAX_LEN = max(data.token_length)
X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=MAX_LEN,truncating='post')
X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=MAX_LEN,truncating='post')
print('{} -- is converted to -- {}'.format(X_train_seq[5], X_train_seq_trunc[5]))

le = LabelEncoder()
y_train_le = le.fit_transform(Y_train)
y_test_le = le.transform(Y_test)
y_train_oh = to_categorical(y_train_le)
y_test_oh = to_categorical(y_test_le)

X_train_emb, X_valid_emb, y_train_emb, y_valid_emb = train_test_split(X_train_seq_trunc, y_train_oh, test_size=0.1,
                                                                      random_state=37)

print('Shape of train set:',X_train_emb.shape)
print('Shape of validation set:',X_valid_emb.shape)

#######################################################################################################################

############################
##Compose the LSTM Network##
############################
BATCH_SIZE = 64
embed_dim = 512
lstm_out = 392
EPOCHS=120


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


############################
####   Naive LSTM   ########
############################

model = Sequential()
model.add(Embedding(max_fatures + 1, embed_dim, input_length = X_train_emb.shape[1],trainable = True))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out,return_sequences=True,recurrent_dropout=0.4))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(3,activation='softmax'))
print(model.summary())


model.compile(loss = "categorical_crossentropy", optimizer='Adam',metrics=[precision,recall])

filepath_a = "Naive.hdf5"
callback = [EarlyStopping(monitor='val_loss', patience=10, verbose = 1),ModelCheckpoint(filepath=filepath_a,
                                                                                        save_best_only = True,
                                                                                        save_weights_only = False)]

history_naive = model.fit(X_train_emb, y_train_emb, epochs = EPOCHS, batch_size=BATCH_SIZE,callbacks = callback,
                           verbose = 2, validation_data=(X_valid_emb, y_valid_emb))

############################
####  LSTM with callback ###
####   and classweights  ###
############################

model = Sequential()
model.add(Embedding(max_fatures + 1, embed_dim, input_length = X_train_emb.shape[1],trainable = True))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out,return_sequences=True,recurrent_dropout=0.4))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(3,activation='softmax'))
print(model.summary())

Y_ints = [Y.argmax() for Y in y_train_emb]
class_weights = {0: (1.3*1387)/303.,1: 1.,2: 1387/569.}
filepath_b = "Classweights.hdf5"
callback = [EarlyStopping(monitor='val_loss', patience=10, verbose = 1),ModelCheckpoint(filepath=filepath_b,
                                                                                        save_best_only = True,
                                                                                        save_weights_only = False)]

model.compile(loss = "categorical_crossentropy", optimizer='Adam',metrics=[precision,recall])
history_Classes = model.fit(X_train_emb, y_train_emb, epochs = EPOCHS, batch_size=BATCH_SIZE,callbacks = callback,
                           class_weight = class_weights,verbose = 2, validation_data=(X_valid_emb, y_valid_emb))

############################
####  LSTM with SMOTE  #####
############################

model = Sequential()
model.add(Embedding(max_fatures + 1, embed_dim, input_length = X_train_emb.shape[1],trainable = True))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out,return_sequences=True,recurrent_dropout=0.4))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(3,activation='softmax'))
print(model.summary())

Y_ints = [Y.argmax() for Y in y_train_emb]
class_weights = {0: (1.3*1387)/303.,1: 1.,2: 1387/569.}

filepath_c = "SMOTE.hdf5"
callback = [EarlyStopping(monitor='val_loss', patience=10, verbose = 1),ModelCheckpoint(filepath=filepath_c,
                                                                                        save_best_only = True,
                                                                                        save_weights_only = False)]
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2) #Include ration for fine tuning
X_train_res, y_train_res = sm.fit_sample(X_train_emb, y_train_emb)#y_train_le
print(data["sentiment"].value_counts(), np.bincount(y_train_res))


model.compile(loss = "categorical_crossentropy", optimizer='Adam',metrics=[precision,recall])
history_SMOTE = model.fit(X_train_res,y_train_res,epochs = EPOCHS,callbacks = callback, batch_size = BATCH_SIZE,
                           class_weight = class_weights, verbose = 2, validation_data=(X_valid_emb, y_valid_emb))
############################
#### LSTM with pretrained ##
#### GloVe Wordembeddings ##
############################

#The file "glove.6B.100d.txt" can be downloaded at : https://nlp.stanford.edu/projects/glove/

#Pretrained wordembeddings:
embeddings_index = {}
f = open('glove.6B.100d.txt',mode = "r") #encoding = "UTF-8"
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

#Constructing embedding matrix:
embedding_matrix = np.zeros((len(word_index) + 1, 100 ))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#Pickle results for easy access:
#with open('embedding_matrix', 'wb') as f:
    #pickle.dump(embedding_matrix, f)

with open('embedding_matrix', 'rb') as f:
    embedding_matrix = pickle.load(f)


glove_model = Sequential()
glove_model.add(Embedding(len(word_index)+1, 100 , input_length=X_train_emb.shape[1], weights=[embedding_matrix],
                          trainable=False))
glove_model.add(SpatialDropout1D(0.4))
glove_model.add(LSTM(lstm_out,return_sequences=True,dropout = 0.4,recurrent_dropout=0.4))
glove_model.add(Flatten())
glove_model.add(Dense(3, activation='softmax'))
print(glove_model.summary())
glove_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[precision,recall])


filepath_d = "Word2Vec.hdf5"
Y_ints = [Y.argmax() for Y in y_train_emb]
class_weights = {0: (1.3*1387)/303.,1: 1.,2: 1387/569.}

callback = [EarlyStopping(monitor='val_loss', patience=10, verbose = 1),ModelCheckpoint(filepath=filepath_d,
                                                                                        save_best_only = True,
                                                                                        save_weights_only = False)]

history_Word2Vec = glove_model.fit(X_train_emb, y_train_emb, epochs = EPOCHS, batch_size=BATCH_SIZE,callbacks = callback,
                                 verbose = 2, class_weight = class_weights,validation_data=(X_valid_emb, y_valid_emb))


#########################################################################################################################

#####################
## Evaluate LSTM  ###
##    Results     ###
#####################

#Input model filepath and evaluate results:
model = load_model(filepath_d,custom_objects={"precision": precision, "recall":recall})
prediction = np.argmax(model.predict(X_test_seq_trunc),axis = 1)
print(classification_report(y_test_le, prediction))
print_results(y_test_le, prediction)


#Checking graphically for overfitting:
def res(a, metric_name, b):
     metric = a.history[metric_name]
     val_metric = a.history['val_' + metric_name]

     e = range(1, 74 + 1) #EPOCHS as 74
     with plt.style.context('Solarize_Light2'):
        plt.plot(e, metric, 'bo', label='Training ' + metric_name)
        plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
        plt.xlabel('epoch', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.legend()
        plt.savefig(metric_name + b)
        plt.close()

model = load_model(filepath_a,custom_objects={"precision": precision, "recall":recall})
res(model, 'loss', 'No_SMOTE_AllagreeBS64_04_02_02')
res(model, 'acc','SMOTE')
