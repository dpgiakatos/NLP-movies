from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
from model import Model
from sklearn.model_selection import train_test_split
import nltk
import numpy as np
from data_prepro import Dataset

dataset = Dataset(punctuation=True, stop_words=True, tokenization=False)
data = dataset.get_dataset()
print(data.columns)
texts = data['Overview'].tolist()  # define documents
labels = np.array(data['Western'].tolist())
# labels = array([1,1,1,1,1,0,0,0,0,0]) # define class labels
print(labels)

tokenizer = Tokenizer()  # initialize tokenizer (embedding look up table technique)
tokenizer.fit_on_texts(texts)  # transform each word using the lookup table technique (each word transforms into a unique number for each unique word)
vocabulary_size = len(tokenizer.word_index) + 1  # setting in the vocabulary size hyperparameter
sequences = tokenizer.texts_to_sequences(texts)  # transform into sequences
train_data = pad_sequences(sequences, maxlen=1000)  # pad the sequences with zeros (creating same length data to pass)
x_train, x_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.3, random_state=42)  # split into training and testing set

# y_train = to_categorical(y_train, 10)  # transform the labels into one hot encoded vectors (keras needs this for categorical crossentropy evaluation metric)
# y_test = to_categorical(y_test, 10)

# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocabulary_size, 100))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model = Model()  # initialize the model
model.create_model(vocabulary_size, embedding_matrix)  # create the model
model.train(x_train, y_train, x_test, y_test)  # train the model with our data
model.test(x_test, y_test)  # evaluate the trained model using the test set
