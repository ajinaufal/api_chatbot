import nltk
nltk.download('punkt')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import Adagrad
from keras.optimizers import Adamax
import random

from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))
        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])    
                  
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print (len(documents), "documents")

print (len(classes), "classes")

print (len(words), "unique lemmatized words")


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

print(model.summary())
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
adx = Adamax(learning_rate=0.02)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=adx, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)


# # summarize history for accuracy
# plt.plot(hist.history['accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# # summarize history for loss
# plt.plot(hist.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

print("model created")

import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()
import pickle
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import time

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
documents = []
msg = ["siapa kamu?", "Apakah Kamu tahu tentang poin?", "Apakah Kamu berbohong?", "apakah kamu tahu tentang pembelian poin?", "saya punya keluhan tentang poin pembelian yang belum masuk."]


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": r[1]})
    return return_list

def getResponse(ints, intents_json):
    result = []
    for intents in ints:
        intent= [intents]
        tag = intent[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result.insert ({"kelas": i['tag'], "respon": random.choice(i['responses']), "konteks": i['context'], "probability": (ints[0]['probability']*100)})
                break
    return result

def bleu_steam(intents, sentence):
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # take each word and tokenize it
            sentence_words = nltk.word_tokenize(pattern)
            sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
            documents.append((sentence_words))
    candidate = clean_up_sentence(sentence)
    score = sentence_bleu(documents, candidate, weights=(1, 0, 0, 0))
    return score

def bleu(intents, sentence):
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # take each word and tokenize it
            sentence_words = nltk.word_tokenize(pattern)
            documents.append((sentence_words))
    candidate = clean_up_sentence(sentence)
    score = sentence_bleu(documents, candidate, weights=(1, 0, 0, 0))
    return score

def chatbot_response(msg):
    ints = []
    res = []
    for mess in msg:
        start_time = time.time()
        ints.extend (predict_class(mess, model))
        # res.append (getResponse(ints, intents))
        # cek = bleu(intents, msg)
        # score_steam = bleu_steam(intents, msg)
        # score = bleu_steam(intents, msg)
        # print("score_bleu_dengan_steaming :","%.2f" % float(score*100))
        # print("--- %s seconds ---" % (time.time() - start_time))
        # print(res)
    return ints

print(chatbot_response(msg))
# ints = chatbot_response(msg)
# result = []
# intent = []
# i = 0
# for intents in ints:
#     intent.extend(intents)
#     print(intent)
    # print(intents)
#     i+=1
    # tag = intent[0]['intent']
    # list_of_intents = intents_json['intents']
    # for i in list_of_intents:
    #     if(i['tag']== tag):
    #         result.append ({"kelas": i['tag'], "respon": random.choice(i['responses']), "konteks": i['context'], "probability": (ints[0]['probability']*100)})
    #         break
