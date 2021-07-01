from fastapi import FastAPI, Form
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

from nltk.translate.bleu_score import sentence_bleu
documents = []


app = FastAPI()

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
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result.append ({"kelas": i['tag'], "respon": random.choice(i['responses']), "konteks": (i['context']), "probability": (ints[0]['probability']*100)})
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

@app.post("/")
async def chat_bot(msg: str = Form(...)):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    score_steam = bleu_steam(intents, msg)
    score = bleu_steam(intents, msg)
    return {"messages": res[0]['respon'], "kelas": res[0]['kelas'], "konteks": res[0]['konteks'][0], "akurasi": "%.2f" % float(res[0]['probability']), "score_bleu_dengan_steaming": "%.2f" % float(score_steam*100), "score_bleu_tanpa_steaming": "%.2f" % float(score*100)}
