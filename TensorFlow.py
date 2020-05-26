import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, jsonify, request, redirect, url_for
from settings import *
from DataModel import *

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
data_size = 1037535
training_size = 800000

datastore = MachineLearningData.get_all_data()

sentences = []
labels = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
for i in sentences:
     if not isinstance(i, str):
         sentences.remove(i)

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length,
    padding=padding_type, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length,
    padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_batch_size=(testing_padded, testing_labels), verbose=2)

@app.route('/ml', methods=['GET', 'POST'])
def sarcasm_machine_learning():    
    if request.method == 'POST':
        user_input_sentence = request.json
        sentence_list = [user_input_sentence]
        sequence = tokenizer.texts_to_sequences(sentence_list)
        padded = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        sarcastic_value = model.predict(padded)
        if sarcastic_value > 0.8:
            return_value = 'Sarcastic'
            return jsonify({'isSarcastic': return_value})
        elif sarcastic_value > 0.5:
            return_value = 'Unsure'
            return jsonify({'isSarcastic': return_value})
        else:
            return_value = 'Not Sarcastic'
            return jsonify({'isSarcastic': return_value})
    else:
        return jsonify({'isSarcastic': ''})

@app.route('/test')
def get_json_test():
    return jsonify({'data': MachineLearningData.get_all_data()})


app.run(port=5000)