import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask

app = Flask(__name__)

@app.route('/')
def sarcasm_machine_learning():
    vocab_size = 10000
    embedding_dim = 16
    max_length = 100
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    with open('Sarcasm_Headlines_Dataset.json', 'r') as f:
        datastore = json.loads("[" + 
            f.read().replace("}\n{", "},\n{") + 
        "]")

    sentences = []
    labels = []
    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

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

    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_' + string])
        plt.xlabel('Epochs')
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.show()

    sentence = [
        'granny starting to fear spiders in the garden might be real',
        'the weather today is bright and sunny',
        'oh, yeah right',
        'of course you did',
        'During the course of the day you did',
        'Did you really',
        'I bet you did',
        'postman fears local dog'
    ]
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    sarcastic_value = model.predict(padded)
    is_sentence_sarcastic = []
    for i in sarcastic_value:
        if i > 0.5:
            is_sentence_sarcastic.append('Sarcastic,  Percentage: ' + str(i))
        else:
            is_sentence_sarcastic.append('Not sarcastic, Percentage: ' + str(i))

    dict_is_sentence_sarcastic = { i : is_sentence_sarcastic[i] for i in range(0, len(is_sentence_sarcastic))}

    print(dict_is_sentence_sarcastic)
    return dict_is_sentence_sarcastic

app.run(port=5000)