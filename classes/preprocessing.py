import json
import re

import boto3
import numpy as np
import pandas as pd
from classes.utils import get_path
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, one_hot, text_to_word_sequence
from sklearn.model_selection import train_test_split
from symspellpy.symspellpy import SymSpell

comprehend = boto3.client(service_name='comprehend', region_name='eu-west-1')
tokenizer = Tokenizer(num_words=5000, split=" ")

max_edit_distance_dictionary = 2
prefix_length = 7
sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
dictionary_path = get_path('dictionary.txt')
term_index = 0
count_index = 1
max_edit_distance_lookup = 2

emoji_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
    "+", flags=re.UNICODE)


# Klassifiziert die einzelnen Strings der Liste mithilfe von Amazon Comprehend
def add_sentiment_to_list(list):
    sentiment_list = []
    for content in list:
        if content:
            sentiment_dict = {'Content': content}
            sentiment_dict.update(comprehend.detect_sentiment(Text=content, LanguageCode='de'))
            delete_unwanted_keys(sentiment_dict, True, True)
            sentiment_list.append(sentiment_dict)
    return sentiment_list


def normalize(string_to_normalize):
    lower_string = string_to_normalize.lower()
    return re.sub('[^a-zA-zÄäÜüÖöß\s]', '', lower_string)


# Wendet auf den übergeben Datensatz (String oder Pandas.DataFrame) eine Rechtschreibprüfung an
def spell_correction(data):
    if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
        print("Wörterbuch nicht gefunden.")
        return
    if type(data) is str:
        suggestions = sym_spell.lookup_compound(data, max_edit_distance_lookup)
        if suggestions[0].term:
            return suggestions[0].term.encode('latin1').decode('utf8')
    else:
        for idx, content in enumerate(data):
            suggestions = sym_spell.lookup_compound(content, max_edit_distance_lookup)
            if suggestions[0].term:
                data.at[idx] = suggestions[0].term


# Entfernt die unwichtigen Teile der Antwort von Amazon Comprehend
def delete_unwanted_keys(dict, sentiment_as_value, nested):
    if sentiment_as_value:
        entries_to_remove = ('Sentiment', 'ResponseMetadata')
        for key in entries_to_remove:
            dict.pop(key, None)
        if not nested:
            dict.update(dict['SentimentScore'])
            dict.pop('SentimentScore', None)
    else:
        entries_to_remove = ('SentimentScore', 'ResponseMetadata')
        for key in entries_to_remove:
            dict.pop(key, None)


def delete_emoji(string):
    return emoji_pattern.sub(r'', string)


def delete_duplicates(input_list):
    return list(dict.fromkeys(input_list))


# Teilt den Datensatz zufällig im Verhältnis 80 zu 20 auf
def split_dataset(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y)
    return (x_train, y_train), (x_test, y_test)


def tokenize_training_data(data_frame):
    tokenizer.fit_on_texts(data_frame['Content'].values)
    X = tokenizer.texts_to_sequences(data_frame['Content'].values)
    X = pad_sequences(X, padding='post')
    Y = pd.get_dummies(data_frame['Sentiment']).values
    return X, Y, len(tokenizer.word_index)


def preprocess_testdata(test_data, spell_correction_enabled):
    data_without_emoji = delete_emoji(test_data)
    normalized_data = normalize(data_without_emoji)
    if spell_correction_enabled:
        normalized_data = spell_correction(normalized_data)
    tokenizer.fit_on_texts(normalized_data)
    X = tokenizer.texts_to_sequences(normalized_data)
    X = pad_sequences(X, padding='post')
    return X
