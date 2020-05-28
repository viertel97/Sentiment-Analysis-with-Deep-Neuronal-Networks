from keras.layers import Activation, Bidirectional, Dense, Dropout, Embedding, GRU, LSTM
from keras.models import Sequential


def custom_gru(hidden_units, vocabulary_dim, embedding_dim):
    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_dim, output_dim=embedding_dim))
    model.add(Dropout(0.3))
    model.add(GRU(units=hidden_units, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    return model


def custom_gru_bi(hidden_units, vocabulary_dim, embedding_dim):
    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_dim, output_dim=embedding_dim))
    model.add(Dropout(0.8222982387634797))
    model.add(Bidirectional(GRU(units=hidden_units, return_sequences=False)))
    model.add(Dropout(0.5827946981839023))
    model.add(Dense(3, activation='softmax'))
    return model
