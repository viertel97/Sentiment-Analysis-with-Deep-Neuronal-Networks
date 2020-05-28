import json
import re

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pd
from classes.data.dataset import get_processed_dataset, get_raw_dataset, load_data, raw_to_processed_dataset
from classes.preprocessing import delete_emoji, delete_unwanted_keys, normalize, preprocess_testdata, spell_correction
from classes.utils import get_path, write_to_file
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperas.utils import eval_hyperopt_space
from hyperopt import STATUS_OK, Trials, rand, tpe
from keras.layers import Activation, Bidirectional, Dense, Dropout, Embedding, GRU, LSTM
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


def data():
    (x_train, y_train), (x_test, y_test), input_dim = load_data(spell_correction_enabled=False)
    epochs = 'X'
    embedding_dim = 32
    hidden_units = 256
    batch_size = 512
    max_evals = 10
    return x_train, y_train, x_test, y_test, epochs, embedding_dim, input_dim, hidden_units, batch_size, max_evals


def create_model(x_train, y_train, x_test, y_test, epochs, embedding_dim, input_dim, hidden_units, batch_size):
    model_name = 'EPOCHS!automatische_hpo_{0}hu_{1}ed_{2}e_{3}bs_gru_bi'.format(str(hidden_units), str(embedding_dim)
                                                                                , str(epochs), str(batch_size))
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dim))
    model.add(Dropout(0.8222982387634797))
    model.add(Bidirectional(GRU(units=hidden_units, return_sequences=False)))
    model.add(Dropout(0.5827946981839023))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    result = model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs={{choice(list(np.arange(74, 78, 1)))}},
                       verbose=2,
                       validation_split=0.2)
   
    hist_df = pd.DataFrame(result.history)
    with open('automatische_hpo.json', mode='a') as f:
        f.write('"' + model_name + '":')
        hist_df.to_json(f)
        f.write("\n")
        f.close()

    acc = np.amax(result.history['acc'])
    loss = np.amin(result.history['loss'])
    validation_acc = np.amax(result.history['val_acc'])
    validation_loss = np.amin(result.history['val_loss'])

    print('Best training loss of epoch:', loss)
    print('Best training acc of epoch:', acc)
    print('Best validation acc of epoch:', validation_acc)
    print('Best validation loss of epoch:', validation_loss)

    # -validation_acc, damit die Validierungs-Genauigkeit optimiert wird
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


x_train, y_train, x_test, y_test, epochs, embedding_dim, input_dim, hidden_units, batch_size, max_evals = data()
best_run, best_model = optim.minimize(model=create_model,
                                             data=data,
                                             algo=rand.suggest,
                                             max_evals=max_evals,
                                             trials=Trials())
print("Evalutation of best performing model:")
[test_loss, test_accuracy] = best_model.evaluate(x_test, y_test)
print(test_loss, test_accuracy)
print("Best performing model chosen hyper-parameters:")
print(best_run)

save_name = '{0}_{1}hu_{2}ed_{3}e_{4}bs_{5}me_gru_bi'.format('hpo', str(hidden_units), str(embedding_dim),
                                                             str(epochs),
                                                             str(batch_size), str(max_evals))

best_model.save('HPO/' + save_name + '.h5')

best_run['test_loss'] = test_loss
best_run['test_accuracy'] = test_accuracy
best_run['name'] = save_name
best_run['algo'] = 'rand.suggest'
with open('hpo.json', 'a') as json_file:
    json.dump(best_run, json_file)
    json_file.write(",\n")
    json_file.close()
