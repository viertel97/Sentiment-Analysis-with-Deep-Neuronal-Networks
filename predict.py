import glob

from classes import cli
from classes.preprocessing import preprocess_testdata
from classes.utils import choose_model
from classes.visualisation import visual_sentiment
from keras.models import load_model

# Hyperparameter
hp = cli.parse_predict()
SC = hp.spell_correction

# Laden des Modells
model = load_model(choose_model())

print('Die Sentiment Analyse kann mit STRG + C beendet werden.')
while True:
    # Entgegennahme der Eingabe
    print('Ihre zu analysierende Eingabe: ', end='')
    utterance = input()

    # Vorverarbeitung der Eingabe
    tokens = preprocess_testdata(utterance, spell_correction_enabled=SC)

    # Vorhersage des Sentiments
    prediction = model.predict(tokens)

    # Visualisierung des Sentiments
    print('Der Nutzereingabe "' + utterance + '" wurde das folgende Sentiment zugewiesen: ' + visual_sentiment(
        prediction))
