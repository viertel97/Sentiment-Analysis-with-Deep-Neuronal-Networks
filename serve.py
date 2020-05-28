import json

import tensorflow as tf
from classes import cli
from classes.preprocessing import preprocess_testdata
from classes.visualisation import visual_sentiment
from classes.utils import choose_model
from flask import Flask, request, jsonify
from keras.models import load_model as load_keras_model

hp = cli.parse_serve()

# Globale Variablen
app = Flask(__name__)
model = None

HOST = 'localhost'
PORT = hp.port


def load_model():
    global model
    model = load_keras_model(choose_model())
    model._make_predict_function()


@app.route('/predict', methods=['POST'])
def predict():
    # Parsen der Anfrage
    body = request.get_json()

    # Validierung der Anfrage
    if 'utterance' in body:
        utterance = body['utterance']
    else:
        return 'Fehlerhaftes JSON, "utterance" Attribut fehlt', 400

    if 'spell_correction' in body:
        spell_correction_enabled = body['spell_correction']

    # Vorverarbeitung
    sequence = preprocess_testdata(utterance, spell_correction_enabled=False)

    # Vorhersage
    prediction = model.predict(sequence)

    # Rückgabe
    return jsonify({'content': utterance,
                    'sentiment': visual_sentiment(prediction)})


if __name__ == '__main__':
    load_model()
    app.run(threaded=True,host=HOST, port=PORT)
