from classes import cli
from classes.data.dataset import load_data
from classes.models import custom_gru_bi
from classes.visualisation import metrics, prediction_real_ratio

hp = cli.parse_train()

# Hyperparameter
HIDDEN_UNITS = hp.units
EMBEDDING_DIM = hp.embedding_dim
EPOCHS = hp.epochs
BATCH_SIZE = hp.batch_size

SPELL_CORRECTION = hp.spell_correction
MODEL_NAME = "sentiment_analysis"

# Datensatz laden
(x_train, y_train), (x_test, y_test), vocabulary_size = load_data(
    spell_correction_enabled=SPELL_CORRECTION
)

# Modell
model = custom_gru_bi(HIDDEN_UNITS, vocabulary_size, EMBEDDING_DIM)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Training
history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
    validation_split=0.2,
)

# Testen
[test_loss, test_accuracy] = model.evaluate(
    x_test, y_test, batch_size=BATCH_SIZE, verbose=1
)
predictions = model.predict(x_test)
prediction_real_ratio(predictions, y_test)

accuracy = round(test_accuracy * 100, 2)
print("Genauigkeit Testdatensatz: " + str(accuracy) + "%")

metrics(predictions, y_test)

# Speichern
save_name = "{0}_{1}hu_{2}ed_{3}e_{4}bs_{5}gru_bi".format(
    MODEL_NAME, str(HIDDEN_UNITS), str(EMBEDDING_DIM), str(EPOCHS), str(BATCH_SIZE),"sc_" if SPELL_CORRECTION else ""
)
model.save(save_name + ".h5")
print('Keras-Modell wurde unter dem Namen "' + save_name + '.h5" gespeichert.')
