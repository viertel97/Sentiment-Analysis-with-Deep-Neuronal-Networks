
# Projekt: RNN Sentiment Detection

## Vorraussetzungen

Zur Verwendung des nachfolgenden Systems muss mindestens Python 3.7 64 Bit und pip installiert sein. Die benötigten Bibliotheken werden mit `pip install -r requirements.txt` installiert.
___
## Bedienung

Trainieren des Modells:
`python train.py [-u , --units <UNITS> -sc / -em , --embedding_dim <EMBEDDING_DIM> / -ep , --epochs <EPOCHS> /  -b , --batch_size <BATCH_SIZE> / --spell_correction true|false]`

Erkennung des Sentiments über Nutzereingabe (CLI):
`python predict.py [-sc , --spell_correction true|false]`

Erkennung des Sentiments über einen Webserver (REST-API):
`python serve.py [-p , --port <PORT>]`

___
## Ablauf

### Training
Als ersten Schritt muss das Modell trainiert werden. Der verwendete Datensatz zur Verfügung stehende Datensatz wurde bereits vorverarbeitet und kann mit dem folgenden Befehl dem neuronalen Netz gelehrt werden:

```bash
python train.py --units 256 --embedding 256 --epochs 25 --batch_size 32 --spell_correction True
```

Nach dem Training wird das gesamte Modell in der Datei `sentiment_model.h5` gespeichert und kann für zukünftige Vorhersagen verwendet werden.
Das mit mitgelieferte Modell, wurde mit den selben Parametern trainiert, wie im Beispiel angegeben.


### Vorhersage

Die Vorhersagen über die CLI laufen in einer Schleife, wodurch mehrere Vorhersagen hintereinander möglich sind. Das Programm kann mit der Tastenkombination `STRG+C` beendet werden. 


```bash
python predict.py --spell_correction true
```

### Schnittstelle

Um das neuronale Netz auch produktiv zu nutzen, können mithilfe eines Webservers REST-API Anfragen verarbeitet werden. 

```bash
python serve.py --port 5000
```
___