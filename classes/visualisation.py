import statistics as s

import numpy as np
import pandas as pd


# Berechnet die Anzahl der korrekt zugeordneten Sentiments dar
def prediction_real_ratio(predictions, real):
    pos_count, neu_count, neg_count = 0, 0, 0
    real_pos, real_neu, real_neg = 0, 0, 0
    for i, prediction in enumerate(predictions):
        if np.argmax(prediction) == 2:
            pos_count += 1
        elif np.argmax(prediction) == 1:
            neu_count += 1
        else:
            neg_count += 1

        if np.argmax(real[i]) == 2:
            real_pos += 1
        elif np.argmax(real[i]) == 1:
            real_neu += 1
        else:
            real_neg += 1

    temp_data_frame = pd.DataFrame({'Positiv': [pos_count, real_pos],
                                    'Neutral': [neu_count, real_neu],
                                    'Negativ': [neg_count, real_neg]})
    temp_data_frame.rename(index={0: 'Vorhersagen', 1: 'Wirklichkeit'}, inplace=True)

    print("Verhältnis korrekt klassifizierter Datensätze:")
    print(temp_data_frame)


def visual_sentiment(prediction):
    polarity = [0, 0, 0]
    for i in prediction:
        for idx, j in enumerate(i):
            polarity[idx] += j
    index = polarity.index(max(polarity))

    if index == 2:
        return "POSITIV"
    elif index == 1:
        return "NEUTRAL"
    else:
        return "NEGATIV"


# Berechnet die Präzision, Trefferquote und das F1 und stellt es dar
def metrics(predictions, real):
    number_of_classes = 3
    pos_tp, pos_fn, pos_fp, neu_tp, neu_fn, neu_fp, neg_tp, neg_fn, neg_fp = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i, prediction in enumerate(predictions):
        if np.argmax(prediction) == 2 and np.argmax(real[i]) == 2:
            pos_tp += 1
        elif not np.argmax(prediction) == 2 and np.argmax(real[i]) == 2:
            pos_fn += 1
        elif np.argmax(prediction) == 2 and not np.argmax(real[i]) == 2:
            pos_fp += 1

        if np.argmax(prediction) == 1 and np.argmax(real[i]) == 1:
            neu_tp += 1
        elif not np.argmax(prediction) == 1 and np.argmax(real[i]) == 1:
            neu_fn += 1
        elif np.argmax(prediction) == 1 and not np.argmax(real[i]) == 1:
            neu_fp += 1

        if np.argmax(prediction) == 0 and np.argmax(real[i]) == 0:
            neg_tp += 1
        elif not np.argmax(prediction) == 0 and np.argmax(real[i]) == 0:
            neg_fn += 1
        elif np.argmax(prediction) == 0 and not np.argmax(real[i]) == 0:
            neg_fp += 1

    pos_p = safe_div(pos_tp, (pos_tp + pos_fp))
    pos_r = safe_div(pos_tp, (pos_tp + pos_fn))
    neu_p = safe_div(neu_tp, (neu_tp + neu_fp))
    neu_r = safe_div(neu_tp, (neu_tp + neu_fn))
    neg_p = safe_div(neg_tp, (neg_tp + neg_fp))
    neg_r = safe_div(neg_tp, (neg_tp + neg_fn))

    pos_f1 = s.harmonic_mean([pos_p, pos_r])
    neu_f1 = s.harmonic_mean([neu_p, neu_r])
    neg_f1 = s.harmonic_mean([neg_p, neg_r])

    p = (pos_p + neu_p + neg_p) / number_of_classes
    r = (pos_r + neu_r + neg_r) / number_of_classes
    f1 = s.harmonic_mean([p, r])

    print("Präzision: \t" + dec_to_per(p))
    print("Trefferquote: \t" + dec_to_per(r))
    print("F1-Maß: \t" + "{0:.4}".format(f1 * 100.0))

    temp_data_frame = pd.DataFrame(
        {'Positiv': [pos_tp, pos_fn, pos_fp, dec_to_per(pos_p), dec_to_per(pos_r), "{0:.4}".format(pos_f1 * 100.0)],
         'Neutral': [neu_tp, neu_fn, neu_fp, dec_to_per(neu_p), dec_to_per(neu_r), "{0:.4}".format(neu_f1 * 100.0)],
         'Negativ': [neg_tp, neg_fn, neg_fp, dec_to_per(neg_p), dec_to_per(neg_r), "{0:.4}".format(neg_f1 * 100.0)]})
    temp_data_frame.rename(
        index={0: 'True Positive', 1: 'False Negative', 2: 'False Positive', 3: 'Präzision', 4: 'Trefferquote',
               5: 'F1-Maß'}, inplace=True)
    temp_data_frame = temp_data_frame.round(2)

    print("Metriken:")
    print(temp_data_frame)


def dec_to_per(dec):
    return "{0:.2%}".format(dec)


def safe_div(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return 0.0
