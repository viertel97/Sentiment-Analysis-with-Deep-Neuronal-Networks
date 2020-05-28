import glob
import json
import re
import sys

from argparse import ArgumentParser
from os import walk
from os.path import dirname, join, realpath


def get_directory(*path):
    file_path = get_path(*path)
    file_names = get_filenames(file_path)
    for file in file_names:
        data_list = []
        data_list.append(get_file(file))
    return data_list


# Ließt übergebenes JSON Datei in Liste ein und gibt sie zurück
def get_file(*path):
    file_path = get_path(*path)
    with open(file_path, encoding='utf-8') as rawJSON:
        json_file = json.load(rawJSON)
    data_list = []
    for json_data in json_file:
        data_list.append(json_data)
    return data_list


# Gibt absoluten Pfad der mitgegeben Parameter zurück
def get_path(*args):
    dir_name = dirname(__file__)
    path = join(dir_name, '..', *args)
    return realpath(path)


# Gibt alle Dateinamen zurück, die im mitgegeben Pfad liegen
def get_filenames(path):
    files = []
    for (dir_path, dir_names, file_names) in walk(path):
        files.extend(file_names)
        break
    return files


# Auswahl des gewünschten Keras-Modells
def choose_model():
    model_list = glob.glob('*.h5')
    if not model_list:
        print("Kein Modell im Verzeichnis gefunden.")
        sys.exit()
    if len(model_list) == 1:
        print('Es wurde nur das Modell mit dem Namen "{0}" gefunden und auch ausgewählt.'.format(model_list[0]))
        return model_list[0]
    else:
        for idx, model in enumerate(model_list):
            print('{0}: {1}'.format(idx + 1, model))
        print('Welches Modell möchten Sie verwenden?')
        print("Bitte verwenden Sie zahlen zwischen {0} und {1}.".format(str(1), str(len(model_list))))
        while True:
            try:
                choosen_model = int(input())
                if 1 > choosen_model or  choosen_model > len(model_list):
                    raise ValueError
            except ValueError:
                print("Bitte verwenden Sie zahlen zwischen {0} und {1}.".format(str(1), str(len(model_list))))
                continue
            else:
                break
        print('Modell mit Namen "{0}" wurde gewählt.'.format(model_list[choosen_model - 1]))
        return model_list[choosen_model - 1]


def write_to_file(filename, data):
    try:
        open(filename, 'w').close()
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, sort_keys=True, indent=2, ensure_ascii=False)
    except FileNotFoundError:
        with open(filename, 'x', encoding='utf-8') as file:
            json.dump(data, file, sort_keys=True, indent=2, ensure_ascii=False)
