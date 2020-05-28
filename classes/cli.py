from argparse import ArgumentParser

parser = ArgumentParser()


def parse_train():
    parser.add_argument('-u', '--units', nargs='?', default=64, type=int, help='Anzahl der GRU Einheiten')
    parser.add_argument('-em', '--embedding_dim', nargs='?', default=32, type=int,
                        help='Anzahl der Dimensionen des Embedding Vektors')
    parser.add_argument('-ep', '--epochs', nargs='?', default=10, type=int, help='Anzahl der Epochen')
    parser.add_argument('-b', '--batch_size', nargs='?', default=32, type=int, help='Größe der Batches')
    parser.add_argument('-sc', '--spell_correction', nargs='?', default=False, type=str2bool,
                        help='Rechtschreibprüfung aktivieren')
    return parser.parse_args()


def parse_predict():
    parser.add_argument('-sc', '--spell_correction', nargs='?', default=False, help='Rechtschreibprüfung aktivieren')

    return parser.parse_args()


def parse_serve():
    parser.add_argument('-p', '--port', nargs='?', default=5000, help='Port des Webservers')

    return parser.parse_args()


# Transformiert jegliche boolean-ähnliche Begriffe in einen Boolean
def str2bool(string):
    if isinstance(string, bool):
        return string
    if string.lower() in ('yes', 'ja', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'nein', 'false', 'f', 'n', '0'):
        return False
    else:
        return False
