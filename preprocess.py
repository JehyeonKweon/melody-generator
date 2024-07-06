import os
import json
import music21 as m21
from tensorflow import keras
import numpy as np

DATASET_PATH = './europa/deutschl/erk'
SAVE_DIR = './dataset'
SINGLE_FILE_DATASET = './single-file-dataset.txt'
MAPPING_PATH = './mapping.json'
SEQUENCE_LENGTH = 64
ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0
]


def load_songs_in_kern(dataset_path):

    songs = []

    # go through all the files in dataset and load them with music21
    for path, subdir, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == 'krn':
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    songs[0].show()
    return songs


def check_acceptable_durations(song, acceptable_durations):
    for note in song.flatten().notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True
        

def transpose(song):
    # get keys from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze('key')

    # get interval for transposition E.g., Bmaj -> Cmaj
    if key.mode == 'major':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('C'))

    elif key.mode == 'minor':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('A'))

    # transpose song by calculated interval
    transposed_song = song.transpose(interval)

    return transposed_song


def encoding_song(song, time_step=0.25):
    # pitch = 60, duration = 1.0 -> [60, '-', '-', '-']
    encoded_song = []

    for event in song.flatten().notesAndRests:

        # handle note
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # 60
        # handle rest
        elif isinstance(event, m21.note.Rest):
            symbol = 'r'

        # convert the nete/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)

        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append('-')

    # encoded_song to str
    encoded_song = ' '.join(map(str, encoded_song))

    return encoded_song


def preprocess(path):

    # load the folk song
    print('Loading songs...')
    songs = load_songs_in_kern(path)
    print(f'Loaded {len(songs)} songs.')

    for i, song in enumerate(songs):
        # filter out songs that have non-acceptable duration
        if not check_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue

        # transpose songs to Cmaj or Amin
        song = transpose(song)

        # encode songs with music time series representation
        encoded_song = encoding_song(song)

        # save song to text file
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path+'.txt', 'w', encoding='utf-8') as file:
            file.write(encoded_song)


def load(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        song = file.read()
    return song


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = '/ ' * sequence_length
    songs = ""
    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + ' ' + new_song_delimiter
    
    songs = songs[:-1]

    # save string of dataset
    with open(file_dataset_path, 'w', encoding='utf-8') as w_file:
        w_file.write(songs)

    return songs


def create_mapping(songs, mapping_path):
    mapping = {}
    
    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))
    
    # create mappings
    for i, symbol in enumerate(vocabulary):
        mapping[symbol] = i

    # save vocabulary to a json file
    with open(mapping_path, 'w') as file:
        json.dump(mapping, file, indent=4)


def convert_songs_to_int(songs):
    int_song = []

    # load mappings
    with open(MAPPING_PATH, 'r') as map_file:
        mapping = json.load(map_file)

    # cast songs string to a list
    songs = songs.split()

    # map songs to int
    for symbol in songs:
        int_song.append(mapping[symbol])

    return int_song


def generate_training_sequences(sequence_length):
    # [11, 12, 13, 14, ...] -> i: [11, 12], t: 13; i:[12, 13], t: 14
    
    # load songs and map them to int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    # generate the training sequences
    # 100 symbols, 64 sl, 100 - 64 = 36
    inputs = []
    targets = []

    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length]) # inputs: (num_sequences, sequences_length)
        targets.append(int_songs[i+sequence_length])

    # one-hot encode the sequences
    # [[0,1,2], [1,1,2]] -> [[[1,0,0],[0,1,0],[0,0,1]],[[0,1,0],[0,1,0],[0,0,1]]]
    # inputs: (num_sequences, sequences_length, vocabulary_size)
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets


def main():
    preprocess(DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    # print(inputs)
    # print(targets)


if __name__ == '__main__':
    main()