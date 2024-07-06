import json
import numpy as np
import music21 as m21
from tensorflow import keras
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

class MelodyGenerator:

    def __init__(self, model_path='./model.h5'):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, 'r') as mapping_file:
            self._mapping = json.load(mapping_file)

        self._start_symbols = ['/'] * SEQUENCE_LENGTH


    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        
        # seed as start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # conver seed into int with mapping
        seed = [self._mapping[symbol] for symbol in seed]

        for _ in range(num_steps):

            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # one hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mapping))
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            output_int = self._sample_with_temperature(probabilities, temperature)

            # update seed
            seed.append(output_int)

            # map int to our encoding
            output_symbol = [k for k, v in self._mapping.items() if v == output_int][0]

            # check whether at the end of a melody
            if output_symbol == '/':
                break

            # update the melody
            melody.append(output_symbol)

        return melody


    def _sample_with_temperature(self, probabilities, temperture):
        prediction = np.log(probabilities) / temperture
        probabilities = np.exp(prediction) / np.sum(np.exp(prediction))

        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index
    

    def save_melody(self, melody, step_duration=0.25, format='midi', file_name='melody.midi'):

        # create music21 stream
        stream = m21.stream.Stream()

        # parse all the sybols in the melody and create note/rest objects
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):

            # note or rest
            if symbol != '-' or i + 1 == len(melody):
                # make sure the symbol is note or rest
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter
                    # rest
                    if start_symbol == 'r':
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    # note
                    else:m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset stap_counter
                    step_counter = 1

                start_symbol = symbol

            # '-'
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)



if __name__ == '__main__':
    mg = MelodyGenerator()
    seeds = ['65 - - - - - 65 - 64 - 64 - 69 - 67 - 67 - -',
             '67 - 66 - 67 - 69 - 67 - 62 - 62 - 62 - 67 - 66 - 67 - 69 - 67',
             '72 - - - 69 -']
    for i, seed in enumerate(seeds):
        melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.8)
        mg.save_melody(melody, file_name=f'./generated_melody/melody{i+1}.midi')
