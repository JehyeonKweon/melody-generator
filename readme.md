## Overview

**Project Title**: Melody Generator

**Project Description**: use forksong data to train lstm model and give the model short note and make it to guess next note.

**Project Goals**: Learn how to use lstm model

## Instructions for Build and Use

Steps to build and/or run the software:

1. preprocess.py
    * Go through path and convert krn files to text file and save them
    * Go through the text files and conbine them into one text file
    * Create json file that mapping every unique symbol that represent note, rest, duration
    * Convert one text file songs to intager with the mapping json file and create numpy array for training feature and target
2. train.py
    * Create deep learning model with LSTM(Long-Short-Term-Memory) layer
    * Train the model with the inputs from the preprocess
    * Save model as .h5 file
3. melody_generator.py
    * Load the trained model
    * Let model make predicts with seed (random notes in intager)
    * Conver the result numpy array into music21 object
    * Save the object as .midi file in path

Instructions for using the software:

1. Run melody_generator.py file.
2. Run .midi files in generated_melody folder with MuseScoreStudio 4
3. Play the melody

## Development Environment 

To recreate the development environment, you need the following software and/or libraries with the specified versions:

* VScode
* Python 3.12.3
    * tensorflow.keras
    * music21
    * json
    * numpy
* MuseScoreStudio4

## Useful Websites to Learn More

I found these websites useful in developing this software:

* [Youtube](https://www.youtube.com/watch?v=FLr0r-QhqH0&list=PL-wATfeyAMNr0KMutwtbeDCmpwvtul-Xz)
* [ChatGPT](https://chatgpt.com/auth/login)
* [w3schools](https://www.w3schools.com)
* [tutorialspoint](https://www.tutorialspoint.com/index.htm)
* [esac-data](http://www.esac-data.org/)

## Future Work

The following items I plan to fix, improve, and/or add to this project in the future:

* [ ] Add chords
