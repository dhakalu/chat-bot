
import os
import numpy as np

from ai.utils.nlp import get_words_in_sentence
from ai.utils.file import read_file_content

class DataUtil:

    def __init__(self, data_directory_path) -> None:
        self.directory_path = data_directory_path

    def _getListOfFiles(self):
        """
        Get list of files in a directory.
        """
        files = os.listdir(os.path.realpath(self.directory_path))
        if (len(files) == 0):
            raise Exception('No data found in data directory')
        return files

    def _read_file_content(self, file_name):
        """
        Read file content and return it as string array with each line as an element.
        """
        file_path = os.path.join(self.directory_path, file_name)
        return read_file_content(file_path)

    def _load_data(self):
        files = self._getListOfFiles()
        data = {}
        for file in files:
            intent = file.split('.')[0]  # name of the file is the intent
            fileContent = self._read_file_content(file)
            data[intent] = fileContent
        return data

    def _process_data(self):
        # Uniqueue words in all the intents
        words = []
        # All the possible intents in our data
        intents = []

        # List of tuple of [([<list of words in utterance>], '<string representing the intent>'])
        word_to_intent = []

        data = self._load_data()
        for intent in data:
            if intent not in intents:
                intents.append(intent)
            for utterance in data[intent]:
                new_words = get_words_in_sentence(utterance)
                words.extend(new_words)
                word_to_intent.append((new_words, intent))
        self.unique_words = list(set(words))
        self.intents = intents
        self.word_to_intent = word_to_intent

    def get_bag_of_words(self, words):
        bag = np.zeros(len(self.unique_words), dtype=np.float32)
        for idx, word in enumerate(self.unique_words):
            if word in words:
                bag[idx] = 1
        return bag

    def get_data(self):
        self._process_data()
        X = []
        Y = []
        for w in self.word_to_intent:
            [utterance_words, intent] = w
            utterance_bag = self.get_bag_of_words(utterance_words)
            X.append(utterance_bag)
            Y.append(self.intents.index(intent))
        return np.array(X), np.array(Y)
