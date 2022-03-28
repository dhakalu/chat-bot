import random
import torch 
import os
from ai.processors.Model import NeuralNet
from ai.utils.nlp import get_words_in_sentence, get_bag_of_words
from ai.utils.file import read_file_content


class ChatBot: 

    def __init__(self, model_file_path):
        data = torch.load(model_file_path)
        self.HIDDEN_SIZE = data['hidden_size']
        self.INPUT_SIZE = data['input_size']
        self.OUTPUT_SIZE = data['output_size']
        self.all_words = data['all_words']
        self.intents = data['intents']
        self._initialize_model(data)


    def start_chat_ui(self):
        while True:
            sentence = input("You: ")
            if (sentence == "exit"):
                break
            intent = self._predict(sentence)
            self._generate_response_for_intent(intent)

    def _initialize_model(self, data):
        model = NeuralNet(input_size=self.INPUT_SIZE, hidden_size=self.HIDDEN_SIZE, output_size=self.OUTPUT_SIZE)
        model.load_state_dict(data['model'])
        model.eval()
        self.model = model


    def _predict(self, sentence):
        words = get_words_in_sentence(sentence)
        X = get_bag_of_words(self.all_words, words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)
        output = self.model(X)
        _, predicted = torch.max(output.data, dim=1)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        probabilities = probabilities.data.numpy()[0]
        if probabilities[predicted] > 0.5:
            return self.intents[predicted]
        return None

    def _get_response(self, intent):
        possible_responses = read_file_content('ai/__data__/responses/' + intent + '.txt')
        random_index = random.randint(0, len(possible_responses) - 1)
        return possible_responses[random_index]

    def _generate_response_for_intent(self, intent):
        if intent is not None:
            response = self._get_response(intent)
            print("Bot: " + response)
        else:
            print("Bot: Oops! I do not understand!!" )

