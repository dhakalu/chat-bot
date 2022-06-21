from tkinter import HIDDEN
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from ChatDataSet import ChatDataSet
from DataUtils import DataUtil
from Model import NeuralNet

DATA_DIRECTORY_PATH = 'ai/data/questions'


class Train:

    def __init__(self, directory_path) -> None:
        self._initialize_hyperparameters()
        self._initialize_data_set(directory_path)
        self._initialize_model_and_device()

    def _initialize_hyperparameters(self):
        # Hyper parameters
        self.BATCH_SIZE = 1
        self.SHUFFLE = True
        self.NUM_OF_WORKERS = 0
        self.HIDDEN_SIZE = 8
        self.OUTPUT_SIZE = len(self.data_set.y_data)
        self.INPUT_SIZE = len(self.data_set.x_data[0])
        self.LEARNING_RATE = 0.001
        self.NUM_EPOCHS = 1000

    def _initialize_data_set(self, directory_path):
        data_util = DataUtil(directory_path)
        X, Y = data_util.get_data()
        self.all_words = data_util.unique_words
        self.intents = data_util.intents
        self.data_set = ChatDataSet(X, Y)

    def _initialize_model_and_device(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralNet(input_size=self.INPUT_SIZE, hidden_size=self.HIDDEN_SIZE,
                               output_size=self.OUTPUT_SIZE).to(self.device)

    def _initialize_loss_and_optimizer(self):
        self.optimizer = Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        self.criterion = CrossEntropyLoss()

    def train(self):
        self._initialize_loss_and_optimizer()
        train_loader = DataLoader(self.data_set, batch_size=self.BATCH_SIZE,
                                  shuffle=self.SHUFFLE, num_workers=self.NUM_OF_WORKERS)

        for epoch in range(self.NUM_EPOCHS):
            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()  # clean up old gradients
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()  # backpropagation
                self.optimizer.step()  # update weights
                if i % 100 == 0:
                    print('Epoch: {}/{}.............'.format(epoch + 1, self.NUM_EPOCHS),
                          'Loss: {:.4f}'.format(loss.item()))

    def save_model(self):
        data = self._get_model_meta_data()
        FILE = "model.pth"
        torch.save(data, FILE)

    def _get_model_meta_data(self):
        return {
            'model': self.model.state_dict(),
            'input_size': self.INPUT_SIZE,
            'hidden_size': self.HIDDEN_SIZE,
            'output_size': self.OUTPUT_SIZE,
            'learning_rate': self.LEARNING_RATE,
            'all_words': self.all_words,
            'intents': self.intents
        }
