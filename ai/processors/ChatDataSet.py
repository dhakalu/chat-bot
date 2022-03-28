from torch.utils.data import Dataset

class ChatDataSet(Dataset):

    def __init__(self, X, Y):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = Y


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]