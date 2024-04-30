import torch
import numpy as np
from torch.utils.data import DataLoader

# TODO: normalization
class SamplesDataset(torch.utils.data.Dataset):

    def __init__(self, path):
        super().__init__()
        # load dataset stored in npz format.
        self.data_dict = [dict(data_key.item()) for data_key in np.load(path, allow_pickle=True).values()]
        self.data = [spectrum_info for spectrum_info in self.data_dict[0].values()]
        self.statistics = self.data_dict[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id = self.data[idx]['id']
        x = torch.tensor(self.data[idx]['x']).to(torch.float32)  # (500, 3)
        y = torch.tensor(self.data[idx]['y']).to(torch.float32)  # (500, 1)
        training_example = (x, y)

        return id, training_example

    def get_statistics(self):
        return self.statistics

def get_data(path, batch_size, shuffle=True):
    dataset = SamplesDataset(path)
    statistics = dataset.get_statistics()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), statistics

