import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset


class Dense(nn.Module):

    def __init__(self, input_channels, out_channels):
        super(Dense, self).__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.fc = nn.Sequential(
            nn.Linear(self.input_channels, 768),
            nn.ReLU(),
            nn.Linear(768,256),
            nn.ReLU(),
            nn.Linear(256, self.out_channels),
            nn.Dropout(0.1)
            # nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.fc(x)
        return output


class Data_training(Dataset):
    def __init__(self, df, vec, index):
        self.df = df
        self.vec = vec
        self.index = index

    def __getitem__(self, idx):
        source = self.vec[idx]
        target = np.float32(self.df.iloc[idx, self.index])
        return source, target

    def __len__(self):
        return len(self.df)


