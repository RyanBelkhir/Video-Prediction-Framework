import torch
from torch import nn

class ConvLSTM3D(nn.Module):
    def __init__(self):
        super(ConvLSTM3D, self).__init__()
        self.conv3D = nn.Conv3D()

# First model, using convolution and lstm for video prediction
class LstmCNN(nn.Module):
    def __init__(self):
        super(LstmCNN, self).__init__()
        self.lstm = [nn.LSTM(10, )]