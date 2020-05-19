import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.cuda.amp import autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ENC3D_CNN(nn.Module):
    def __init__(self, model='r3d_18', pretrained=True, z_dimension=128, mode='projection_head'):
        super(ENC3D_CNN, self).__init__()
        self.mode = mode

        enc_3d = torchvision.models.video.__dict__[model]\
        				(pretrained=pretrained)

        self.features = nn.Sequential(*list(enc_3d.children())[:-1])
        
        # self.enc_3d.fc = torch.nn.Linear(self.enc_3d.fc.in_features, z_dimension)
        self.num_ftrs = enc_3d.fc.in_features

        # projection MLP
        self.l1 = nn.Linear(self.num_ftrs, 1)#self.num_ftrs)
        self.l2 = nn.Linear(self.num_ftrs, z_dimension)

        self.sigmoid = nn.Sigmoid()

    @autocast()
    def forward(self, x):

        x = self.features(x) #512
        x = x.squeeze()

        # print("output of cnn shape: ", x.shape)

        if self.mode == 'projection_head':
            h = self.l1(x)
            h = F.relu(h)
            h = self.l2(h)

        x = self.l1(x)
        x = self.sigmoid(x)
        return x



def construct_3d_enc(model='r3d_18', pretrained=True, z_dimension=128, mode='projection_head'):
	return ENC3D_CNN(model, pretrained, z_dimension, mode)

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.bn_input = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.bn1 = nn.BatchNorm1d(int(hidden_size/2))
        self.fc2 = nn.Linear(int(hidden_size/2), num_classes)
        self.act = nn.Sigmoid()
    
    # @autocast()
    def forward(self, x, lengths): # x: shape(batch_size, seq_length, feature size)
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # print(x.shape)
        # x = F.relu(self.bn_input(x))
        x = pack_padded_sequence(x, lengths.cpu().numpy(), batch_first=True).to(device) 
        
        # print("pack_padded_sequence shape: ", x.shape)
        # print("pack_padded_sequence: ", x)
        # Forward propagate LSTM
        # with autocast():
        
        out, (hn, _) = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # out, _ = pad_packed_sequence(out, batch_first=True)

        # out = out.contiguous()
        # print("hn of rnn ", hn[-1, :, :].shape)
        # print("output of rnn ", out)
        
        # Decode the hidden state of the last time step
        out = self.bn1(self.fc1(hn[-1, :, :]))
        out = F.relu(out)
        out = F.dropout(out, 0.2, training=self.training)
        # out = F.relu(self.fc1(x))
        out = self.act(self.fc2(out))
        # print("output of rnn shape ", out.shape)
        
        return out

def construct_rnn(input_size = 512, hidden_size=256, num_layers=2, num_classes=1):
    return RNN(input_size, hidden_size, num_layers, num_classes)


