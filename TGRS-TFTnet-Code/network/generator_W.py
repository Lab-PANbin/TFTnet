import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F


class AttentionModule_W(nn.Module):
    def __init__(self, input_channels):
        super(AttentionModule_W, self).__init__()
        # self.conv1 = nn.Conv2d(input_channels, 1, kernel_size=1)
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=0)
        self.fc1 = nn.Linear(input_channels, input_channels)
        self.fc2 = nn.Linear(input_channels, input_channels)
        self.fc3 = nn.Linear(input_channels, input_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm2d(input_channels)
        self.dropout = nn.Dropout(0.5)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_ED, x_noisy):
        x_concat = torch.add(torch.add(x, x_ED), x_noisy)
        # x_concat = torch.add(torch.add(x, x_ED), 0.1 * x_noisy)

        x_concat_1 = self.relu1(self.conv1(x_concat))
        x_concat_2 = self.relu2(self.conv2(x_concat_1))
        # x_concat = self.batch_norm(x_concat)
        # x_concat = self.dropout(x_concat)

        x_concat = x_concat_2[:, :, x_concat_2.size(2) // 2, x_concat_2.size(3) // 2].unsqueeze(2).unsqueeze(3)

        x_concat = x_concat.permute(0, 2, 3, 1)
        attention_weights_1 = torch.sigmoid(self.fc1(x_concat))
        attention_weights_2 = torch.sigmoid(self.fc2(x_concat))
        attention_weights_3 = torch.sigmoid(self.fc3(x_concat))

        attention_weights_1 = attention_weights_1.permute(0, 3, 1, 2)
        attention_weights_2 = attention_weights_2.permute(0, 3, 1, 2)
        attention_weights_3 = attention_weights_3.permute(0, 3, 1, 2)

        return attention_weights_1, attention_weights_2, attention_weights_3